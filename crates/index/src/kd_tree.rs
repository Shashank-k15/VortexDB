use crate::{VectorIndex, distance};
use defs::{DbError, DenseVector, IndexedVector, PointId, Similarity};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    vec,
};
use uuid::Uuid;

pub struct KDTree {
    dim: usize,
    root: Option<Box<KDTreeNode>>,
    // In memory point ids, to check existence before O(n) deletion logic
    point_ids: HashSet<PointId>,
    // Rebuild tracking
    total_nodes: usize,
    deleted_count: usize,
}

// the node which will be the part of the KD Tree
pub struct KDTreeNode {
    indexed_vector: IndexedVector,
    left: Option<Box<KDTreeNode>>,
    right: Option<Box<KDTreeNode>>,
    is_deleted: bool,

    subtree_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Neighbor {
    id: PointId,
    distance: f32,
}

impl Eq for Neighbor {}

// Custom Ord implementation for the max-heap
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl KDTree {
    // Rebuild threshold
    const BALANCE_THRESHOLD: f32 = 0.7;
    const DELETE_REBUILD_RATIO: f32 = 0.25;

    // Build an empty index with no points
    pub fn build_empty(dim: usize) -> Self {
        KDTree {
            dim,
            root: None,
            point_ids: HashSet::new(),
            total_nodes: 0,
            deleted_count: 0,
        }
    }

    // Builds the vector index from provided vectors, there should atleast be single vector for dim calculation
    pub fn build(mut vectors: Vec<IndexedVector>) -> Result<Self, DbError> {
        if vectors.is_empty() {
            Err(DbError::IndexInitError)
        } else {
            let dim = vectors[0].vector.len();

            let mut point_ids = HashSet::with_capacity(vectors.len());
            for indexed_vector in vectors.iter() {
                point_ids.insert(indexed_vector.id);
            }

            let root_node = Self::build_recursive(&mut vectors, 0, dim);
            Ok(KDTree {
                dim,
                root: Some(root_node),
                point_ids,
                total_nodes: vectors.len(),
                deleted_count: 0,
            })
        }
    }

    // Builds the tree recursively with given vectors and returns the pointer of the root node
    pub fn build_recursive(
        vectors: &mut [IndexedVector],
        depth: usize,
        dim: usize,
    ) -> Box<KDTreeNode> {
        if vectors.is_empty() {
            panic!("Cannot build from an empty slice recursively");
        }

        let axis = depth % dim;
        let mid_idx = vectors.len() / 2;

        vectors.select_nth_unstable_by(mid_idx, |a, b| {
            let a_at_axis = a.vector[axis];
            let b_at_axis = b.vector[axis];
            a_at_axis.partial_cmp(&b_at_axis).unwrap_or(Ordering::Equal)
        });

        // Using swap so that we don't need to clone the whole vector
        let mut median_vec = IndexedVector {
            id: Uuid::new_v4(),
            vector: vec![],
        }; // dummy
        std::mem::swap(&mut vectors[mid_idx], &mut median_vec);

        let (left_points, right_points_with_median) = vectors.split_at_mut(mid_idx);
        let right_points = &mut right_points_with_median[1..]; // Exclude the swapped-out median

        let left = if left_points.is_empty() {
            None
        } else {
            Some(Self::build_recursive(left_points, depth + 1, dim))
        };

        let right = if right_points.is_empty() {
            None
        } else {
            Some(Self::build_recursive(right_points, depth + 1, dim))
        };

        let left_size = left.as_ref().map_or(0, |n| n.subtree_size);
        let right_size = right.as_ref().map_or(0, |n| n.subtree_size);

        Box::new(KDTreeNode {
            indexed_vector: median_vec,
            left,
            right,
            is_deleted: false,
            subtree_size: left_size + right_size + 1,
        })
    }

    pub fn insert_point(&mut self, new_vector: IndexedVector) {
        // Add to point_ids
        self.point_ids.insert(new_vector.id);
        self.total_nodes += 1;

        // use a traverse function to get the final leaf where this belongs
        if self.root.is_none() {
            self.root = Some(Box::new(KDTreeNode {
                indexed_vector: new_vector,
                left: None,
                right: None,
                is_deleted: false,
                subtree_size: 1,
            }));
            return;
        }

        let mut path: Vec<(usize, bool)> = Vec::new();
        let dim = self.dim;

        let mut current_link = &mut self.root;
        let mut depth = 0;
        // let dim = self.dim;

        while let Some(node_box) = current_link {
            let axis = depth % dim;
            let current_node = node_box.as_mut();

            current_node.subtree_size += 1;

            let va = new_vector.vector[axis];
            let vb = current_node.indexed_vector.vector[axis];

            let go_left = va <= vb;
            path.push((depth, go_left));

            if go_left {
                current_link = &mut current_node.left;
            } else {
                current_link = &mut current_node.right;
            }
            depth += 1;
        }

        // Assign the new node to current link which is &mut Option<Box<KDTreeNode>>
        let new_node = Box::new(KDTreeNode {
            indexed_vector: new_vector,
            left: None,
            right: None,
            is_deleted: false,
            subtree_size: 1,
        });

        *current_link = Some(new_node);

        self.check_and_rebalance(&path);
    }

    // Rebuild helper methods
    fn is_unbalanced(node: &KDTreeNode) -> bool {
        let left_size = node.left.as_ref().map_or(0, |n| n.subtree_size);
        let right_size = node.right.as_ref().map_or(0, |n| n.subtree_size);
        let max_child = left_size.max(right_size);

        max_child as f32 > Self::BALANCE_THRESHOLD * node.subtree_size as f32
    }

    fn collect_recursive(node: KDTreeNode, result: &mut Vec<IndexedVector>) {
        if !node.is_deleted {
            result.push(node.indexed_vector);
        }
        if let Some(left) = node.left {
            Self::collect_recursive(*left, result);
        }
        if let Some(right) = node.right {
            Self::collect_recursive(*right, result);
        }
    }

    fn collect_active_vectors(node: KDTreeNode) -> Vec<IndexedVector> {
        let mut result = Vec::with_capacity(node.subtree_size);
        Self::collect_recursive(node, &mut result);
        result
    }

    fn rebuild_at_depth(&mut self, path: &[(usize, bool)], target_depth: usize) {
        let dim = self.dim;

        // Navigate to parent of target node
        if target_depth == 0 {
            // Rebuild root
            if let Some(root) = self.root.take() {
                let old_size = root.subtree_size;
                let mut vectors = Self::collect_active_vectors(*root);
                let new_size = vectors.len();
                if !vectors.is_empty() {
                    self.root = Some(Self::build_recursive(&mut vectors, 0, dim));
                }
                // Update global counts as deleted nodes were purged
                self.total_nodes -= old_size - new_size;
                self.deleted_count = 0;
            }
        } else {
            // Navigate to target node
            let mut current_link = &mut self.root;
            for (_depth, go_left) in path.iter().take(target_depth) {
                let node = current_link.as_mut().unwrap();
                current_link = if *go_left {
                    &mut node.left
                } else {
                    &mut node.right
                };
            }

            // Rebuild tree at current link
            if let Some(subtree_root) = current_link.take() {
                let old_size = subtree_root.subtree_size;
                let mut vectors = Self::collect_active_vectors(*subtree_root);
                let new_size = vectors.len();

                if !vectors.is_empty() {
                    *current_link = Some(Self::build_recursive(&mut vectors, target_depth, dim));
                }

                // Only update ancestors if size changed (deleted nodes were purged)
                if old_size != new_size {
                    let size_diff = old_size - new_size;
                    self.subtract_size_from_ancestors(path, target_depth, size_diff);

                    self.total_nodes -= size_diff;
                    self.deleted_count = self.deleted_count.saturating_sub(size_diff);
                }
            }
        }
    }

    fn subtract_size_from_ancestors(
        &mut self,
        path: &[(usize, bool)],
        up_to_depth: usize,
        diff: usize,
    ) {
        let mut current = &mut self.root;
        for (_, go_left) in path.iter().take(up_to_depth) {
            if let Some(node) = current {
                node.subtree_size -= diff;
                current = if *go_left {
                    &mut node.left
                } else {
                    &mut node.right
                };
            }
        }
    }

    fn check_and_rebalance(&mut self, path: &[(usize, bool)]) {
        // Find the shallowest (closest to root) depth where imbalance occurs
        // so that rebuilding fixes the largest unbalanced subtree
        let mut unbalanced_depth: Option<usize> = None;

        let mut current = self.root.as_ref();

        // Check root first (depth 0)
        if let Some(node) = current
            && Self::is_unbalanced(node)
        {
            unbalanced_depth = Some(0);
        }

        // Then traverse the path and check each node
        // Once we find the shallowest unbalanced node, break immediately
        for (idx, (_depth, go_left)) in path.iter().enumerate() {
            if unbalanced_depth.is_some() {
                break;
            }

            if let Some(node) = current {
                current = if *go_left {
                    node.left.as_ref()
                } else {
                    node.right.as_ref()
                };

                // Check the child node we just moved to (at depth idx + 1)
                if let Some(child) = current
                    && Self::is_unbalanced(child)
                {
                    unbalanced_depth = Some(idx + 1);
                    break;
                }
            }
        }

        if let Some(target_depth) = unbalanced_depth {
            self.rebuild_at_depth(path, target_depth);
        }
    }

    fn should_rebuild_global(&self) -> bool {
        self.total_nodes > 0
            && (self.deleted_count as f32 / self.total_nodes as f32) > Self::DELETE_REBUILD_RATIO
    }

    // Returns true if point found and deleted, else false
    pub fn delete_point(&mut self, point_id: &PointId) -> bool {
        if self.point_ids.contains(point_id) {
            let deleted = Self::find_and_mark_deleted(&mut self.root, *point_id);
            if deleted {
                self.deleted_count += 1;
                self.point_ids.remove(point_id);
            }

            if Self::should_rebuild_global(self)
                && let Some(root) = self.root.take()
            {
                let mut vectors = Self::collect_active_vectors(*root);
                if !vectors.is_empty() {
                    self.root = Some(Self::build_recursive(&mut vectors, 0, self.dim));
                }

                self.total_nodes = vectors.len();
                self.deleted_count = 0;
            }

            return deleted;
        }
        false
    }

    fn find_and_mark_deleted(node_opt: &mut Option<Box<KDTreeNode>>, target_id: PointId) -> bool {
        if let Some(node) = node_opt {
            if node.indexed_vector.id == target_id {
                node.is_deleted = true;
                return true;
            }

            // Search left first then right
            Self::find_and_mark_deleted(&mut node.left, target_id)
                || Self::find_and_mark_deleted(&mut node.right, target_id)
        } else {
            false
        }
    }

    pub fn search_top_k(
        &self,
        query_vector: DenseVector,
        k: usize,
        dist_type: Similarity,
    ) -> Vec<(PointId, f32)> {
        //Searches for top k closest vectors according to specified metric

        if self.root.is_none() || k == 0 {
            return Vec::new();
        }

        let mut best_neighbours = BinaryHeap::with_capacity(k);

        self.search_recursive(
            &self.root,
            &query_vector,
            k,
            &mut best_neighbours,
            0,
            dist_type,
        );

        best_neighbours
            .into_sorted_vec()
            .iter()
            .map(|neighbor| (neighbor.id, neighbor.distance))
            .collect()
    }

    fn search_recursive(
        &self,
        node_opt: &Option<Box<KDTreeNode>>,
        query_vector: &DenseVector,
        k: usize,
        heap: &mut BinaryHeap<Neighbor>,
        depth: usize,
        dist_type: Similarity,
    ) {
        // Base case is that we hit a leaf node don't do anything
        if let Some(node) = node_opt {
            let axis = depth % self.dim;

            let (near_side, far_side) = if query_vector[axis] <= node.indexed_vector.vector[axis] {
                (&node.left, &node.right)
            } else {
                (&node.right, &node.left)
            };

            // Recurse on near side first
            self.search_recursive(near_side, query_vector, k, heap, depth + 1, dist_type);

            // Process the current node
            if !node.is_deleted {
                // TODO: Possible overhead, here heap stores sqrt euclidean distance, we can eliminate that by storing squared distances in case of euclidean
                let distance = distance(query_vector, &node.indexed_vector.vector, dist_type);
                if heap.len() < k {
                    heap.push(Neighbor {
                        id: node.indexed_vector.id,
                        distance,
                    });
                } else if distance < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(Neighbor {
                        id: node.indexed_vector.id,
                        distance,
                    });
                }
            }

            // Pruning on the farther side to check if there are better candidates
            // For Euclidean: the heap stores sqrt distances, so we compare axis_diff with the heap's max distance
            // For Manhattan: direct comparison works since it's a sum of absolute differences
            let axis_diff = (query_vector[axis] - node.indexed_vector.vector[axis]).abs();
            let should_search_far = match dist_type {
                Similarity::Euclidean | Similarity::Manhattan => {
                    heap.len() < k || axis_diff < heap.peek().unwrap().distance
                }
                _ => true, // Cosine/Hamming - no effective pruning, always search
            };

            if should_search_far {
                self.search_recursive(far_side, query_vector, k, heap, depth + 1, dist_type);
            }
        }
    }
}

impl VectorIndex for KDTree {
    fn insert(&mut self, vector: IndexedVector) -> Result<(), DbError> {
        self.insert_point(vector);
        Ok(())
    }

    fn delete(&mut self, point_id: PointId) -> Result<bool, DbError> {
        Ok(self.delete_point(&point_id))
    }

    fn search(
        &self,
        query_vector: DenseVector,
        similarity: Similarity,
        k: usize,
    ) -> Result<Vec<PointId>, DbError> {
        if matches!(similarity, Similarity::Cosine | Similarity::Hamming) {
            return Err(DbError::UnsupportedSimilarity);
        }

        let results = self.search_top_k(query_vector, k, similarity);
        Ok(results.into_iter().map(|(id, _)| id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(vector: Vec<f32>) -> IndexedVector {
        IndexedVector {
            id: Uuid::new_v4(),
            vector,
        }
    }

    fn make_vector_with_id(id: Uuid, vector: Vec<f32>) -> IndexedVector {
        IndexedVector { id, vector }
    }

    // Build Tests

    #[test]
    fn test_build_empty() {
        let tree = KDTree::build_empty(3);
        assert!(tree.root.is_none());
        assert_eq!(tree.dim, 3);
        assert_eq!(tree.total_nodes, 0);
        assert!(tree.point_ids.is_empty());
    }

    #[test]
    fn test_build_with_empty_vectors_returns_error() {
        let result = KDTree::build(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_single_vector() {
        let id = Uuid::new_v4();
        let vectors = vec![make_vector_with_id(id, vec![1.0, 2.0, 3.0])];
        let tree = KDTree::build(vectors).unwrap();

        assert!(tree.root.is_some());
        assert_eq!(tree.dim, 3);
        assert_eq!(tree.total_nodes, 1);
        assert!(tree.point_ids.contains(&id));
    }

    #[test]
    fn test_build_multiple_vectors() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![1.0, 2.0]),
            make_vector_with_id(id2, vec![3.0, 4.0]),
            make_vector_with_id(id3, vec![5.0, 6.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        assert!(tree.root.is_some());
        assert_eq!(tree.dim, 2);
        assert_eq!(tree.total_nodes, 3);
        assert!(tree.point_ids.contains(&id1));
        assert!(tree.point_ids.contains(&id2));
        assert!(tree.point_ids.contains(&id3));
    }

    // Insert Tests

    #[test]
    fn test_insert_into_empty_tree() {
        let mut tree = KDTree::build_empty(2);
        let id = Uuid::new_v4();
        let vector = make_vector_with_id(id, vec![1.0, 2.0]);

        let result = tree.insert(vector);
        assert!(result.is_ok());
        assert_eq!(tree.total_nodes, 1);
        assert!(tree.point_ids.contains(&id));
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_insert_multiple_vectors() {
        let mut tree = KDTree::build_empty(2);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        tree.insert(make_vector_with_id(id1, vec![1.0, 2.0]))
            .unwrap();
        tree.insert(make_vector_with_id(id2, vec![3.0, 4.0]))
            .unwrap();
        tree.insert(make_vector_with_id(id3, vec![5.0, 6.0]))
            .unwrap();

        assert_eq!(tree.total_nodes, 3);
        assert!(tree.point_ids.contains(&id1));
        assert!(tree.point_ids.contains(&id2));
        assert!(tree.point_ids.contains(&id3));
    }

    // Delete Tests

    #[test]
    fn test_delete_existing_point() {
        let mut ids = Vec::new();
        let mut vectors = Vec::new();

        // Create enough vectors so deleting one doesn't trigger global rebuild
        for i in 0..10 {
            let id = Uuid::new_v4();
            ids.push(id);
            vectors.push(make_vector_with_id(id, vec![i as f32, i as f32]));
        }

        let mut tree = KDTree::build(vectors).unwrap();

        let result = tree.delete(ids[0]).unwrap();
        assert!(result);
        assert!(!tree.point_ids.contains(&ids[0]));
        assert_eq!(tree.deleted_count, 1);
    }

    #[test]
    fn test_delete_non_existing_point() {
        let id1 = Uuid::new_v4();
        let vectors = vec![make_vector_with_id(id1, vec![1.0, 2.0])];
        let mut tree = KDTree::build(vectors).unwrap();

        let non_existing_id = Uuid::new_v4();
        let result = tree.delete(non_existing_id).unwrap();
        assert!(!result);
        assert_eq!(tree.deleted_count, 0);
    }

    #[test]
    fn test_delete_from_empty_tree() {
        let mut tree = KDTree::build_empty(2);
        let result = tree.delete(Uuid::new_v4()).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_deleted_point_not_in_search_results() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![0.0, 0.0]),
            make_vector_with_id(id2, vec![1.0, 1.0]),
            make_vector_with_id(id3, vec![10.0, 10.0]),
        ];
        let mut tree = KDTree::build(vectors).unwrap();

        // Delete the closest point
        tree.delete(id1).unwrap();

        // Search should not return the deleted point
        let results = tree
            .search(vec![0.0, 0.0], Similarity::Euclidean, 2)
            .unwrap();
        assert!(!results.contains(&id1));
        assert!(results.contains(&id2));
    }

    // Search Tests (VectorIndex trait)

    #[test]
    fn test_search_empty_tree() {
        let tree = KDTree::build_empty(2);
        let results = tree
            .search(vec![1.0, 2.0], Similarity::Euclidean, 5)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_euclidean() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![1.0, 1.0]),
            make_vector_with_id(id2, vec![2.0, 2.0]),
            make_vector_with_id(id3, vec![10.0, 10.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![0.0, 0.0], Similarity::Euclidean, 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], id1); // Closest
        assert_eq!(results[1], id2); // Second closest
    }

    #[test]
    fn test_search_manhattan() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![1.0, 1.0]),
            make_vector_with_id(id2, vec![2.0, 2.0]),
            make_vector_with_id(id3, vec![5.0, 5.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![0.0, 0.0], Similarity::Manhattan, 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], id1);
        assert_eq!(results[1], id2);
    }

    #[test]
    fn test_search_unsupported_similarity_cosine() {
        let vectors = vec![make_vector(vec![1.0, 2.0])];
        let tree = KDTree::build(vectors).unwrap();

        let result = tree.search(vec![1.0, 2.0], Similarity::Cosine, 1);
        assert!(matches!(result, Err(DbError::UnsupportedSimilarity)));
    }

    #[test]
    fn test_search_unsupported_similarity_hamming() {
        let vectors = vec![make_vector(vec![1.0, 2.0])];
        let tree = KDTree::build(vectors).unwrap();

        let result = tree.search(vec![1.0, 2.0], Similarity::Hamming, 1);
        assert!(matches!(result, Err(DbError::UnsupportedSimilarity)));
    }

    #[test]
    fn test_search_k_larger_than_tree_size() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![1.0, 1.0]),
            make_vector_with_id(id2, vec![2.0, 2.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![0.0, 0.0], Similarity::Euclidean, 10)
            .unwrap();
        assert_eq!(results.len(), 2); // Should return all available points
    }

    #[test]
    fn test_search_exact_match() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![5.0, 5.0]),
            make_vector_with_id(id2, vec![10.0, 10.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![5.0, 5.0], Similarity::Euclidean, 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], id1);
    }

    // Search Correctness Tests

    #[test]
    fn test_search_correctness_3d() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let id4 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![0.0, 0.0, 0.0]),
            make_vector_with_id(id2, vec![1.0, 1.0, 1.0]),
            make_vector_with_id(id3, vec![2.0, 2.0, 2.0]),
            make_vector_with_id(id4, vec![10.0, 10.0, 10.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![0.5, 0.5, 0.5], Similarity::Euclidean, 2)
            .unwrap();
        // id1 at distance sqrt(0.75) ≈ 0.866
        // id2 at distance sqrt(0.75) ≈ 0.866
        // Both are equidistant, should return both
        assert_eq!(results.len(), 2);
        assert!(results.contains(&id1) || results.contains(&id2));
    }

    #[test]
    fn test_search_after_insert() {
        let mut tree = KDTree::build_empty(2);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        tree.insert(make_vector_with_id(id1, vec![10.0, 10.0]))
            .unwrap();
        tree.insert(make_vector_with_id(id2, vec![1.0, 1.0]))
            .unwrap();
        tree.insert(make_vector_with_id(id3, vec![5.0, 5.0]))
            .unwrap();

        let results = tree
            .search(vec![0.0, 0.0], Similarity::Euclidean, 2)
            .unwrap();
        assert_eq!(results[0], id2); // Closest to origin
        assert_eq!(results[1], id3); // Second closest
    }

    #[test]
    fn test_search_high_dimensional() {
        let dim = 10;
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let vectors = vec![
            make_vector_with_id(id1, vec![0.0; dim]),
            make_vector_with_id(id2, vec![1.0; dim]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let query = vec![0.1; dim];
        let results = tree.search(query, Similarity::Euclidean, 1).unwrap();
        assert_eq!(results[0], id1); // Closer to all-zeros
    }

    // Rebalancing Tests

    #[test]
    fn test_many_inserts_maintains_searchability() {
        let mut tree = KDTree::build_empty(2);
        let mut ids = Vec::new();

        // Insert many points that would cause imbalance
        for i in 0..20 {
            let id = Uuid::new_v4();
            ids.push(id);
            tree.insert(make_vector_with_id(id, vec![i as f32, i as f32]))
                .unwrap();
        }

        // Search should still work correctly
        let results = tree
            .search(vec![0.0, 0.0], Similarity::Euclidean, 5)
            .unwrap();
        assert_eq!(results.len(), 5);
        // First result should be the point at (0, 0)
        assert_eq!(results[0], ids[0]);
    }

    #[test]
    fn test_delete_triggers_rebuild() {
        let mut ids = Vec::new();
        let mut vectors = Vec::new();

        for i in 0..10 {
            let id = Uuid::new_v4();
            ids.push(id);
            vectors.push(make_vector_with_id(id, vec![i as f32, i as f32]));
        }

        let mut tree = KDTree::build(vectors).unwrap();

        // Delete enough points to trigger rebuild (> 25%)
        for id in ids.iter().take(3) {
            tree.delete(*id).unwrap();
        }

        // Tree should still function correctly
        let results = tree
            .search(vec![5.0, 5.0], Similarity::Euclidean, 3)
            .unwrap();
        assert_eq!(results.len(), 3);
        // Deleted points should not appear
        for id in ids.iter().take(3) {
            assert!(!results.contains(id));
        }
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_single_point_search() {
        let id = Uuid::new_v4();
        let vectors = vec![make_vector_with_id(id, vec![5.0, 5.0])];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![0.0, 0.0], Similarity::Euclidean, 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], id);
    }

    #[test]
    fn test_duplicate_coordinates() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![1.0, 1.0]),
            make_vector_with_id(id2, vec![1.0, 1.0]), // Same coordinates
            make_vector_with_id(id3, vec![2.0, 2.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![1.0, 1.0], Similarity::Euclidean, 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        // Both id1 and id2 should be in results (both at distance 0)
        assert!(results.contains(&id1) || results.contains(&id2));
    }

    #[test]
    fn test_negative_coordinates() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let vectors = vec![
            make_vector_with_id(id1, vec![-1.0, -1.0]),
            make_vector_with_id(id2, vec![1.0, 1.0]),
        ];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![-0.5, -0.5], Similarity::Euclidean, 1)
            .unwrap();
        assert_eq!(results[0], id1);
    }

    #[test]
    fn test_search_with_zero_k() {
        let vectors = vec![make_vector(vec![1.0, 2.0])];
        let tree = KDTree::build(vectors).unwrap();

        let results = tree
            .search(vec![1.0, 2.0], Similarity::Euclidean, 0)
            .unwrap();
        assert!(results.is_empty());
    }
}
