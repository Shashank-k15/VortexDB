use super::index::KDTree;
use crate::VectorIndex;
use crate::distance;
use crate::flat::FlatIndex;
use defs::{DbError, IndexedVector, Similarity};
use std::collections::HashSet;
use uuid::Uuid;

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

// Edge Cases

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

// Comparison Tests: KDTree vs FlatIndex

/// Helper to create a fixed set of 10 vectors with known UUIDs for comparison tests
fn create_test_vectors_2d() -> Vec<IndexedVector> {
    let ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
    vec![
        make_vector_with_id(ids[0], vec![0.5, 0.5]),
        make_vector_with_id(ids[1], vec![2.3, 1.7]),
        make_vector_with_id(ids[2], vec![-1.0, 3.0]),
        make_vector_with_id(ids[3], vec![4.5, -2.0]),
        make_vector_with_id(ids[4], vec![7.0, 7.0]),
        make_vector_with_id(ids[5], vec![-3.5, -1.5]),
        make_vector_with_id(ids[6], vec![1.0, 5.0]),
        make_vector_with_id(ids[7], vec![6.0, 2.0]),
        make_vector_with_id(ids[8], vec![-2.0, -4.0]),
        make_vector_with_id(ids[9], vec![3.0, 3.0]),
    ]
}

fn create_test_vectors_3d() -> Vec<IndexedVector> {
    let ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
    vec![
        make_vector_with_id(ids[0], vec![1.0, 2.0, 3.0]),
        make_vector_with_id(ids[1], vec![-1.5, 0.5, 2.0]),
        make_vector_with_id(ids[2], vec![4.0, 4.0, 4.0]),
        make_vector_with_id(ids[3], vec![0.0, 0.0, 0.0]),
        make_vector_with_id(ids[4], vec![2.5, -1.0, 3.5]),
        make_vector_with_id(ids[5], vec![-2.0, 3.0, -1.0]),
        make_vector_with_id(ids[6], vec![5.0, 1.0, 2.0]),
        make_vector_with_id(ids[7], vec![3.0, 3.0, 3.0]),
        make_vector_with_id(ids[8], vec![-0.5, -0.5, 1.0]),
        make_vector_with_id(ids[9], vec![1.5, 2.5, 0.5]),
    ]
}

/// Helper to verify that two result sets are valid k-nearest neighbor results
/// Both should return the k closest points (by distance), but may differ on tie-breaking
fn verify_same_results(
    tree_results: &[Uuid],
    flat_results: &[Uuid],
    vectors: &[IndexedVector],
    query: &[f32],
    similarity: Similarity,
    k: usize,
) {
    // Same length
    assert_eq!(
        tree_results.len(),
        flat_results.len(),
        "Result lengths differ"
    );

    // Both should return at most k results
    assert!(tree_results.len() <= k);

    // Get distances for all results
    let query_vec = query.to_vec();
    let get_distance = |id: &Uuid| -> f32 {
        let vec = vectors.iter().find(|v| v.id == *id).unwrap();
        distance(&vec.vector, &query_vec, similarity)
    };

    // Verify tree results are sorted by distance
    for i in 1..tree_results.len() {
        let d1 = get_distance(&tree_results[i - 1]);
        let d2 = get_distance(&tree_results[i]);
        assert!(
            d1 <= d2 + 1e-6,
            "KDTree results not sorted: {} > {}",
            d1,
            d2
        );
    }

    // Verify flat results are sorted by distance
    for i in 1..flat_results.len() {
        let d1 = get_distance(&flat_results[i - 1]);
        let d2 = get_distance(&flat_results[i]);
        assert!(d1 <= d2 + 1e-6, "Flat results not sorted: {} > {}", d1, d2);
    }

    // The maximum distance in both result sets should be the same (k-th nearest distance)
    if !tree_results.is_empty() {
        let tree_max_dist = get_distance(tree_results.last().unwrap());
        let flat_max_dist = get_distance(flat_results.last().unwrap());
        assert!(
            (tree_max_dist - flat_max_dist).abs() < 1e-6,
            "Max distances differ: tree={}, flat={}",
            tree_max_dist,
            flat_max_dist
        );
    }

    // Verify that for each result in tree_results, either:
    // 1. It's also in flat_results, OR
    // 2. It has the same distance as the last element (tie-breaking difference)
    let flat_set: HashSet<_> = flat_results.iter().collect();
    let flat_max_dist = if flat_results.is_empty() {
        0.0
    } else {
        get_distance(flat_results.last().unwrap())
    };

    for id in tree_results {
        if !flat_set.contains(id) {
            // This ID is not in flat results, verify it's a tie
            let dist = get_distance(id);
            assert!(
                (dist - flat_max_dist).abs() < 1e-6,
                "KDTree returned {:?} with distance {} but it's not in flat results and not a tie (flat max: {})",
                id,
                dist,
                flat_max_dist
            );
        }
    }

    // Similarly verify flat_results
    let tree_set: HashSet<_> = tree_results.iter().collect();
    let tree_max_dist = if tree_results.is_empty() {
        0.0
    } else {
        get_distance(tree_results.last().unwrap())
    };

    for id in flat_results {
        if !tree_set.contains(id) {
            let dist = get_distance(id);
            assert!(
                (dist - tree_max_dist).abs() < 1e-6,
                "Flat returned {:?} with distance {} but it's not in tree results and not a tie (tree max: {})",
                id,
                dist,
                tree_max_dist
            );
        }
    }
}

#[test]
fn test_kdtree_vs_flat_euclidean_2d() {
    let vectors = create_test_vectors_2d();
    let tree = KDTree::build(vectors.clone()).unwrap();
    let flat = FlatIndex::build(vectors.clone());

    // Test multiple query points and different k values
    let queries = vec![
        vec![0.0, 0.0],
        vec![3.0, 3.0],
        vec![-1.0, 2.0],
        vec![5.0, 5.0],
    ];

    for query in queries {
        for k in [1, 3, 5, 10] {
            let tree_results = tree
                .search(query.clone(), Similarity::Euclidean, k)
                .unwrap();
            let flat_results = flat
                .search(query.clone(), Similarity::Euclidean, k)
                .unwrap();

            verify_same_results(
                &tree_results,
                &flat_results,
                &vectors,
                &query,
                Similarity::Euclidean,
                k,
            );
        }
    }
}

#[test]
fn test_kdtree_vs_flat_euclidean_3d() {
    let vectors = create_test_vectors_3d();
    let tree = KDTree::build(vectors.clone()).unwrap();
    let flat = FlatIndex::build(vectors.clone());

    let queries = vec![
        vec![0.0, 0.0, 0.0],
        vec![2.0, 2.0, 2.0],
        vec![-1.0, 1.0, 1.0],
        vec![4.0, 3.0, 3.0],
    ];

    for query in queries {
        for k in [1, 3, 5, 10] {
            let tree_results = tree
                .search(query.clone(), Similarity::Euclidean, k)
                .unwrap();
            let flat_results = flat
                .search(query.clone(), Similarity::Euclidean, k)
                .unwrap();

            verify_same_results(
                &tree_results,
                &flat_results,
                &vectors,
                &query,
                Similarity::Euclidean,
                k,
            );
        }
    }
}

#[test]
fn test_kdtree_vs_flat_euclidean_5d() {
    // Test with higher dimensionality
    let ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
    let vectors = vec![
        make_vector_with_id(ids[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        make_vector_with_id(ids[1], vec![-1.0, 0.0, 1.0, 2.0, 3.0]),
        make_vector_with_id(ids[2], vec![5.0, 4.0, 3.0, 2.0, 1.0]),
        make_vector_with_id(ids[3], vec![0.0, 0.0, 0.0, 0.0, 0.0]),
        make_vector_with_id(ids[4], vec![2.5, 2.5, 2.5, 2.5, 2.5]),
        make_vector_with_id(ids[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]),
        make_vector_with_id(ids[6], vec![3.0, 3.0, 3.0, 3.0, 3.0]),
        make_vector_with_id(ids[7], vec![1.0, 1.0, 1.0, 1.0, 1.0]),
        make_vector_with_id(ids[8], vec![4.0, 0.0, -1.0, 2.0, 5.0]),
        make_vector_with_id(ids[9], vec![-0.5, 1.5, 2.5, 3.5, 4.5]),
    ];

    let tree = KDTree::build(vectors.clone()).unwrap();
    let flat = FlatIndex::build(vectors.clone());

    let queries = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![2.0, 2.0, 2.0, 2.0, 2.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![-1.0, -1.0, 0.0, 1.0, 1.0],
    ];

    for query in queries {
        for k in [1, 3, 5, 10] {
            let tree_results = tree
                .search(query.clone(), Similarity::Euclidean, k)
                .unwrap();
            let flat_results = flat
                .search(query.clone(), Similarity::Euclidean, k)
                .unwrap();

            verify_same_results(
                &tree_results,
                &flat_results,
                &vectors,
                &query,
                Similarity::Euclidean,
                k,
            );
        }
    }
}
