//! Eigenvector centrality via power iteration.
//!
//! Computes eigenvector centrality where a vertex's score is proportional
//! to the sum of its neighbors' scores.

use crate::Graph;

/// Compute eigenvector centrality using power iteration.
///
/// Returns `(centrality_values, dominant_eigenvalue)`.
pub fn eigenvector_centrality(graph: &Graph, max_iter: usize, tol: f64) -> (Vec<f64>, f64) {
    let n = graph.vertex_count();
    if n == 0 {
        return (vec![], 0.0);
    }

    let mut x = vec![1.0; n];
    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // Multiply by adjacency matrix
        let mut new_x = vec![0.0; n];
        #[allow(clippy::needless_range_loop)]
        for u in 0..n {
            for &v in graph.neighbors(u) {
                new_x[u] += x[v];
            }
        }

        let new_eigenvalue = new_x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if new_eigenvalue > 1e-15 {
            for v in new_x.iter_mut() {
                *v /= new_eigenvalue;
            }
        }

        let diff = new_x
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        eigenvalue = new_eigenvalue;
        x = new_x;

        if diff < tol {
            break;
        }
    }

    // Ensure positive values (flip sign if needed)
    if x.iter().sum::<f64>() < 0.0 {
        for v in x.iter_mut() {
            *v = -*v;
        }
    }

    (x, eigenvalue)
}

/// Compute normalized eigenvector centrality (values sum to 1).
pub fn normalized_eigenvector(graph: &Graph, max_iter: usize, tol: f64) -> Vec<f64> {
    let (cent, _) = eigenvector_centrality(graph, max_iter, tol);
    let sum: f64 = cent.iter().sum();
    if sum.abs() < 1e-15 {
        return cent;
    }
    cent.iter().map(|c| c / sum).collect()
}

/// Find the vertex with the highest eigenvector centrality.
pub fn most_central_vertex(graph: &Graph) -> usize {
    let (cent, _) = eigenvector_centrality(graph, 1000, 1e-10);
    cent.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_star_center() {
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i);
        }
        let (cent, _) = eigenvector_centrality(&g, 1000, 1e-10);
        // Center should have highest absolute centrality
        let max_abs = cent.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        assert!((cent[0].abs() - max_abs).abs() < 1e-6, "center abs={}, max abs={}", cent[0].abs(), max_abs);
    }

    #[test]
    fn test_complete_equal() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j);
            }
        }
        let (cent, _) = eigenvector_centrality(&g, 1000, 1e-10);
        for c in &cent[1..] {
            assert!((c - cent[0]).abs() < 0.1);
        }
    }

    #[test]
    fn test_normalized() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let norm = normalized_eigenvector(&g, 1000, 1e-10);
        let sum: f64 = norm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_most_central() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3);
        g.add_edge(1, 2);
        // Vertex 0 has degree 3, vertex 1 has degree 2, rest have degree 1-2
        let (cent, _) = eigenvector_centrality(&g, 1000, 1e-10);
        let max_idx = cent.iter().enumerate().max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
        assert_eq!(max_idx, 0, "expected 0, got {max_idx}: {:?}", cent);
    }

    #[test]
    fn test_empty() {
        let g = Graph::new(0);
        let (cent, _) = eigenvector_centrality(&g, 100, 1e-10);
        assert!(cent.is_empty());
    }
}
