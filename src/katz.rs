//! Katz centrality.
//!
//! Computes Katz centrality which measures the influence of a vertex
//! considering all walks from the vertex, with exponentially decreasing weight.

use crate::Graph;

/// Compute Katz centrality.
///
/// `alpha` is the attenuation factor (must be less than `1/lambda_max` where `lambda_max`
/// is the largest eigenvalue of the adjacency matrix). `beta` is a constant added to
/// each vertex's score.
pub fn katz_centrality(graph: &Graph, alpha: f64, beta: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = graph.vertex_count();
    if n == 0 {
        return vec![];
    }

    let mut x = vec![beta; n];

    for _ in 0..max_iter {
        let mut new_x = vec![beta; n];
        #[allow(clippy::needless_range_loop)]
        for u in 0..n {
            for &v in graph.neighbors(u) {
                new_x[u] += alpha * x[v];
            }
        }

        let diff: f64 = new_x
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        x = new_x;

        if diff < tol {
            break;
        }
    }

    x
}

/// Compute normalized Katz centrality (vector norm = 1).
pub fn normalized_katz(graph: &Graph, alpha: f64, beta: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let kc = katz_centrality(graph, alpha, beta, max_iter, tol);
    let norm: f64 = kc.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return kc;
    }
    kc.iter().map(|c| c / norm).collect()
}

/// Find the vertex with the highest Katz centrality.
pub fn most_central_vertex(graph: &Graph, alpha: f64, beta: f64) -> usize {
    let kc = katz_centrality(graph, alpha, beta, 1000, 1e-10);
    kc.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Estimate a safe alpha value (1 / (max_degree + 1)).
pub fn safe_alpha(graph: &Graph) -> f64 {
    let max_deg = (0..graph.vertex_count())
        .map(|v| graph.out_degree(v))
        .max()
        .unwrap_or(1);
    1.0 / (max_deg + 1) as f64
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
        let kc = katz_centrality(&g, 0.1, 1.0, 1000, 1e-10);
        assert!(kc[0] > kc[1], "center should have higher Katz centrality");
    }

    #[test]
    fn test_path_graph() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let kc = katz_centrality(&g, 0.1, 1.0, 1000, 1e-10);
        assert_eq!(kc.len(), 4);
        // All should be positive
        for c in &kc {
            assert!(*c > 0.0);
        }
    }

    #[test]
    fn test_normalized() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let nk = normalized_katz(&g, 0.1, 1.0, 1000, 1e-10);
        let norm: f64 = nk.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_safe_alpha() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(0, 3);
        let alpha = safe_alpha(&g);
        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_empty() {
        let g = Graph::new(0);
        let kc = katz_centrality(&g, 0.1, 1.0, 100, 1e-10);
        assert!(kc.is_empty());
    }

    #[test]
    fn test_most_central() {
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i);
        }
        assert_eq!(most_central_vertex(&g, 0.1, 1.0), 0);
    }
}
