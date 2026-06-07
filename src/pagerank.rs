//! PageRank algorithm.
//!
//! Computes PageRank scores using the power iteration method with
//! damping factor for random surfer model.

use crate::Graph;

/// Compute PageRank using power iteration.
///
/// `damping` is the probability of following a link (typically 0.85).
pub fn pagerank(graph: &Graph, damping: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = graph.vertex_count();
    if n == 0 {
        return vec![];
    }

    let mut pr = vec![1.0 / n as f64; n];
    let out_degree: Vec<usize> = (0..n).map(|v| graph.out_degree(v)).collect();

    for _ in 0..max_iter {
        let mut new_pr = vec![(1.0 - damping) / n as f64; n];

        for u in 0..n {
            if out_degree[u] > 0 {
                let share = damping * pr[u] / out_degree[u] as f64;
                for &v in graph.neighbors(u) {
                    new_pr[v] += share;
                }
            } else {
                // Dangling node: distribute to all
                let share = damping * pr[u] / n as f64;
                #[allow(clippy::needless_range_loop)]
                for v in 0..n {
                    new_pr[v] += share;
                }
            }
        }

        let diff: f64 = new_pr
            .iter()
            .zip(pr.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        pr = new_pr;

        if diff < tol {
            break;
        }
    }

    pr
}

/// Compute personalized PageRank with a preference vector.
///
/// `preference[v]` is the probability of jumping to vertex `v` (should sum to 1).
pub fn personalized_pagerank(
    graph: &Graph,
    damping: f64,
    preference: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = graph.vertex_count();
    if n == 0 {
        return vec![];
    }

    let mut pr = preference.to_vec();
    let out_degree: Vec<usize> = (0..n).map(|v| graph.out_degree(v)).collect();

    for _ in 0..max_iter {
        let mut new_pr: Vec<f64> = preference.iter().map(|&p| (1.0 - damping) * p).collect();

        for u in 0..n {
            if out_degree[u] > 0 {
                let share = damping * pr[u] / out_degree[u] as f64;
                for &v in graph.neighbors(u) {
                    new_pr[v] += share;
                }
            } else {
                let share = damping * pr[u];
                for (v, pref) in preference.iter().enumerate() {
                    new_pr[v] += share * pref;
                }
            }
        }

        let diff: f64 = new_pr
            .iter()
            .zip(pr.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        pr = new_pr;

        if diff < tol {
            break;
        }
    }

    pr
}

/// Find the vertex with the highest PageRank.
pub fn most_central_vertex(graph: &Graph) -> usize {
    let pr = pagerank(graph, 0.85, 1000, 1e-10);
    pr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute the PageRank convergence rate.
pub fn pagerank_convergence_rate(damping: f64) -> f64 {
    damping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_pagerank_simple() {
        let mut g = Graph::new_directed(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        let pr = pagerank(&g, 0.85, 1000, 1e-10);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pagerank_star_directed() {
        let mut g = Graph::new_directed(4);
        g.add_edge(1, 0);
        g.add_edge(2, 0);
        g.add_edge(3, 0);
        let pr = pagerank(&g, 0.85, 1000, 1e-10);
        assert!(pr[0] > pr[1]);
        assert!(pr[0] > pr[2]);
    }

    #[test]
    fn test_pagerank_undirected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let pr = pagerank(&g, 0.85, 1000, 1e-10);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_personalized_pagerank() {
        let mut g = Graph::new_directed(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        let pref = vec![1.0, 0.0, 0.0];
        let pr = personalized_pagerank(&g, 0.85, &pref, 1000, 1e-10);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(0);
        let pr = pagerank(&g, 0.85, 100, 1e-10);
        assert!(pr.is_empty());
    }

    #[test]
    fn test_damping_zero() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let pr = pagerank(&g, 0.0, 100, 1e-10);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // With damping=0, all vertices have equal PageRank
        for c in &pr {
            assert!((c - 1.0 / 3.0).abs() < 1e-6);
        }
    }
}
