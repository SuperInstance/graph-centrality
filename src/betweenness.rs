//! Betweenness centrality using Brandes' algorithm.
//!
//! Computes betweenness centrality for all vertices in O(VE) time
//! for unweighted graphs.

use crate::Graph;
use std::collections::VecDeque;

/// Compute betweenness centrality for all vertices using Brandes' algorithm.
///
/// Returns a vector where `result[v]` is the betweenness centrality of vertex `v`.
pub fn betweenness_centrality(graph: &Graph) -> Vec<f64> {
    let n = graph.vertex_count();
    let mut cb = vec![0.0; n];

    for s in 0..n {
        let mut stack = Vec::new();
        let mut predecessors = vec![vec![]; n];
        let mut sigma = vec![0usize; n];
        sigma[s] = 1;
        let mut dist = vec![None::<usize>; n];
        dist[s] = Some(0);
        let mut queue = VecDeque::new();
        queue.push_back(s);

        // BFS from s
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let dv = dist[v].unwrap_or(0);
            for &w in graph.neighbors(v) {
                if dist[w].is_none() {
                    dist[w] = Some(dv + 1);
                    queue.push_back(w);
                }
                if dist[w] == Some(dv + 1) {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Back-propagation
        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                if sigma[w] > 0 {
                    delta[v] += (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w]);
                }
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // For undirected graphs, divide by 2
    if !graph.is_directed() {
        for c in &mut cb {
            *c /= 2.0;
        }
    }

    cb
}

/// Find the vertex with the highest betweenness centrality.
pub fn most_central_vertex(graph: &Graph) -> usize {
    let bc = betweenness_centrality(graph);
    bc.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute edge betweenness centrality.
///
/// Returns edge betweenness as a flat map of (u, v) -> centrality.
pub fn edge_betweenness(graph: &Graph) -> Vec<((usize, usize), f64)> {
    let n = graph.vertex_count();
    let mut edge_cb: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();

    for s in 0..n {
        let mut stack = Vec::new();
        let mut predecessors = vec![vec![]; n];
        let mut sigma = vec![0usize; n];
        sigma[s] = 1;
        let mut dist = vec![None::<usize>; n];
        dist[s] = Some(0);
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let dv = dist[v].unwrap_or(0);
            for &w in graph.neighbors(v) {
                if dist[w].is_none() {
                    dist[w] = Some(dv + 1);
                    queue.push_back(w);
                }
                if dist[w] == Some(dv + 1) {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                if sigma[w] > 0 {
                    let c = (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w]);
                    delta[v] += c;
                    let key = (v.min(w), v.max(w));
                    *edge_cb.entry(key).or_insert(0.0) += c;
                }
            }
        }
    }

    if !graph.is_directed() {
        for v in edge_cb.values_mut() {
            *v /= 2.0;
        }
    }

    edge_cb.into_iter().collect()
}

/// Compute normalized betweenness centrality.
///
/// Normalizes by `(n-1)(n-2)/2` for undirected, `(n-1)(n-2)` for directed.
pub fn normalized_betweenness(graph: &Graph) -> Vec<f64> {
    let n = graph.vertex_count();
    let bc = betweenness_centrality(graph);
    if n <= 2 {
        return bc;
    }
    let norm = if graph.is_directed() {
        (n - 1) * (n - 2)
    } else {
        (n - 1) * (n - 2) / 2
    };
    bc.iter().map(|c| c / norm as f64).collect()
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
        let bc = betweenness_centrality(&g);
        // Center has highest betweenness
        assert!(bc[0] > 0.0);
        // Leaves have zero betweenness
        for i in 1..5 {
            assert!((bc[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_path_endpoints() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let bc = betweenness_centrality(&g);
        // Interior vertices have higher betweenness
        assert!(bc[1] > bc[0]);
        assert!(bc[2] > bc[3]);
    }

    #[test]
    fn test_complete_graph() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j);
            }
        }
        let bc = betweenness_centrality(&g);
        // All vertices in K4 have zero betweenness (all shortest paths are direct)
        for c in &bc {
            assert!(c.abs() < 1e-10);
        }
    }

    #[test]
    fn test_most_central() {
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i);
        }
        assert_eq!(most_central_vertex(&g), 0);
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(3);
        let bc = betweenness_centrality(&g);
        assert_eq!(bc.len(), 3);
        assert!(bc.iter().all(|&c| c.abs() < 1e-10));
    }

    #[test]
    fn test_normalized() {
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i);
        }
        let nbc = normalized_betweenness(&g);
        assert_eq!(nbc.len(), 5);
        assert!(nbc[0] > 0.0);
    }
}
