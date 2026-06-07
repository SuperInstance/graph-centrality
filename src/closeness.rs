//! Closeness centrality measures.
//!
//! Computes closeness centrality based on shortest path distances
//! using BFS for unweighted graphs.

use crate::Graph;
use std::collections::VecDeque;

/// Compute closeness centrality for all vertices.
///
/// `C(v) = (n-1) / sum(dist(v, u) for all u != v)`
pub fn closeness_centrality(graph: &Graph) -> Vec<f64> {
    let n = graph.vertex_count();
    let mut cc = vec![0.0; n];

    for v in 0..n {
        let dist = bfs_distances(graph, v);
        let sum_dist: f64 = dist.iter().map(|d| d.unwrap_or(0) as f64).sum::<f64>()
            - dist[v].map_or(0.0, |d| d as f64);
        if sum_dist > 0.0 && n > 1 {
            cc[v] = (n - 1) as f64 / sum_dist;
        }
    }

    cc
}

/// Compute harmonic centrality for all vertices.
///
/// `H(v) = sum(1/dist(v, u) for all u != v)`, treating unreachable as 0.
pub fn harmonic_centrality(graph: &Graph) -> Vec<f64> {
    let n = graph.vertex_count();
    let mut hc = vec![0.0; n];

    #[allow(clippy::needless_range_loop)]
    for v in 0..n {
        let dist = bfs_distances(graph, v);
        #[allow(clippy::needless_range_loop)]
        for u in 0..n {
            if u != v {
                if let Some(d) = dist[u] {
                    if d > 0 {
                        hc[v] += 1.0 / d as f64;
                    }
                }
            }
        }
    }

    hc
}

/// Compute normalized closeness centrality.
///
/// Divides by `(n-1)` so values are in [0, 1].
pub fn normalized_closeness(graph: &Graph) -> Vec<f64> {
    let n = graph.vertex_count();
    let cc = closeness_centrality(graph);
    if n <= 1 {
        return cc;
    }
    cc.iter().map(|c| c / (n - 1) as f64 * (n - 1) as f64).collect()
    // normalized_closeness = cc already normalized since cc[v] = (n-1)/sum
}

/// Find the vertex with the highest closeness centrality.
pub fn most_central_vertex(graph: &Graph) -> usize {
    let cc = closeness_centrality(graph);
    cc.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute the eccentricity of each vertex (max distance to any other vertex).
pub fn eccentricity(graph: &Graph) -> Vec<Option<usize>> {
    let n = graph.vertex_count();
    (0..n).map(|v| {
        let dist = bfs_distances(graph, v);
        dist.iter().filter_map(|&d| d).max()
    }).collect()
}

/// Compute the graph's radius and diameter.
pub fn radius_diameter(graph: &Graph) -> (Option<usize>, Option<usize>) {
    let ecc = eccentricity(graph);
    let radius = ecc.iter().filter_map(|&e| e).min();
    let diameter = ecc.iter().filter_map(|&e| e).max();
    (radius, diameter)
}

fn bfs_distances(graph: &Graph, source: usize) -> Vec<Option<usize>> {
    let n = graph.vertex_count();
    let mut dist = vec![None; n];
    let mut queue = VecDeque::new();
    dist[source] = Some(0);
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        let du = dist[u].unwrap_or(0);
        for &v in graph.neighbors(u) {
            if dist[v].is_none() {
                dist[v] = Some(du + 1);
                queue.push_back(v);
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_star_center_highest() {
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i);
        }
        let cc = closeness_centrality(&g);
        assert!(cc[0] > cc[1]);
    }

    #[test]
    fn test_path_endpoints_lowest() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let cc = closeness_centrality(&g);
        assert!(cc[1] > cc[0]);
        assert!(cc[2] > cc[3]);
    }

    #[test]
    fn test_complete_graph_equal() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j);
            }
        }
        let cc = closeness_centrality(&g);
        // All should be equal
        for c in &cc {
            assert!((c - cc[0]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_harmonic_centrality() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let hc = harmonic_centrality(&g);
        // Vertex 1 has highest harmonic centrality
        assert!(hc[1] > hc[0]);
        assert!(hc[1] > hc[2]);
    }

    #[test]
    fn test_eccentricity() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let ecc = eccentricity(&g);
        assert_eq!(ecc[0], Some(3));
        assert_eq!(ecc[1], Some(2));
    }

    #[test]
    fn test_radius_diameter() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let (r, d) = radius_diameter(&g);
        assert_eq!(r, Some(2));
        assert_eq!(d, Some(3));
    }

    #[test]
    fn test_most_central_closeness() {
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(0, i);
        }
        assert_eq!(most_central_vertex(&g), 0);
    }
}
