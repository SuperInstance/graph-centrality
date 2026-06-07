//! # graph-centrality
//!
//! Graph centrality measures for Rust. Pure `std` — no external dependencies.
//!
//! ## Modules
//!
//! - [`betweenness`] — Betweenness centrality (Brandes algorithm)
//! - [`closeness`] — Closeness centrality
//! - [`eigenvector`] — Eigenvector centrality (power iteration)
//! - [`pagerank`] — PageRank
//! - [`katz`] — Katz centrality

pub mod betweenness;
pub mod closeness;
pub mod eigenvector;
pub mod pagerank;
pub mod katz;

/// A simple unweighted graph represented as an adjacency list.
#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    adj: Vec<Vec<usize>>,
    directed: bool,
}

impl Graph {
    /// Create a new undirected graph with `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            directed: false,
        }
    }

    /// Create a new directed graph with `n` vertices.
    pub fn new_directed(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            directed: true,
        }
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.n
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        let count: usize = self.adj.iter().map(|v| v.len()).sum();
        if self.directed { count } else { count / 2 }
    }

    /// Add an edge from `u` to `v`.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n && v < self.n, "Vertex index out of bounds");
        self.adj[u].push(v);
        if !self.directed && u != v {
            self.adj[v].push(u);
        }
    }

    /// Get the neighbors of vertex `v`.
    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    /// Get the out-degree of vertex `v`.
    pub fn out_degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Get the in-degree of vertex `v`.
    pub fn in_degree(&self, v: usize) -> usize {
        self.adj.iter().filter(|adj| adj.contains(&v)).count()
    }

    /// Whether the graph is directed.
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Get the adjacency list.
    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adj
    }
}
