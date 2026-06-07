# graph-centrality

Graph centrality measures for Rust. Pure `std` — no external dependencies.

## Features

- **Betweenness centrality** — Brandes' O(VE) algorithm with edge betweenness
- **Closeness centrality** — Standard and harmonic centrality, eccentricity, radius/diameter
- **Eigenvector centrality** — Power iteration with normalization
- **PageRank** — Standard and personalized PageRank with damping
- **Katz centrality** — Attenuated walk-based influence measure

## Usage

```rust
use graph_centrality::{Graph, betweenness, closeness, pagerank, katz};

let mut g = Graph::new(5);
for i in 1..5 { g.add_edge(0, i); }

let bc = betweenness::betweenness_centrality(&g);
println!("Betweenness: {bc:?}");

let cc = closeness::closeness_centrality(&g);
println!("Closeness: {cc:?}");

let pr = pagerank::pagerank(&g, 0.85, 1000, 1e-10);
println!("PageRank: {pr:?}");
```

## License

MIT
