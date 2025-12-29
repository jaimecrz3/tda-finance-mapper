# TDA in Finance: Mapper (Bachelor's Thesis)

This repository contains the code and experiments for my Bachelor's thesis on **Topological Data Analysis (TDA)** applied to **high-dimensional financial data**, with a focus on the **Mapper** algorithm (and optional extensions with persistent homology / Ball Mapper).

## Project goal
Build an end-to-end, reproducible pipeline to:
- construct rolling-window representations of market data (returns, volatility, etc.),
- apply **Mapper** to visualize and summarize market structure,
- identify **market regimes** and transitions,
- compare with standard baselines (e.g., PCA + clustering).

## Method overview
1. **Data**: daily prices (indices and/or a basket of stocks).
2. **Preprocessing**: log-returns, rolling features, scaling.
3. **Lens (filter)**: PCA/UMAP projections or market indicators.
4. **Cover**: overlapping intervals (n_intervals, overlap).
5. **Local clustering**: DBSCAN or single-linkage (MST cut).
6. **Mapper graph**: nodes = local clusters, edges = shared points.
7. **Evaluation**: stability across parameters, regime coherence, baselines.
