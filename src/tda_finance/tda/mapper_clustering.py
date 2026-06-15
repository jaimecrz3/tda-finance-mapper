"""Mapper-based clustering and portfolio-weight construction.

This module contains the main utilities used to build Mapper graphs from
financial price windows and to transform the resulting clusters into portfolio
weights.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from umap import UMAP


HACALinkage = Literal["complete", "average", "single"]


@dataclass(frozen=True)
class MapperParams:
    """Hyperparameters of the Mapper pipeline.

    Parameters
    ----------
    pca_var : float, default=0.80
        Fraction of variance retained by the PCA step before applying UMAP.
    umap_dim : int, default=1
        Dimension of the final lens used by Mapper.
    random_state : int, default=1
        Random seed used in PCA and UMAP for reproducibility.
    n_cubes : int, default=10
        Number of intervals or hypercubes used in the Mapper cover.
    perc_overlap : float, default=0.2
        Percentage of overlap between adjacent cover elements.
    dbscan_eps : float, default=0.3
        Maximum distance between two samples for DBSCAN.
    dbscan_min_samples : int, default=2
        Minimum number of samples required by DBSCAN to form a core point.
    clusterer : {"dbscan", "haca", "hac", "agglomerative"}, default="dbscan"
        Local clustering algorithm used inside each cover element.
    haca_distance_threshold : float, default=0.7
        Distance threshold used by the agglomerative clustering wrapper.
    haca_linkage : {"complete", "average", "single"}, default="average"
        Linkage criterion used by the agglomerative clustering wrapper.
    min_assets : int, default=5
        Minimum number of assets required to attempt Mapper construction.
    verbose : bool, default=False
        Whether to print diagnostic information during Mapper construction.
    """

    pca_var: float = 0.80
    umap_dim: int = 1
    random_state: int = 1

    n_cubes: int = 10
    perc_overlap: float = 0.2

    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2

    clusterer: str = "dbscan"

    haca_distance_threshold: float = 0.7
    haca_linkage: HACALinkage = "average"

    min_assets: int = 5
    verbose: bool = False


class HACAClusterer(BaseEstimator, ClusterMixin):
    """Agglomerative clustering wrapper compatible with KeplerMapper.

    The wrapper computes a correlation distance matrix and then applies
    agglomerative clustering with a precomputed distance matrix. It is useful
    when the local clustering step should not discard observations as noise.

    Parameters
    ----------
    distance_threshold : float, default=0.7
        Distance threshold above which clusters are not merged.
    linkage : {"complete", "average", "single"}, default="average"
        Linkage criterion used by AgglomerativeClustering.
    """

    def __init__(
        self,
        distance_threshold: float = 0.7,
        linkage: HACALinkage = "average",
    ) -> None:
        self.distance_threshold: float = float(distance_threshold)
        self.linkage: HACALinkage = linkage

    def fit(self, X: Any, y: Any = None) -> "HACAClusterer":
        """Fit the estimator.

        This method is provided for compatibility with the scikit-learn API.

        Parameters
        ----------
        X : array-like
            Input data.
        y : object, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        HACAClusterer
            Fitted estimator.
        """
        return self

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Cluster observations using agglomerative clustering.

        Parameters
        ----------
        X : array-like
            Input data with observations in rows.
        y : object, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        numpy.ndarray
            Cluster labels for each observation.
        """
        X_array = np.asarray(X)
        n_samples = X_array.shape[0]

        if n_samples == 0:
            return np.array([], dtype=int)

        if n_samples == 1:
            return np.zeros(1, dtype=int)

        if X_array.shape[1] < 2:
            return np.zeros(n_samples, dtype=int)

        # Correlation is undefined for constant vectors. NumPy may produce
        # NaNs, which are replaced below before clustering.
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X_array)

        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)

        dist = 1.0 - corr
        np.fill_diagonal(dist, 0.0)

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
        )

        return model.fit_predict(dist)


def build_clusters_from_prices(
    prices_window: pd.DataFrame,
    params: MapperParams,
) -> Optional[List[List[List[str]]]]:
    """Build Mapper clusters from a price window.

    The function represents each asset as a vector of recent log-returns,
    standardizes each asset individually, builds a Mapper graph and returns the
    connected components of that graph as nested clusters of asset identifiers.

    Parameters
    ----------
    prices_window : pandas.DataFrame
        Price matrix indexed by date, with one column per asset.
    params : MapperParams
        Hyperparameters used by the Mapper pipeline.

    Returns
    -------
    list of list of list of str or None
        Nested cluster structure. The outer list contains connected components
        of the Mapper graph. Each component contains Mapper nodes, and each node
        contains the asset identifiers assigned to that node. Returns None when
        there are not enough assets or observations to build Mapper.

    Notes
    -----
    Missing prices are forward-filled inside the window, but backward filling is
    not used. This avoids introducing future information into the construction.
    """
    # Forward filling is allowed within the historical window. Backward filling
    # is avoided because it would introduce future information.
    prices = prices_window.sort_index().ffill().dropna(axis=1, how="any")

    if prices.shape[1] < params.min_assets:
        return None

    price_ratio = prices / prices.shift(1)
    log_ret_t = price_ratio.apply(np.log).dropna()

    if log_ret_t.shape[0] < 10:
        return None

    # Assets are represented as points: each row is one asset and each column is
    # one return observation from the lookback window.
    log_returns = log_ret_t.T

    if log_returns.shape[0] < params.min_assets:
        return None

    X = log_returns.values.astype(float)

    # Standardize each asset separately so Mapper compares return patterns
    # rather than differences in volatility level.
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, ddof=0, keepdims=True)
    sigma = np.where(sigma == 0.0, 1.0, sigma)

    Xz = (X - mu) / sigma
    log_returns_z = pd.DataFrame(
        Xz,
        index=log_returns.index,
        columns=log_returns.columns,
    )

    mapper = km.KeplerMapper()

    # The lens is built in two steps: PCA first reduces dimensionality and UMAP
    # then produces the final low-dimensional representation used by Mapper.
    projection_pipeline: Any = [
        PCA(n_components=params.pca_var, random_state=params.random_state),
        UMAP(
            n_components=params.umap_dim,
            random_state=params.random_state,
            n_jobs=1,
            metric="correlation",
        ),
    ]

    projected = mapper.fit_transform(
        log_returns_z,
        projection=projection_pipeline,
    )

    cover = km.Cover(
        n_cubes=params.n_cubes,
        perc_overlap=params.perc_overlap,
    )

    if params.clusterer.lower() == "dbscan":
        clusterer = DBSCAN(
            metric="correlation",
            eps=params.dbscan_eps,
            min_samples=params.dbscan_min_samples,
            n_jobs=1,
        )
    elif params.clusterer.lower() in ("haca", "hac", "agglomerative"):
        clusterer = HACAClusterer(
            distance_threshold=params.haca_distance_threshold,
            linkage=params.haca_linkage,
        )
    else:
        raise ValueError(f"Unsupported clusterer: {params.clusterer}")

    graph = mapper.map(
        projected,
        log_returns_z,
        cover=cover,
        clusterer=clusterer,
    )

    nodes = graph.get("nodes", {})
    links = graph.get("links", {})

    if params.verbose:
        print("=== Mapper graph ===")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of links: {sum(len(v) for v in links.values())}")

    if not nodes:
        return None

    # Convert the Mapper output into a NetworkX graph to extract connected
    # components. Each component groups related Mapper nodes.
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(nodes.keys())

    for node, neighbors in links.items():
        for neighbor in neighbors:
            graph_nx.add_edge(node, neighbor)

    components = list(nx.connected_components(graph_nx))
    tickers = list(log_returns.index)

    if params.verbose:
        print("=== Connected components ===")
        print(f"Number of components: {len(components)}")
        print("Component sizes:", [len(component) for component in components])

    clustered: List[List[List[str]]] = []

    for i, component in enumerate(components):
        if params.verbose:
            print(f"\nComponent {i} | {len(component)} nodes")

        giant_cluster: List[List[str]] = []

        for node_id in component:
            idxs = nodes.get(node_id, [])

            if not idxs:
                continue

            # KeplerMapper stores integer positions. They are mapped back to the
            # original asset identifiers.
            small_cluster = [
                tickers[idx]
                for idx in idxs
                if 0 <= idx < len(tickers)
            ]

            # Remove possible duplicates inside the same Mapper node.
            unique_cluster = list(dict.fromkeys(small_cluster))

            if params.verbose:
                print(f"  Node {node_id}: {len(unique_cluster)} assets")

            if unique_cluster:
                giant_cluster.append(unique_cluster)

        if giant_cluster:
            clustered.append(giant_cluster)

    if params.verbose:
        covered = {
            asset
            for giant_cluster in clustered
            for small_cluster in giant_cluster
            for asset in small_cluster
        }
        missing = [asset for asset in tickers if asset not in covered]

        print("=== Coverage ===")
        print(f"Total assets: {len(tickers)}")
        print(f"Covered assets: {len(covered)} | Missing: {len(missing)}")

        if missing:
            print("Missing examples:", missing[:5])

        print("End of Mapper construction\n")

    return clustered


def cap_weights_strict(w: Dict[str, float], cap: float) -> Dict[str, float]:
    """Apply a strict maximum weight cap with iterative redistribution.

    Parameters
    ----------
    w : dict of str to float
        Raw asset weights.
    cap : float
        Maximum allowed weight per asset.

    Returns
    -------
    dict of str to float
        Normalized asset weights satisfying the weight cap when feasible.
    """
    if not w:
        return {}

    cap = float(cap)
    w = {asset: float(weight) for asset, weight in w.items() if weight > 0}

    if not w:
        return {}

    total = sum(w.values())
    w = {asset: weight / total for asset, weight in w.items()}

    # If some weights exceed the cap, fix them at the cap and redistribute the
    # remaining mass among the uncapped assets. Repeat until feasible.
    while True:
        over_cap = {asset for asset, weight in w.items() if weight > cap}

        if not over_cap:
            return w

        fixed_mass = cap * len(over_cap)

        if fixed_mass >= 1.0 - 1e-12:
            n_assets = len(w)
            base_weight = min(cap, 1.0 / n_assets)
            capped = {asset: base_weight for asset in w}
            total_capped = sum(capped.values())
            return {asset: weight / total_capped for asset, weight in capped.items()}

        under_cap = [asset for asset in w if asset not in over_cap]

        if not under_cap:
            n_assets = len(w)
            base_weight = min(cap, 1.0 / n_assets)
            capped = {asset: base_weight for asset in w}
            total_capped = sum(capped.values())
            return {asset: weight / total_capped for asset, weight in capped.items()}

        fixed_weights = {asset: cap for asset in over_cap}
        remaining_mass = 1.0 - fixed_mass

        under_sum = sum(w[asset] for asset in under_cap)

        if under_sum <= 0:
            per_asset = remaining_mass / len(under_cap)
            under_weights = {asset: per_asset for asset in under_cap}
        else:
            under_weights = {
                asset: remaining_mass * (w[asset] / under_sum)
                for asset in under_cap
            }

        w = {**fixed_weights, **under_weights}


def weight_distribution_node_overlap(
    clustered_symbols: List[List[List[str]]],
    max_weight: float = 0.1,
) -> Dict[str, float]:
    """Convert Mapper nodes into portfolio weights using node overlap.

    Each Mapper node receives the same total weight. The weight of a node is
    distributed equally among the assets contained in that node. Assets that
    appear in several overlapping nodes accumulate weight.

    Parameters
    ----------
    clustered_symbols : list of list of list of str
        Nested cluster structure returned by build_clusters_from_prices.
    max_weight : float, default=0.1
        Maximum allowed weight per asset.

    Returns
    -------
    dict of str to float
        Portfolio weights indexed by asset identifier.
    """
    if not clustered_symbols:
        return {}

    weights: Dict[str, float] = {}

    total_nodes = sum(len(giant_cluster) for giant_cluster in clustered_symbols)

    if total_nodes == 0:
        return {}

    weight_per_node = 1.0 / total_nodes

    # Node-overlap weighting rewards assets that appear in several overlapping
    # Mapper nodes, because they are more central in the Mapper representation.
    for giant_cluster in clustered_symbols:
        for small_cluster in giant_cluster:
            if not small_cluster:
                continue

            weight_per_asset = weight_per_node / len(small_cluster)

            for asset in small_cluster:
                weights[asset] = weights.get(asset, 0.0) + weight_per_asset

    return cap_weights_strict(weights, cap=max_weight)


def weight_distribution_macro_cluster(
    clustered_symbols: List[List[List[str]]],
    max_weight: float = 0.1,
) -> Dict[str, float]:
    """Convert Mapper connected components into portfolio weights.

    Each connected component of the Mapper graph receives the same total
    weight. Inside each component, the weight is distributed equally among the
    unique assets contained in that component.

    Parameters
    ----------
    clustered_symbols : list of list of list of str
        Nested cluster structure returned by build_clusters_from_prices.
    max_weight : float, default=0.1
        Maximum allowed weight per asset.

    Returns
    -------
    dict of str to float
        Portfolio weights indexed by asset identifier.
    """
    if not clustered_symbols:
        return {}

    weights: Dict[str, float] = {}
    valid_components = []

    for giant_cluster in clustered_symbols:
        unique_assets = set()

        for node in giant_cluster:
            unique_assets.update(node)

        if unique_assets:
            valid_components.append(unique_assets)

    if not valid_components:
        return {}

    component_weight = 1.0 / len(valid_components)

    for assets in valid_components:
        weight_per_asset = component_weight / len(assets)

        for asset in assets:
            weights[asset] = weights.get(asset, 0.0) + weight_per_asset

    return cap_weights_strict(weights, cap=max_weight)


def weight_distribution(
    clustered_symbols: List[List[List[str]]],
    max_weight: float = 0.1,
    method: str = "node_overlap",
) -> Dict[str, float]:
    """Convert Mapper clusters into portfolio weights.

    Parameters
    ----------
    clustered_symbols : list of list of list of str
        Nested cluster structure returned by build_clusters_from_prices.
    max_weight : float, default=0.1
        Maximum allowed weight per asset.
    method : {"node_overlap", "macro_cluster"}, default="node_overlap"
        Weighting method used to transform Mapper clusters into asset weights.

    Returns
    -------
    dict of str to float
        Portfolio weights indexed by asset identifier.

    Raises
    ------
    ValueError
        If method is not supported.
    """
    if method == "node_overlap":
        return weight_distribution_node_overlap(
            clustered_symbols,
            max_weight=max_weight,
        )

    if method == "macro_cluster":
        return weight_distribution_macro_cluster(
            clustered_symbols,
            max_weight=max_weight,
        )

    raise ValueError(f"Unsupported weight_method: {method}")
