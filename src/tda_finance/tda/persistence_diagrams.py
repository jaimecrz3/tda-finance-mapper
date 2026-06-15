"""Persistent-homology utilities for financial return windows.

This module contains functions to clean return windows, build correlation-based
distance matrices and compute persistence diagrams using Vietoris--Rips
persistent homology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


CorrelationMethod = Literal["pearson", "kendall", "spearman"]
DistanceVariant = Literal["sqrt", "linear"]


@dataclass(frozen=True)
class PHParams:
    """Hyperparameters for persistent-homology computation.

    Parameters
    ----------
    maxdim : int, default=1
        Maximum homology dimension computed by ``ripser``. With ``maxdim=1``,
        both H0 and H1 diagrams are returned.
    corr_method : {"pearson", "kendall", "spearman"}, default="pearson"
        Correlation method used to build the asset dependence matrix.
    dist_variant : {"sqrt", "linear"}, default="sqrt"
        Transformation from correlation to distance. ``"sqrt"`` uses
        ``sqrt(2 * (1 - rho))`` and ``"linear"`` uses ``1 - rho``.
    winsor_q : float, optional, default=0.01
        Quantile used for winsorization by asset. If ``None``, winsorization is
        disabled.
    min_non_nan_frac : float, default=0.98
        Minimum fraction of non-missing observations required to keep an asset.
    min_std : float, default=1e-8
        Minimum standard deviation required to keep an asset.
    """

    maxdim: int = 1
    corr_method: CorrelationMethod = "pearson"
    dist_variant: DistanceVariant = "sqrt"
    winsor_q: Optional[float] = 0.01
    min_non_nan_frac: float = 0.98
    min_std: float = 1e-8


def _winsorize_by_col(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Winsorize each column of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    q : float
        Lower-tail quantile. The upper-tail quantile is ``1 - q``.

    Returns
    -------
    pandas.DataFrame
        Winsorized DataFrame.
    """
    lower = df.quantile(q)
    upper = df.quantile(1.0 - q)

    return df.clip(lower=lower, upper=upper, axis=1)


def clean_returns_window(
    returns_window: pd.DataFrame,
    params: PHParams,
) -> pd.DataFrame:
    """Clean a return window before computing persistent homology.

    The cleaning step removes assets with too many missing observations,
    optionally winsorizes extreme returns, fills remaining missing values with
    zero and removes almost-constant assets.

    Parameters
    ----------
    returns_window : pandas.DataFrame
        Return matrix indexed by date, with one column per asset.
    params : PHParams
        Parameters controlling the cleaning process.

    Returns
    -------
    pandas.DataFrame
        Cleaned return matrix.
    """
    if returns_window.empty:
        return returns_window

    clean_window = returns_window.copy()

    non_nan_frac = 1.0 - clean_window.isna().mean(axis=0)
    keep_assets = non_nan_frac >= params.min_non_nan_frac
    clean_window = clean_window.loc[:, keep_assets]

    if clean_window.shape[1] == 0:
        return clean_window

    if params.winsor_q is not None:
        clean_window = _winsorize_by_col(clean_window, params.winsor_q)

    clean_window = clean_window.fillna(0.0)

    asset_std = clean_window.std(axis=0, ddof=0)
    clean_window = clean_window.loc[:, asset_std >= params.min_std]

    return clean_window


def corr_distance_matrix(
    returns_window: pd.DataFrame,
    params: PHParams,
) -> Tuple[np.ndarray, List[str]]:
    """Compute a correlation-based distance matrix.

    Parameters
    ----------
    returns_window : pandas.DataFrame
        Return matrix indexed by date, with one column per asset.
    params : PHParams
        Parameters controlling cleaning, correlation and distance conversion.

    Returns
    -------
    tuple of numpy.ndarray and list of str
        Distance matrix and list of asset identifiers associated with the rows
        and columns of the matrix.

    Raises
    ------
    ValueError
        If ``params.dist_variant`` is not supported.
    """
    clean_window = clean_returns_window(returns_window, params)
    symbols = list(clean_window.columns)

    if len(symbols) == 0:
        return np.empty((0, 0), dtype=float), []

    corr = clean_window.corr(method=params.corr_method).to_numpy()

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    if params.dist_variant == "sqrt":
        distance = np.sqrt(2.0 * (1.0 - corr))
    elif params.dist_variant == "linear":
        distance = 1.0 - corr
    else:
        raise ValueError("params.dist_variant must be 'sqrt' or 'linear'.")

    np.fill_diagonal(distance, 0.0)

    return distance.astype(float), symbols


def compute_persistence_diagrams_from_returns(
    returns_window: pd.DataFrame,
    params: PHParams,
) -> Dict[str, Any]:
    """Compute persistence diagrams from a return window.

    The function first builds a correlation-based distance matrix between
    assets and then computes Vietoris--Rips persistence diagrams using
    ``ripser``.

    Parameters
    ----------
    returns_window : pandas.DataFrame
        Return matrix indexed by date, with one column per asset.
    params : PHParams
        Parameters used to clean the window and compute persistence diagrams.

    Returns
    -------
    dict
        Dictionary with two entries:

        ``"dgms"``
            List of persistence diagrams returned by ``ripser``.
        ``"symbols"``
            Asset identifiers used to build the distance matrix.

    Raises
    ------
    ImportError
        If ``ripser`` is not installed.
    """
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'ripser'. Install it with: pip install ripser"
        ) from exc

    distance, symbols = corr_distance_matrix(returns_window, params)

    if distance.size == 0:
        return {"dgms": [], "symbols": []}

    output = ripser(
        distance,
        distance_matrix=True,
        maxdim=params.maxdim,
    )

    return {
        "dgms": output["dgms"],
        "symbols": symbols,
    }
