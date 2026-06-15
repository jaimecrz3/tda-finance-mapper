"""Feature extraction from persistence diagrams.

This module provides simple summary features from persistence diagrams. These
features are mainly used as diagnostics for the persistent-homology regime
control.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _lifetimes(diagram: Optional[np.ndarray]) -> np.ndarray:
    """Compute finite lifetimes from a persistence diagram.

    Parameters
    ----------
    diagram : numpy.ndarray or None
        Persistence diagram with two columns: birth and death.

    Returns
    -------
    numpy.ndarray
        Array containing finite non-negative lifetimes.
    """
    if diagram is None or diagram.size == 0:
        return np.array([], dtype=float)

    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    lifetimes = lifetimes[lifetimes >= 0.0]

    return lifetimes


def ph_summary_features(dgms: List[np.ndarray]) -> Dict[str, float]:
    """Compute summary features from persistence diagrams.

    The returned features are intentionally simple and interpretable. They are
    used to describe the persistence diagrams obtained from each return window,
    especially the H1 diagram.

    Parameters
    ----------
    dgms : list of numpy.ndarray
        Persistence diagrams. The first element corresponds to H0 and the
        second element, when available, corresponds to H1.

    Returns
    -------
    dict of str to float
        Dictionary with counts, total persistence, maximum persistence and
        selected quantiles of H1 lifetimes.
    """
    h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    lifetimes_h0 = _lifetimes(h0)
    lifetimes_h1 = _lifetimes(h1)

    features: Dict[str, float] = {
        "H0_count": float(len(lifetimes_h0)),
        "H1_count": float(len(lifetimes_h1)),
        "H1_total_persistence": (
            float(lifetimes_h1.sum()) if lifetimes_h1.size else 0.0
        ),
        "H1_max_persistence": (
            float(lifetimes_h1.max()) if lifetimes_h1.size else 0.0
        ),
        "H1_q50": (
            float(np.quantile(lifetimes_h1, 0.50)) if lifetimes_h1.size else 0.0
        ),
        "H1_q75": (
            float(np.quantile(lifetimes_h1, 0.75)) if lifetimes_h1.size else 0.0
        ),
        "H1_q90": (
            float(np.quantile(lifetimes_h1, 0.90)) if lifetimes_h1.size else 0.0
        ),
    }

    return features
