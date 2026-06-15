"""Regime-detection utilities based on persistence landscapes.

This module provides a simple topological anomaly detector. It summarizes each
persistence diagram through the L2 norm of a persistence landscape and compares
that value with a rolling history of previous norms.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np
from gudhi.representations import Landscape


def compute_landscape_norm(
    dgms: List[np.ndarray],
    dimension: int = 1,
    resolution: int = 100,
    num_landscapes: int = 5,
) -> float:
    """Compute the L2 norm of a persistence landscape.

    Parameters
    ----------
    dgms : list of numpy.ndarray
        Persistence diagrams indexed by homology dimension.
    dimension : int, default=1
        Homology dimension used to compute the landscape. The default value
        corresponds to H1.
    resolution : int, default=100
        Number of grid points used to discretize the landscape.
    num_landscapes : int, default=5
        Number of landscape layers computed by GUDHI.

    Returns
    -------
    float
        L2 norm of the selected persistence landscape. If the requested diagram
        is missing or empty, the function returns 0.0.
    """
    if dgms is None or len(dgms) <= dimension:
        return 0.0

    diagram = dgms[dimension]

    if diagram is not None and len(diagram) > 0:
        diagram = diagram[np.isfinite(diagram[:, 1])]

    if diagram is None or len(diagram) == 0:
        return 0.0

    landscape = Landscape(
        resolution=resolution,
        num_landscapes=num_landscapes,
    )

    vector = landscape.fit_transform([diagram])[0]  # type: ignore[operator]

    return float(np.linalg.norm(vector))


class TopologicalAnomalyDetector:
    """Rolling detector for topological anomalies.

    The detector stores a rolling history of persistence-landscape norms. Once
    enough history is available, the current norm is compared with a historical
    quantile. Values above that threshold are treated as anomalous.

    Parameters
    ----------
    history_len : int, default=12
        Maximum number of past norms stored by the detector.
    danger_quantile : float, default=0.90
        Quantile used as anomaly threshold.
    min_history : int, default=10
        Minimum number of past observations required before the detector starts
        flagging anomalies.
    """

    def __init__(
        self,
        history_len: int = 12,
        danger_quantile: float = 0.90,
        min_history: int = 10,
    ) -> None:
        self.history_len = history_len
        self.danger_quantile = danger_quantile
        self.min_history = min_history
        self._norms: Deque[float] = deque(maxlen=history_len)

    def is_market_safe(self, current_norm: float) -> bool:
        """Return whether the current regime is considered safe.

        Parameters
        ----------
        current_norm : float
            Current persistence-landscape norm.

        Returns
        -------
        bool
            True if the current regime is considered safe, and False if the
            current norm is anomalously high relative to its recent history.
        """
        if current_norm <= 0:
            return True

        if len(self._norms) < self.min_history:
            self._norms.append(current_norm)
            return True

        threshold = np.quantile(list(self._norms), self.danger_quantile)

        self._norms.append(current_norm)

        return bool(current_norm <= threshold)
