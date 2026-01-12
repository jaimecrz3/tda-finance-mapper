from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


def bottleneck_distance_H1(dgms_prev: List[np.ndarray], dgms_curr: List[np.ndarray]) -> float:
    """
    Bottleneck distance entre H1 de dos ventanas.
    """
    try:
        from persim import bottleneck
    except ImportError as e:
        raise ImportError(
            "Falta dependencia 'persim'. Instala con: pip install persim"
        ) from e

    H1_prev = dgms_prev[1] if len(dgms_prev) > 1 else np.empty((0, 2))
    H1_curr = dgms_curr[1] if len(dgms_curr) > 1 else np.empty((0, 2))
    return float(bottleneck(H1_prev, H1_curr))


@dataclass
class RegimeController:
    """
    Control online (sin look-ahead):
      - mantiene historial de scores pasados
      - estima umbral por quantil sobre ese historial
      - produce alpha_t para mezclar pesos TDA con equal-weight

    Parámetros:
      - history_len: ventana rolling para el quantil (p.ej. 252 si rebalance semanal/diario, ajusta).
      - quantile: p.ej. 0.90 (cambio de régimen si score supera el P90 histórico reciente)
      - min_history: no aplica control hasta tener cierta historia
      - min_alpha: nunca desactiva completamente TDA
    """
    history_len: int = 252
    quantile: float = 0.90
    min_history: int = 30
    min_alpha: float = 0.20

    def __post_init__(self):
        self._scores: Deque[float] = deque(maxlen=self.history_len)

    def threshold(self) -> Optional[float]:
        if len(self._scores) < self.min_history:
            return None
        arr = np.array(self._scores, dtype=float)
        return float(np.quantile(arr, self.quantile))

    def alpha_from_score(self, score: Optional[float]) -> float:
        """
        alpha=1 si no hay score o si no hay historia suficiente.
        Si hay umbral, alpha decae cuando score > thr.
        """
        if score is None:
            return 1.0
        thr = self.threshold()
        if thr is None or thr <= 0:
            return 1.0

        if score <= thr:
            return 1.0

        # Decaimiento suave: score=thr -> 1; score=2*thr -> ~0.5; con suelo min_alpha
        x = score / thr
        alpha = 1.0 / x
        alpha = float(np.clip(alpha, self.min_alpha, 1.0))
        return alpha

    def update_history(self, score: Optional[float]) -> None:
        if score is None:
            return
        if np.isfinite(score) and score >= 0.0:
            self._scores.append(float(score))
