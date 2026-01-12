from __future__ import annotations

from typing import Dict, List
import numpy as np


def _lifetimes(diag: np.ndarray) -> np.ndarray:
    """
    diag: array (k,2) con (birth, death)
    Devuelve lifetimes finitos (death-birth).
    """
    if diag is None or diag.size == 0:
        return np.array([], dtype=float)
    lt = diag[:, 1] - diag[:, 0]
    lt = lt[np.isfinite(lt)]
    lt = lt[lt >= 0.0]
    return lt


def ph_summary_features(dgms: List[np.ndarray]) -> Dict[str, float]:
    """
    Features simples (muy defendibles y fáciles de depurar):
      - conteos H0/H1
      - total persistence H1
      - max persistence H1
      - quantiles H1
    """
    H0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
    H1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    lt0 = _lifetimes(H0)
    lt1 = _lifetimes(H1)

    feats: Dict[str, float] = {
        "H0_count": float(len(lt0)),
        "H1_count": float(len(lt1)),
        "H1_total_persistence": float(lt1.sum()) if lt1.size else 0.0,
        "H1_max_persistence": float(lt1.max()) if lt1.size else 0.0,
        "H1_q50": float(np.quantile(lt1, 0.50)) if lt1.size else 0.0,
        "H1_q75": float(np.quantile(lt1, 0.75)) if lt1.size else 0.0,
        "H1_q90": float(np.quantile(lt1, 0.90)) if lt1.size else 0.0,
    }
    return feats
