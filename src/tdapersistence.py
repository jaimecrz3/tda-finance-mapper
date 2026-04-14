from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


"""
Remarkable property -> Una propiedad importante que hace la homología persistente
adecuada para analizar datos con ruido es su robustez ante pequeñas perturbaciones. 
De manera informal, esta propiedad dice que si la nube de puntos subyacente cambia un 
"poco", entonces el correspondiente diagrama de persistencia se meve solo un "poco" 
respecto a la Wesserstein distance. Fuente: D. Chen-Steiner, et. al. Stability of 
persistance diagrams, Discrete and Computational Geometry 37 (2007) 103.

1) Se parte de:

- Un conjunto de puntos X_t = {x_1, ..., x_N} (activos).
- Una distancia d(i,j), por ejemplo d = sqrt(2(1 - rho_ij)).
- Una matriz D_t ∈ R^{NxN}.

La idea central es construir una filtración de complejos simpliciales
(Vietoris-Rips) usando la misma distancia. No se construye una distancia
para H0 y otra para H1: la diferencia está únicamente en la dimensión de
la homología que se calcula.

--------------------------------------------------
Paso A: Filtración Vietoris-Rips VR(X_t, ε)
--------------------------------------------------

Para cada ε ≥ 0:

- Se incluye un 0-símplice por cada punto.
- Se incluye una arista [i,j] si d(i,j) ≤ ε.
- Se incluye un triángulo [i,j,k] si todas las distancias por pares ≤ ε.
- En general, un k-símplice se incluye si todas las distancias por pares
  dentro del conjunto son ≤ ε.

Al aumentar ε, el complejo solo crece: esto define la filtración.

--------------------------------------------------
Paso B: D_t^(0) (H0)
--------------------------------------------------

H0 mide componentes conexas.

- En ε = 0: cada activo es una componente.
- Al crecer ε: aparecen aristas y las componentes se fusionan.

En el diagrama de persistencia:

- Birth ≈ 0 para cada componente.
- Death = ε en el que se fusiona con otra.
- Una componente persiste hasta infinito.

--------------------------------------------------
Paso C: D_t^(1) (H1)
--------------------------------------------------

H1 mide ciclos 1D (agujeros tipo lazo).

Un ciclo nace cuando hay un lazo de aristas no rellenado por triángulos.
Muere cuando aparecen triángulos que lo rellenan.

Por tanto, H1 no depende solo del grafo de aristas, sino de toda la
estructura simplicial.

--------------------------------------------------
Resumen
--------------------------------------------------

Mismo input: D_t → VR(X_t, ε)

- H0: fusiones de clusters.
- H1: ciclos que aparecen y desaparecen.

En librerías como ripser:

dgms[0] = D_t^(0)
dgms[1] = D_t^(1)

==================================================
2) Winsorización por ventana
==================================================

Winsorizar por ventana significa recortar valores extremos dentro de cada
ventana de lookback.

Para cada activo i:

q_low  = Q_1%(r_{·,i})
q_high = Q_99%(r_{·,i})

Se transforma cada retorno:

r' = min(max(r, q_low), q_high)

Interpretación:

- Los outliers no se eliminan, pero se limitan.
- Se reduce su impacto en correlaciones y distancias.
- Mejora la estabilidad de la topología obtenida.

==================================================
3) Texto para memoria / TFG
==================================================

En este trabajo se emplean dos herramientas complementarias de Topological
Data Analysis sobre ventanas deslizantes de retornos del NASDAQ-100. Por
un lado, el algoritmo Mapper se utiliza para construir una representación
gráfica interpretable de la estructura de similitud entre activos y derivar
reglas de diversificación basadas en clústeres. Por otro, la homología
persistente resume la geometría subyacente de la misma nube de activos de
forma multiescala mediante diagramas de persistencia, evitando depender de
una única elección de escala.

Al comparar diagramas entre ventanas consecutivas mediante distancias como
la bottleneck, se obtiene una señal cuantitativa de cambios estructurales
en la estructura cross-sectional (correlaciones) del mercado. Esta señal
se emplea como mecanismo de regularización: cuando la estructura es
estable, se permite una asignación más agresiva basada en clústeres;
cuando se detecta inestabilidad, se aplica shrinkage hacia un baseline
(equal-weight) para mejorar robustez y controlar rotación y riesgo.

El rendimiento se evalúa con backtests controlados y estudios de ablación
(sin PH vs con PH), manteniendo constante el universo, el periodo y los
costes de transacción.
"""


@dataclass(frozen=True)
class PHParams:
    """
    Parámetros del cálculo de homología persistente sobre una ventana.

    maxdim: normalmente 1 en finanzas (H0 y H1).
    corr_method: 'pearson' o 'spearman' (ablation robusto).
    dist_variant: 'sqrt' -> sqrt(2*(1-rho)), 'linear' -> 1-rho.
    winsor_q: winsorización por columna (activos) en la ventana. None desactiva.
    min_non_nan_frac: fracción mínima de datos no-NaN para mantener un activo en esa ventana.
    min_std: filtro para evitar series casi constantes que rompen la correlación.
    """
    maxdim: int = 1
    corr_method: str = "pearson"
    dist_variant: str = "sqrt"
    winsor_q: Optional[float] = 0.01
    min_non_nan_frac: float = 0.98
    min_std: float = 1e-8


def _winsorize_by_col(df: pd.DataFrame, q: float) -> pd.DataFrame:
    lo = df.quantile(q)
    hi = df.quantile(1.0 - q)
    return df.clip(lower=lo, upper=hi, axis=1)


def clean_returns_window(
    returns_window: pd.DataFrame,
    params: PHParams,
) -> pd.DataFrame:
    """
    Limpia una ventana de retornos (L x N):
      - elimina activos con demasiados NaN
      - rellena NaN restantes con 0 (conservador; alternativa: forward-fill si lo justificas)
      - elimina activos casi constantes
      - winsoriza si procede

    Nota: que el fillna(0) sea aceptable depende de tu pipeline de datos;
    si ya tienes retornos limpios, puedes desactivar/ajustar esto.
    """
    if returns_window.empty:
        return returns_window

    X = returns_window.copy()

    # Filtrar columnas con demasiados NaN
    non_nan_frac = 1.0 - X.isna().mean(axis=0)
    keep = non_nan_frac >= params.min_non_nan_frac
    X = X.loc[:, keep]

    if X.shape[1] == 0:
        return X

    # Winsorizar antes de imputar, para que cuantiles no se contaminen con imputación
    if params.winsor_q is not None:
        X = _winsorize_by_col(X, params.winsor_q)

    # Imputación simple (evita NaNs en corr)
    X = X.fillna(0.0)

    # Filtrar columnas con std muy baja
    std = X.std(axis=0, ddof=0)
    X = X.loc[:, std >= params.min_std]

    return X


def corr_distance_matrix(
    returns_window: pd.DataFrame,
    params: PHParams,
) -> Tuple[np.ndarray, List[str]]:
    """
    Devuelve:
      D: matriz NxN
      symbols: lista de columnas (orden asociado a D)
    """
    X = clean_returns_window(returns_window, params)
    symbols = list(X.columns)

    if len(symbols) == 0:
        return np.empty((0, 0), dtype=float), []

    rho = X.corr(method=params.corr_method).to_numpy()

    # Si hay columnas constantes o problemas numéricos, corr puede dar NaN
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    rho = np.clip(rho, -1.0, 1.0)

    # Asegurar diagonal
    np.fill_diagonal(rho, 1.0)

    if params.dist_variant == "sqrt":
        D = np.sqrt(2.0 * (1.0 - rho))
    elif params.dist_variant == "linear":
        D = 1.0 - rho
    else:
        raise ValueError("PHParams.dist_variant must be 'sqrt' or 'linear'")

    np.fill_diagonal(D, 0.0)
    return D.astype(float), symbols


def compute_persistence_diagrams_from_returns(
    returns_window: pd.DataFrame,
    params: PHParams,
):
    """
    Computa diagramas de persistencia (H0..Hmaxdim) usando ripser sobre una matriz de distancias.
    """
    try:
        from ripser import ripser
    except ImportError as e:
        raise ImportError(
            "Falta dependencia 'ripser'. Instala con: pip install ripser"
        ) from e

    D, symbols = corr_distance_matrix(returns_window, params)
    if D.size == 0:
        return {"dgms": [], "symbols": []}

    out = ripser(D, distance_matrix=True, maxdim=params.maxdim)
    return {"dgms": out["dgms"], "symbols": symbols}
