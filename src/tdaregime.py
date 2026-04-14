from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from gudhi.representations import Landscape


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

"""
El problema del "Bottleneck Distance"
La distancia de Bottleneck(caso P=infinito de la Wasserstein distance) se define por el punto que más se movió.
Riesgo: Si un solo activo se vuelve loco (ruido idiosincrático) y desplaza un punto del diagrama muy lejos, 
el Score se dispara, aunque el resto del mercado (99 activos) esté igual. Es una métrica muy sensible a outliers. 
Ventaja: A pesar de este riesgo, es más rápido y en general suficiente para detectar cambios de régimen.

La alternativa superior: Wasserstein DistanceLa distancia Wasserstein (W_1 o W_2) suma el coste de mover todos 
los puntos de un diagrama al otro.
Ventaja: Es una medida agregada. Si el mercado entero cambia un poco, W sube. Si solo un activo cambia, W sube poco. 
Es mucho más estable para series temporales financieras.

En TDA sobre correlaciones/distancias, H_0 representa cómo se agrupan los activos (clusters).
Qué mide: Mide la "fragmentación" del mercado. Las barras de H_0 te dicen a qué distancia se fusionan los activos en un solo bloque.
Comportamiento en Crisis: En un crash financiero, las correlaciones tienden a 1 (todos los activos caen juntos). 
H_1 representa las relaciones circulares (A se parece a B, B a C, C a D, pero A no se parece a D).
Qué mide: Mide la complejidad y las ineficiencias del mercado.
Comportamiento en Crisis: A menudo, en un pánico total, la estructura compleja colapsa (todo se vuelve lineal).
 Sin embargo, H_1 es muy bueno detectando cambios de régimen sutiles (ej. rotación sectorial, estanflación) donde la volatilidad no necesariamente se dispara, 
pero la estructura interna del mercado cambia.
Veredicto: H_1 es más sofisticado pero más ruidoso. A veces, un solo activo errático puede crear o destruir un ciclo, disparando la 
distancia de Bottleneck sin que el mercado real haya cambiado tanto.

Sugerencia:
Para un sistema de control de riesgo robusto (RegimeController), lo mejor es no elegir uno solo, sino combinarlos, pero dando prioridad a la robustez.
Usando dos pesos w_0 y w_1 que deben de ser no negativs y deben se sumar 1.
"""

def compute_composite_score(prev_dgms: List[np.ndarray], dgms: List[np.ndarray], w0=0.6, w1=0.4) -> float:
    """
    Combina la estabilidad de H0 con la sensibilidad de H1.
    """
    try:
        from persim import bottleneck
    except ImportError as e:
        raise ImportError(
            "Falta dependencia 'persim'. Instala con: pip install persim"
        ) from e

    def remove_infinity(dgm):
        """Elimina puntos con vida infinita del diagrama."""
        if dgm is None or len(dgm) == 0:
            return np.empty((0, 2))
        # Filtra filas donde la segunda columna (death) sea finita
        return dgm[np.isfinite(dgm[:, 1])]

    # --- Procesamiento H0 (Clusters) ---
    # Limpiamos los infinitos, crucial para H0
    h0_prev = remove_infinity(prev_dgms[0])
    h0_curr = remove_infinity(dgms[0])
    
    # Protección: si un diagrama quedó vacío tras limpiar (raro, pero posible)
    if len(h0_prev) == 0 or len(h0_curr) == 0:
        score_h0 = 0.0 # No hay estructura comparable
    else:
        score_h0 = bottleneck(h0_prev, h0_curr) 

    # --- Procesamiento H1 (Ciclos) ---
    # H1 raramente tiene infinitos en finanzas (los ciclos suelen morir),
    # pero es buena práctica limpiar por si acaso.
    h1_prev = remove_infinity(prev_dgms[1])
    h1_curr = remove_infinity(dgms[1])

    if len(h1_prev) == 0 or len(h1_curr) == 0:
        # Si no hay ciclos en uno de los tiempos (mercado muy plano o lineal)
        # La distancia es la máxima persistencia del que sí tiene ciclos, 
        # o 0 si ambos están vacíos.
        if len(h1_prev) == 0 and len(h1_curr) == 0:
            score_h1 = 0.0
        else:
            # Si uno está vacío, bottleneck contra vacío suele dar error en librerías.
            # Asumimos que la distancia es "lo que dura el ciclo más largo del que existe"
            non_empty = h1_prev if len(h1_prev) > 0 else h1_curr
            # Max persistencia (death - birth)
            score_h1 = np.max(non_empty[:, 1] - non_empty[:, 0])
    else:
        score_h1 = bottleneck(h1_prev, h1_curr)

    # Combinación lineal
    return w0 * score_h0 + w1 * score_h1



# Interpretación Financiera:
# Este código implementa una gestión de riesgo dinámica.
# Escenario: El "score" mide la inestabilidad topológica del mercado (señales de crash).
# Funcionamiento: Mientras la inestabilidad esté dentro de lo "habitual" (bajo el percentil 90 histórico), 
# el algoritmo opera agresivamente (alpha=1). Si la inestabilidad se dispara por encima de lo que se ha 
# visto recientemente, el algoritmo se vuelve defensivo automáticamente reduciendo alpha.
@dataclass
class RegimeController:
    """
    Control online (sin look-ahead):
      - mantiene historial de scores pasados
      - estima umbral por quantil sobre ese historial
      - produce alpha_t para mezclar pesos TDA con equal-weight

    Parámetros:
      - history_len: ventana rolling para el quantil (p.ej. 252 si rebalance semanal/diario, ajusta).
      - quantile: p.ej. 0.90 (cambio de régimen si score supera el P90 histórico reciente), es decir,
                   solo reacciona si el score actual es superior al 90% de los scores vistos recientemente.
      - min_history: no aplica control hasta tener cierta historia
      - min_alpha: nunca desactiva completamente TDA
    """
    # 1. Estos campos se asignan AUTOMÁTICAMENTE en el __init__ generado
    history_len: int = 252
    quantile: float = 0.90
    min_history: int = 30
    min_alpha: float = 0.20

    # La función __post_init__ es un método especial exclusivo de las Python Data Classes (@dataclass).
    # Sirve para ejecutar lógica de inicialización después de que el método __init__ (que la dataclass genera automáticamente) 
    # haya terminado de asignar los valores iniciales.
    # Sin dataclass: Puedes usar el __init__ y poner lógica extra ahí (validaciones, cálculos, crear listas vacías).
    # Con dataclass: Como el __init__ es automático, no puedes "meterte dentro" para agregar código. __post_init__ es la puerta trasera 
    # que te da Python para agregar esa lógica extra.
    def __post_init__(self):
        # Al usar un deque con maxlen, al agregar un dato nuevo, automáticamente expulsa el más antiguo si ya se llenó la ventana.
        # Tmbien al usar una double-ended queue nos permite generaliar una cola y una pila, ya que podemos insertar y remover 
        # elementos desde cualquer extremo
        self._scores: Deque[float] = deque(maxlen=self.history_len)

    # Calcula el percentil 90 (o el que se haya configurado) del historial de scores.
    # Ejemplo: Si los scores pasados oscilan entre 1 y 5, y el percentil 90 es 4.5.
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

        # Comparación: Si Score Actual <= Umbral: El mercado está tranquilo. Devuelve 1.0.
        # Si Score Actual > Umbral: Alerta de cambio de régimen.
        if score <= thr:
            return 1.0

        # Cuanto mayor sea el score respecto a la diferencia, menor será alpha, aunque como mínimo
        # puede valer min_alpha
        # Se aplica un weight decay, si el score es exactamente el umbral, alpha = 1. Si el score es 
        # el doble del umbral, alpha = 0.5.Si el score es enorme, alpha baja hasta topar con min_alpha. 
        x = score / thr
        alpha = 1.0 / x
        alpha = float(np.clip(alpha, self.min_alpha, 1.0))
        return alpha

    def update_history(self, score: Optional[float]) -> None:
        if score is None:
            return
        if np.isfinite(score) and score >= 0.0:
            self._scores.append(float(score))



"""
DETECCIÓN DE RÉGIMEN TOPOLÓGICO (LANDSCAPES + EWMA)
================================================================================

1. QUÉ PROBLEMA RESUELVE
------------------------
La implementación ingenua mide el "cambio topológico" comparando el diagrama de
persistencia de la ventana actual con el de la inmediata anterior (t vs t-1).
Esto equivale a una señal tipo derivada: captura variaciones instantáneas, pero
es extremadamente sensible al ruido y a outliers.

En finanzas, donde existen shocks idiosincráticos y micro-rupturas frecuentes,
este enfoque dispara falsos positivos: no siempre que "hoy difiere de ayer"
estamos ante un nuevo régimen de mercado.

2. POR QUÉ LANDSCAPES (Y NO PROMEDIAR DIAGRAMAS)
------------------------------------------------
El cambio clave es pasar de una comparación instantánea a una pregunta estadística:
"¿Qué tan raro es el comportamiento de hoy respecto a lo normal?".

Para responder a esto, necesitamos un objeto sobre el que tenga sentido calcular
medias móviles. Los diagramas de persistencia no forman, en general, un espacio
vectorial donde promediar sea natural (no puedes sumar diagramas fácilmente).

En cambio, los Persistence Landscapes son funciones (y por discretización, vectores)
en un espacio de Banach/Hilbert donde sí tiene sentido sumar, promediar y aplicar
técnicas estadísticas. Esta es la motivación formal del landscape como
representación estadística (Bubenik, 2015).

3. CÓMO FUNCIONA EL EWMA TOPOLÓGICO
-----------------------------------
En cada rebalanceo:
    1. Se convierten los diagramas D_t a vectores L_t (landscapes discretizados).
    2. Se mantiene un baseline histórico (L_bar) mediante una Media Móvil
       Exponencial (EWMA):

       L_bar_t = (1 - alpha) * L_bar_{t-1} + alpha * L_t

    3. El "Score de Régimen" se define como la distancia (norma L2) entre el
       landscape actual (L_t) y el baseline histórico (L_bar_{t-1}).

Conceptualmente: Si la geometría/correlación del mercado se parece a lo habitual,
la distancia es baja. Si hay una reorganización estructural (crash, rotación),
la distancia aumenta significativamente.

4. ROBUSTEZ Y CONEXIÓN CON EL REGIMECONTROLLER
----------------------------------------------
Este score alimenta el RegimeController, actuando como un gating causal:
    - Si Score > Umbral (ej. cuantil histórico): Se reduce el leverage o se
      aplica shrinkage hacia equal-weight (modo defensivo).
    - Racional: La homología persistente es estable ante perturbaciones pequeñas
      (Cohen-Steiner et al., 2007), por lo que un pico en la distancia de
      landscapes no es ruido, sino una señal de cambio estructural genuino
      (Gidea & Katz, 2018).

5. REFERENCIAS ACADÉMICAS 
------------------------------------
- Bubenik, P. (2015). "Statistical Topological Data Analysis using Persistence
    Landscapes". Journal of Machine Learning Research (JMLR).
    (Justificación: Landscapes como objeto estadístico promediable).

- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). "Stability of
    Persistence Diagrams". Discrete & Computational Geometry.
    (Justificación: Estabilidad de PH ante ruido).

- Gidea, M., Katz, Y. (2018). "Topological Data Analysis of Financial Time
    Series: Landscapes of Crashes".
    (Justificación: Evidencia empírica de normas L-p en landscapes como
    detectores de crisis).
"""
class TopoMovingAverage:
    """
    Mantiene un "baseline" topológico con EWMA sobre Persistence Landscapes y devuelve
    un score de régimen como distancia (normalizada) entre el landscape actual y el baseline.

    Idea:
      score_t = || L(dgm_t) - EWMA(L(dgm_{<t})) ||   (con normalización y mezcla H0/H1)
    """

    def __init__(
        self,
        alpha: float = 0.15,            # EWMA del baseline
        resolution: int = 100,
        num_landscapes: int = 5,
        w0: float = 0.6,                # peso H0
        w1: float = 0.4,                # peso H1
        dist_ema_beta: float = 0.05,    # EWMA para escalar distancias (H0 vs H1)
        eps: float = 1e-12,
    ):
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha debe estar en (0,1].")
        if dist_ema_beta <= 0.0 or dist_ema_beta > 1.0:
            raise ValueError("dist_ema_beta debe estar en (0,1].")
        if w0 < 0 or w1 < 0:
            raise ValueError("w0 y w1 deben ser >= 0.")
        if (w0 + w1) <= 0:
            raise ValueError("w0 + w1 debe ser > 0.")

        # Normaliza pesos por si no suman 1
        s = float(w0 + w1)
        self.w0 = float(w0 / s)
        self.w1 = float(w1 / s)

        self.alpha = float(alpha)
        self.beta = float(dist_ema_beta)
        self.eps = float(eps)

        # Transformadores (separados por dimensión para evitar acoplamientos)
        self._land_h0 = Landscape(resolution=resolution, num_landscapes=num_landscapes)
        self._land_h1 = Landscape(resolution=resolution, num_landscapes=num_landscapes)

        # Dimensión fija del vector
        self._dim = int(resolution * num_landscapes)

        # Estado EWMA (baseline)
        self.avg_h0 = None
        self.avg_h1 = None

        # Escalas EWMA de distancias para normalizar H0/H1 (evita que una domine por escala)
        self.scale_h0 = None
        self.scale_h1 = None

        # Flags para intentar mantener un grid consistente si la clase lo soporta
        self._fitted_h0 = False
        self._fitted_h1 = False

    @staticmethod
    def _clean_diagram(dgm):
        """
        Limpia un diagrama:
          - convierte a np.array
          - elimina death infinitos (crítico en H0)
          - si queda vacío, devuelve (0,2)
        """
        if dgm is None:
            return np.empty((0, 2), dtype=float)

        arr = np.asarray(dgm, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Diagrama inválido: se espera array de forma (n,2).")

        # Quitar infinitos en 'death'
        arr = arr[np.isfinite(arr[:, 1])]
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)

        return arr

    def _vectorize(self, dgm: np.ndarray, which: int) -> np.ndarray:
        """
        Vectoriza un diagrama con Landscape a un vector 1D de tamaño fijo.
        which=0 -> H0, which=1 -> H1
        """
        if dgm is None or len(dgm) == 0:
            return np.zeros(self._dim, dtype=float)

        land = self._land_h0 if which == 0 else self._land_h1
        fitted_flag = "_fitted_h0" if which == 0 else "_fitted_h1"

        # Intento: si existe transform y ya está "fitted", usa transform; si no, fit_transform.
        try:
            is_fitted = getattr(self, fitted_flag)
            if is_fitted and hasattr(land, "transform"):
                vec = land.transform([dgm])[0]
            else:
                vec = land.fit_transform([dgm])[0]
                setattr(self, fitted_flag, True)
        except Exception:
            # Fallback robusto
            vec = land.fit_transform([dgm])[0]
            setattr(self, fitted_flag, True)

        vec = np.asarray(vec, dtype=float).reshape(-1)

        # Control de consistencia dimensional
        if vec.size != self._dim:
            raise ValueError(
                f"Dimensión del landscape cambió (esperado {self._dim}, got {vec.size}). "
                "Asegura parámetros resolution/num_landscapes y pipeline coherentes."
            )
        return vec

    def update_and_score(self, dgms, normalize_dim: bool = True) -> float:
        """
        1) Vectoriza H0/H1 del estado actual.
        2) Score = distancia al baseline histórico (antes de actualizar).
        3) Actualiza baseline EWMA.

        normalize_dim:
          - True: divide norma L2 por sqrt(dim) para que el score no dependa del tamaño del vector.
        """
        if dgms is None or len(dgms) < 2:
            return 0.0

        h0 = self._clean_diagram(dgms[0])
        h1 = self._clean_diagram(dgms[1])

        v0 = self._vectorize(h0, which=0)
        v1 = self._vectorize(h1, which=1)

        # Inicialización (primer punto): baseline = hoy, score=0
        if self.avg_h0 is None:
            self.avg_h0 = v0.copy()
            self.avg_h1 = v1.copy()
            self.scale_h0 = 1.0
            self.scale_h1 = 1.0
            return 0.0

        denom = np.sqrt(self._dim) if normalize_dim else 1.0

        # Distancias al baseline (ANTES del update, causal)
        d0 = float(np.linalg.norm(v0 - self.avg_h0) / denom)
        d1 = float(np.linalg.norm(v1 - self.avg_h1) / denom)

        # Normalización por escala EWMA (evita dominancia H0 vs H1 por magnitud)
        self.scale_h0 = (1.0 - self.beta) * float(self.scale_h0) + self.beta * d0
        self.scale_h1 = (1.0 - self.beta) * float(self.scale_h1) + self.beta * d1

        d0n = d0 / (float(self.scale_h0) + self.eps)
        d1n = d1 / (float(self.scale_h1) + self.eps)

        score = self.w0 * d0n + self.w1 * d1n

        # Update EWMA del baseline (DESPUÉS de medir)
        self.avg_h0 = (1.0 - self.alpha) * self.avg_h0 + self.alpha * v0
        self.avg_h1 = (1.0 - self.alpha) * self.avg_h1 + self.alpha * v1

        return float(score)
