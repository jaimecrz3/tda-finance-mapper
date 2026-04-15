from typing import List, Optional, Deque
from collections import deque
import numpy as np
from gudhi.representations import Landscape

def compute_landscape_norm(dgms: List[np.ndarray], dimension: int = 1, resolution: int = 100, num_landscapes: int = 5) -> float:
    """
    Calcula la norma L2 del Persistence Landscape para una dimensión topológica dada (por defecto H1).
    Basado en Gidea & Katz (2018): Los picos en la norma de los landscapes anticipan inestabilidad.
    """
    if dgms is None or len(dgms) <= dimension:
        return 0.0
        
    dgm = dgms[dimension]
    
    # Limpieza: quitar infinitos (puntos que nunca mueren)
    if dgm is not None and len(dgm) > 0:
        dgm = dgm[np.isfinite(dgm[:, 1])]
    
    if dgm is None or len(dgm) == 0:
        return 0.0

    # Generar el Landscape
    # 1. Creamos el "molde" del Landscape. 
    # resolution=100 significa que discretiza la función continua en 100 puntos.
    # num_landscapes=5 significa que extrae las 5 capas principales del paisaje.
    land = Landscape(resolution=resolution, num_landscapes=num_landscapes)

    # 2. Transforma el diagrama (dgm) en un vector que representa el Persistence Landscape
    vec = land.fit_transform([dgm])[0]
    
    # 3. Ahora sí, como 'vec' es un vector en un espacio vectorial estándar, 
    # podemos calcular la norma L2
    return float(np.linalg.norm(vec))


class TopologicalAnomalyDetector:
    """
    Mantiene un historial de las normas de los Landscapes.
    Si la norma actual es anómalamente alta (supera el cuantil especificado),devuelve una señal de 
    PELIGRO (False), indicando que es mejor estar en liquidez (Cash), es decir, no invertimos.

    Se ha optado por un umbral dinámico basado en una ventana móvil de 1 año (HACER ANALISIS DE SENSIBILIDAD). En el análisis de series 
    temporales financieras, los mercados son inherentemente no estacionarios. Una ventana histórica infinita generaría un umbral estático
    incapaz de adaptarse a los cambios de régimen macroeconómicos a largo plazo. Al limitar la memoria topológica al último año natural, 
    el algoritmo normaliza la norma L_2 respecto a las condiciones locales del mercado, detectando anomalías relativas en lugar de absolutas.
    """
    def __init__(self, history_len: int = 12, danger_quantile: float = 0.90, min_history: int = 10):
        self.history_len = history_len
        self.danger_quantile = danger_quantile
        self.min_history = min_history
        # Con maxlen=history_len, esta lista borra automáticamente el dato más antiguo cuando entra uno nuevo
        self._norms: Deque[float] = deque(maxlen=history_len) 

    def is_market_safe(self, current_norm: float) -> bool:
        """
        Devuelve True si el mercado es topológicamente estable (invertir usando Mapper).
        Devuelve False si hay anomalía topológica (ir a Cash/Liquidez).
        """
        if current_norm <= 0:
            return True # Si no hay datos topológicos, asumimos normalidad

        # Si no tenemos suficiente historial, no podemos detectar anomalías, somos optimistas
        if len(self._norms) < self.min_history:
            self._norms.append(current_norm)
            return True

        # Calcular el umbral de peligro basado en la historia pasada
        threshold = np.quantile(list(self._norms), self.danger_quantile)
        
        # Guardamos el valor actual para el futuro
        self._norms.append(current_norm)

        # ¿Supera la norma actual el umbral de peligro?
        return current_norm <= threshold