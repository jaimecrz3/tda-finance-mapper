from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

# KeplerMapper: implementación del algoritmo Mapper (TDA)
import kmapper as km

# networkx: para operar con grafos (componentes conexas, etc.)
import networkx as nx

# PCA + DBSCAN (sklearn) y UMAP (umap-learn) para la parte de proyección y clustering local
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.base import BaseEstimator, ClusterMixin
from umap import UMAP



# 1) Parámetros 
@dataclass(frozen=True)
class MapperParams:
    """
    Contenedor inmutable (frozen=True) para los hiperparámetros.
    Para evitar pasar muchos argumentos sueltos y mejoras reproducibilidad.

    - pca_var: PCA conserva las componentes necesarias para explicar este % de varianza (0.80 = 80%).
    - umap_dim: dimensión del "lens" final usado por Mapper (1D suele ser más estable/interpretables).
    - random_state: semilla para reproducibilidad de PCA/UMAP.

    - n_cubes / perc_overlap: definen el "cover" del lens:
        n_cubes: número de intervalos/celdas en los que se parte el espacio del lens.
        perc_overlap: solapamiento entre celdas; permite que un punto aparezca en varios nodos.

    - dbscan_eps / dbscan_min_samples: hiperparámetros de DBSCAN para clustering local dentro de cada celda.
    - min_assets: mínimo de activos/tickers para intentar construir clusters (si hay menos, devolvemos None).
    """
    pca_var: float = 0.80
    umap_dim: int = 1
    random_state: int = 1

    n_cubes: int = 10
    perc_overlap: float = 0.2

    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2

    # Selección del algoritmo de clustering local en cada celda del cover.
    # - "dbscan": si encuentra un punto (un activo financiero) que está muy lejos de los demás, lo toma como ruido y le pone una etiqueta de -1 (lo descarta)
    #               Si por ej el algoritmo descarta a Apple porque se mueve distinto al resto, estariamos quitando una empresa muy importante
    # - "haca": hierarchical agglomerative clustering (sin "ruido") sobre distancia de correlación (1 - corr)
    #           Nunca descarta a nadie, si un activo es muy raro, lo pone en un cluster solitario
    clusterer: str = "dbscan"

    # HACA: umbral de distancia (1 - corr). Valores típicos: 0.3–1.0
    haca_distance_threshold: float = 0.7
    haca_linkage: str = "average"

    min_assets: int = 5


# 1.1) Clusterer alternativo: HACA (Agglomerative) sobre distancia de correlación
# Necesitamos hacerlo porque KeplerMapper solo acepta algoritmos que tengan la estructura oficial de scikit-learn, con metodos como fit. 
# Al heredar de BaseEstimator, heredamos esos métodos automáticamente y Mapper la acepta.
# Haca entiende de distancias no de correlaciones:
# Si dos activos se mueven exactamente igual (Correlación = 1), la distancia es 1 - 1 = 0 (son el mismo punto).
# Si no tienen nada que ver (Correlación = 0), la distancia es 1 - 0 = 1 (Están lejos).
# Si se mueven exactamente al revés (Correlación = -1), la distancia es 1 - (-1) = 2 (están en polos opuestos).
class HACAClusterer(BaseEstimator, ClusterMixin):
    """
    Wrapper Agglomerative para KeplerMapper con distancia de correlación.
    Hereda BaseEstimator -> KeplerMapper puede llamar get_params() sin error.
    """
    def __init__(self, distance_threshold: float = 0.7, linkage: str = "average"):
        self.distance_threshold = float(distance_threshold)
        self.linkage = str(linkage)

    def fit(self, X, y=None):
        # KeplerMapper normalmente usa fit_predict, pero dejamos fit por compatibilidad sklearn.
        return self

    def fit_predict(self, X, y=None):
        X = np.asarray(X)
        n = X.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.zeros(1, dtype=int)

        # Si no hay suficientes features para correlación, devolvemos un único cluster
        if X.shape[1] < 2:
            return np.zeros(n, dtype=int)

        # Distancia de correlación: d = 1 - corr
        corr = np.corrcoef(X)
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)

        dist = 1.0 - corr
        # garantizar que la distancia de un activo consigo mismo sea siempre un cero
        np.fill_diagonal(dist, 0.0)

        # Compatibilidad sklearn (metric vs affinity)
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )
        except TypeError:
            # sklearn antiguo
            model = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )

        return model.fit_predict(dist)


# 2) Construcción de clusters vía Mapper a partir de precios
def build_clusters_from_prices(
    prices_window: pd.DataFrame, #informativo, no impone el tipo en tiempo de ejecución
    params: MapperParams
) -> Optional[List[List[List[str]]]]: #Tipo de retorno, optional significa que puede devolver None
    """
    Entrada:
      prices_window: DataFrame con index=fechas y columnas=tickers; valores=precios (idealmente Adj Close).
        Nota: El adjusted close es una versión modifcada del precio de cierre de una ación que tiene en cuenta 
        acciones corporativas como splits e acciones, dividendos y emisiones de derechos.  
      params: hiperparámetros del pipeline.

    Salida:
      Una estructura anidada de clusters, o None si no se puede construir:
        clustered = [
            giant_cluster_1,
            giant_cluster_2,
            ...
        ]

      donde cada giant_cluster_i es una lista de "small clusters" (nodos de Mapper):
        giant_cluster_i = [
            [tickerA, tickerB, ...],   # small cluster (nodo)
            [tickerC, tickerD, ...],   # otro nodo
            ...
        ]

    Intuición:
      - Mapper crea un grafo:
          Nodos: clusters locales dentro de cada región del cover
          Aristas: conectan nodos que comparten miembros (por solapamiento)
      - Las "giant clusters" aquí son componentes conexas del grafo de nodos.
    """

    # 2.1) Preparación de datos: ordenar con sort_index y rellenar hacia delante con
    # ffill: rellena NaNs usando el último valor conocido hacia adelante.
    # Importante: NO usamos bfill (relleno hacia atrás) para no "usar" valores futuros (Data Leakage)
    # en periodos anteriores (aunque sea solo imputación, metodológicamente es mejor evitarlo).
    #
    # dropna(axis=1, how="any"): elimina tickers que tengan algún NaN residual en la ventana.
    # Esto fuerza que todos los activos usados tengan serie completa para el periodo.
    prices = prices_window.sort_index().ffill().dropna(axis=1, how="any")
    #prices = prices_window.sort_index().ffill()

    # prices.shape[1] es el numero de columnas (activos)
    # Si quedan muy pocos activos, no intentamos clusterizar
    if prices.shape[1] < params.min_assets:
        return None

    
    # 2.2) Pasar de precios a retornos (log-returns)
    # log_returns(t) = log(P_t / P_{t-1})
    # - reduce efecto de escala (una acción de 20$ y otra de 2000$)
    # - para correlación es más natural usar retornos que precios
    log_ret_t = np.log(prices / prices.shift(1)).dropna()

    # Si tenemos muy pocas observaciones temporales, la correlación será inestable
    if log_ret_t.shape[0] < 10:
        return None

    # Transponemos para que cada fila sea un ticker (un "punto" en Mapper)
    # y cada columna sea el tiempo (features).
    # Luego lo que hacemos es tratar cada ticker como un "punto" con muchas
    # features(los retornos a lo largo del tiempo)
    log_returns = log_ret_t.T  # filas=tickers, columnas=tiempo

    # Si quedan muy pocos activos, no intentamos clusterizar
    if log_returns.shape[0] < params.min_assets:
        return None

    
    # 2.3) Lens / proyección: PCA -> UMAP
    #
    # KeplerMapper trabaja así:
    #   1) proyecta cada punto a un espacio (lens)
    #   2) crea un cover sobre ese lens (intervalos/celdas con solapamiento)
    #   3) en cada celda hace clustering local en el espacio original
    #
    # Aquí usamos un pipeline de proyección:
    #   - PCA conserva ~80% varianza: reduce ruido y dimensión antes de UMAP
    #   - UMAP a 1D: nos da un lens 1D, más simple y estable para el cover
    # ---------------------------------------------------------------
    # Cada fila = ticker, columnas = tiempo.
    # Normalizamos por fila(z-score por fila) que elimina el efecto de escala/volatilidad: 
    # sin ella, PCA/UMAP tienden a separar activos “por volatilidad” más que por patrón de co-movimiento.
    # Ejemplo: Si comparamos un bono que se mueve 0.1% al día con una criptomoneda que se mueve 10% al día, sin normalizar, 
    # los algoritmos matemáticos (como PCA) ven a la criptomoneda moverse tanto que agrupan todo alrededor de ella.
    # Al hacer el Z-Score por fila, convertimos el bono y la cripto a la misma escala. Ahora el algoritmo ya no mira cuánto se mueven, 
    # sino la forma en que se mueven. Si ambos suben y bajan los mismos días, los agrupará juntos, ignorando que uno es más agresivo que el otro.
    X = log_returns.values.astype(float)

    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, ddof=0, keepdims=True)
    sigma = np.where(sigma == 0.0, 1.0, sigma)  # evita división por cero

    Xz = (X - mu) / sigma # Normalizamos
    log_returns_z = pd.DataFrame(Xz, index=log_returns.index, columns=log_returns.columns)

    # Si le das 60 meses de historia, matemáticamente eso significa que los activos viven en un espacio de 60 dimensiones. Con PCA y UMAP aplastamos esas 60 dimensiones en una sola línea (1D).
    # Fase 1 (PCA): El Análisis de Componentes Principales es rápido y lineal. Se queda solo con los factores principales que explican el 80% (pca_var=0.80) de todo lo que pasa en la bolsa.
    # Fase 2 (UMAP): Es un algoritmo de Inteligencia Artificial moderno y no lineal. Su trabajo es tomar esas 5-10 dimensiones limpias que dejó PCA y dejarlas en una sola dimensión (umap_dim=1), 
    # forzando a que las empresas que tienen alta correlación acaben físicamente cerca unas de otras en ese eje 1D.
    mapper = km.KeplerMapper()
    projected = mapper.fit_transform(
        log_returns_z,
        projection=[
            PCA(n_components=params.pca_var, random_state=params.random_state),
            UMAP(n_components=params.umap_dim, random_state=params.random_state, n_jobs=1, metric="correlation"),
        ]
    )

    # 2.4) Cover: define cómo troceamos el lens
    # n_cubes: cuántas celdas/intervalos
    # perc_overlap: cuánto se solapan
    # El solapamiento es lo que permite que un ticker aparezca en varios nodos
    # (y eso genera aristas entre nodos).
    cover = km.Cover(n_cubes=params.n_cubes, perc_overlap=params.perc_overlap)

    
    # 2.5) Clustering local en cada celda del cover
    # DBSCAN puede marcar puntos como ruido (-1) -> tickers "missing".
    # HACA (agglomerative) asigna todos los puntos a algún cluster.
    if params.clusterer.lower() == "dbscan":
        clusterer = DBSCAN(
            metric="correlation",
            eps=params.dbscan_eps,
            min_samples=params.dbscan_min_samples,
            n_jobs=1
        )
    elif params.clusterer.lower() in ("haca", "hac", "agglomerative"):
        clusterer = HACAClusterer(
            distance_threshold=params.haca_distance_threshold,
            linkage=params.haca_linkage
        )
    else:
        raise ValueError(f"clusterer no soportado: {params.clusterer}")

    # Devuelve un diccionario de Python (graph) que contiene:
    # graph["nodes"]: Una lista de todos los micro-grupos que encontró. Por ejemplo: "El Nodo 1 tiene a Apple y Microsoft. El Nodo 2 tiene a Exxon y Chevron".
    # graph["links"]: Si Exxon cayó en la frontera de dos cubos, el algoritmo la agrupará en un nodo del Cubo A, y también en un nodo del Cubo B. 
    #                   Al detectar que Exxon está en ambos lados, Mapper dibuja una línea conectando el Nodo A con el Nodo B.
    graph = mapper.map(
        projected,
        log_returns_z,
        cover=cover,
        clusterer=clusterer
    )
    
    # 2.6) Extraer nodos y links del grafo de Mapper
    # graph["nodes"]: dict node_id -> lista de índices (filas) de log_returns
    # graph["links"]: dict node_id -> lista de vecinos (por solapamiento)
    nodes = graph.get("nodes", {})
    links = graph.get("links", {})

    DEBUG = True
    if DEBUG:
        print("=== Mapper graph ===")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of links: {sum(len(v) for v in links.values())}")


    # Si no hay nodos, falló el clustering / cover (o todo quedó como ruido)
    if not nodes:
        return None

    
    # 2.7) Convertir el grafo de Mapper a un grafo NetworkX
    # Esto lo hacemos para calcular "componentes conexas".
    # Componentes conexas = grupos de nodos conectados por aristas.
    #
    # Interpretación:
    #   - un grupo grande de nodos conectados implica que hay solapamientos
    #     consistentes entre clusters locales, lo que sugiere una "macro-estructura".
    G = nx.Graph() # Creamos un grafo vacio
    G.add_nodes_from(nodes.keys()) # Añadimos todos los nodos
    for n, nbrs in links.items(): # Vamos añadiendo las aristas entre los nodos que tienen links
        for m in nbrs:
            G.add_edge(n, m)

    # Buscamos las componentes conexas en el grafo
    components = list(nx.connected_components(G))

    # Guardamos en una lista los nombres reales de las acciones o sectores
    tickers = list(log_returns.index) 

    if DEBUG:
        print("=== Connected components ===")
        print(f"Number of components: {len(components)}")
        print("Component sizes (nodes per component):",
            [len(c) for c in components]) 

    
    # 2.8) Convertir componentes -> estructura anidada de tickers
    # clustered acaba siendo algo como:
    # clustered = [
    #      [["AAPL", "MSFT"], ["GOOG"]],
    #      [["TSLA"], ["AMZN", "META"]]
    # ]
    clustered: List[List[List[str]]] = []
    for i, comp in enumerate(components): # Recorremos las componentes conexas
        if DEBUG:
            print(f"\nComponent {i} | {len(comp)} nodes")
        giant: List[List[str]] = []  # componente conexa (giant cluster)

        for node_id in comp:
            idxs = nodes.get(node_id, [])
            if not idxs:
                continue

            # Pasamos de índices (posiciones) a tickers reales
            small = [tickers[i] for i in idxs if 0 <= i < len(tickers)]
            if DEBUG:
                print(f"  Node {node_id}: {len(small)} tickers")

            # Dedupe: en un nodo podría repetirse (raro, pero mejor robustez)
            # Si un activo entra dos veces al mismo nodo, la función de tesorería le daría el doble de dinero por error.
            seen = set()
            uniq = []
            for s in small:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)

            if uniq:
                # Cada "uniq" es un small cluster (un nodo de Mapper)
                giant.append(uniq)

        if giant:
            clustered.append(giant)

    if DEBUG:
        print(f"Component {i} -> {len(giant)} small clusters")
        print("  Example clusters:", giant[:2])

    
    # 2.9) Cobertura: medir missing 
    covered = set()
    for giant in clustered:
        for small in giant:
            covered.update(small)

    missing = [s for s in tickers if s not in covered] # Para comprobar cuantos activos no incluyo DBSCAN

    if DEBUG:
        print("=== Coverage ===")
        print(f"Total tickers: {len(tickers)}")
        print(f"Covered tickers (pre): {len(covered)} | Missing (pre): {len(missing)}")
        #print(f"Covered tickers (post): {len(covered_post)} | Missing (post): {len(missing_post)}")

        if missing:
            print("Missing examples:", missing[:5])

        print("Fin rebalanceo\n\n\n")

    return clustered

def cap_weights_strict(w: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap estricto con redistribución iterativa (water-filling)."""
    if not w:   # no hace nada si no hay pesos
        return {}
    cap = float(cap)
    w = {k: float(v) for k, v in w.items() if v > 0} # elimina activos con peso negativo o cero
    if not w:   # si despues del filtrado no quedan activos con peso positov no hace nada
        return {}

    # divide todos los pesos por la suma total para asegurar que suman exactamente 1 (es decir, el 100% del capital)
    s = sum(w.values())
    w = {k: v / s for k, v in w.items()}

    while True:
        over = {k for k, v in w.items() if v > cap} # buscamos qué activos superan el límite (cap)
        if not over: # si ninguno lo supera, acabamos
            return w

        fixed_mass = cap * len(over)
        # Si tenemos 5 acciones en la cartera y el límite máximo es del 10% por acción. Es matemáticamente imposible invertir el 
        # 100% del dinero (5 acciones * 10% = 50% máximo). Este bloque detecta esa paradoja y devuelve una distribución segura a 
        # partes iguales para que el código no colapse.
        if fixed_mass >= 1.0 - 1e-12: # El -1e-12 es una salvaguarda contra errores de precisión de los decimales en Python
            n = len(w)
            base = min(cap, 1.0 / n)
            out = {k: base for k in w}
            ss = sum(out.values())
            return {k: v / ss for k, v in out.items()}

        under = [k for k in w if k not in over] # activos que están por debajo del límite
        if not under: # si no hay ninguno, volvemos a repartir a partes iguales
            n = len(w)
            base = min(cap, 1.0 / n)
            out = {k: base for k in w}
            ss = sum(out.values())
            return {k: v / ss for k, v in out.items()}

        # los activos que se pasaban del limite les da el peso limite (cap)
        w_fixed = {k: cap for k in over}
        rem_mass = 1.0 - fixed_mass # masa de capital libre

        # Reparte ese capital libre entre los activos pequeños (under), pero respetando su proporción original. 
        # Si el activo X era el doble de grande que el activo Y, al repartir el sobrante, X recibirá el doble que Y.
        under_sum = sum(w[k] for k in under)
        if under_sum <= 0:
            per = rem_mass / len(under)
            w_under = {k: per for k in under}
        else:
            w_under = {k: rem_mass * (w[k] / under_sum) for k in under}

        # Junta los activos capados con los activos incrementados y vuelve al inicio del while True para 
        # comprobar que nadie nuevo se haya pasado
        w = {**w_fixed, **w_under}

def weight_distribution(
    clustered_symbols: List[List[List[str]]],
    max_weight: float = 0.1
) -> Dict[str, float]:
    """
    Distribución basada en la estructura de Mappe:
    1. Se da el mismo peso a cada nodo (small cluster) del grafo
    2. Se reparte el peso del nodo equitativamente entre sus activos
    3. Los activos en solapamientos (pertenecen a >1 nodo) acumulan peso
    A diferencia de un clustering particional donde cada elemento pertenece a una sola clase, el recubrimiento (cover) de Mapper 
    permite pertenencia múltiple. Asignar peso aditivo a los elementos en las intersecciones actúa como un mecanismo natural para 
    identificar y premiar la centralidad (nodos con muchas conexiones); es decir, los activos más representativos y centrales en la 
    estructura de correlación general
    """
    if not clustered_symbols:
        return {}

    weights: Dict[str, float] = {}
    
    # 1. Contar el número total de nodos (small clusters) en todo el grafo
    total_nodes = sum(len(giant) for giant in clustered_symbols)
    if total_nodes == 0:
        return {}

    # 2. El presupuesto asignado a cada nodo
    weight_per_node = 1.0 / total_nodes

    # 3. Repartir
    for giant in clustered_symbols:
        for small in giant:
            if not small:
                continue
                
            # Reparto equitativo local
            weight_per_asset = weight_per_node / len(small)
            
            for asset in small:
                weights[asset] = weights.get(asset, 0.0) + weight_per_asset

    # 4. Limitar el peso máximo de un solo activo por control de riesgo (ej. 5%)
    return cap_weights_strict(weights, cap=max_weight) # para asegurar que, aunque un activo sea muy central en el grafo, no se lleve demasiado dinero
