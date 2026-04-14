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


"""
# 3) Distribución de pesos a partir de la estructura anidada, VARIANTE 1
#
# Problema: cada elemento hijo del nivel actual recibe exactamente el 
# mismo “share”, independientemente de si ese hijo representa:
# un subcluster con 2 símbolos, o un subcluster con 20 símbolos.
# 
# Ejemplo simple (sin penalización por profundidad para entender la idea)
# Imagina que en un nivel tienes 2 small clusters dentro de un giant cluster:
# Small1 = [A, B] (2 símbolos)
# Small2 = [C, D, E, ..., V] (20 símbolos). Esta función hace:
# share_per_child = 1/2 para cada small cluster.
# Luego, dentro de cada small cluster, cada símbolo recibe aprox:
# En Small1: cada símbolo ≈ (1/2) / 2 = 1/4
# En Small2: cada símbolo ≈ (1/2) / 20 = 1/40
# Resultado: A y B pesan 10 veces más que cualquier símbolo del cluster grande.
# Eso es lo que se llama sesgo hacia clusters pequeños
"""
# def weight_distribution(clustered_symbols: List[List[List[str]]]) -> Dict[str, float]:
#     """
#     Convierte la estructura clustered_symbols en un diccionario {ticker: peso}.

#     Intuición:
#     - Recorremos recursivamente la estructura.
#     - En cada nivel repartimos el "share" del padre entre los hijos.
#     - Penalizamos por profundidad con (2 ** (level - 1)):
#         => cuanto más profundo, menor contribución marginal.

#     Importante:
#     - En Mapper, un ticker puede aparecer en varios nodos (por solapamiento).
#       Como aquí acumulamos pesos (weights[ticker] += ...), un ticker que aparece
#       en varios nodos puede acabar con más peso final (soft membership).
#     """
#     weights: Dict[str, float] = {}

#     def assign(nested_list, level: int = 1, parent_share: float = 1.0):
#         # Caso base: lista vacía
#         if not nested_list:
#             return

#         n = len(nested_list)
#         if n == 0:
#             return

#         # Reparto igualitario del share del padre entre los hijos de este nivel
#         share_per_child = parent_share / n

#         for item in nested_list:
#             if isinstance(item, list):
#                 # Si el hijo es otra lista, bajamos un nivel.
#                 # Penalizamos por profundidad: / (2 ** (level - 1))
#                 assign(item, level + 1, share_per_child / (2 ** (level - 1)))
#             else:
#                 # Si el hijo no es lista, asumimos que es un ticker (hoja).
#                 # Sumamos contribución (acumulativa) al peso del ticker.
#                 weights[item] = weights.get(item, 0.0) + share_per_child / (2 ** (level - 1))

#     # Iniciamos con parent_share=1.0 (100% del presupuesto de pesos)
#     assign(clustered_symbols)

#     # Normalización final para asegurar suma(weights)=1
#     total = float(sum(weights.values()))
#     if total <= 0:
#         return {}

#     return {k: v / total for k, v in weights.items()}


def apply_cap_and_renormalize(weights: Dict[str, float], max_weight: float = 0.03) -> Dict[str, float]:
    if not weights:
        return {}

    # Cap
    capped = {k: min(float(v), max_weight) for k, v in weights.items() if v > 0}

    total = sum(capped.values())
    if total <= 0:
        return {}

    # Renormaliza a 1
    capped = {k: v / total for k, v in capped.items()}

    # Ojo: tras renormalizar, algún peso podría volver a superar el cap si muchos quedaron capados.
    # Si quieres respetar el cap estrictamente, necesitas un "water-filling" iterativo.
    return capped

"""
# 3) Distribución de pesos a partir de la estructura anidada, VARIANTE 2
#
# Definición:
# Sea G el nº de giant clusters (componentes conexas).
# Cada giant cluster recibe 1/G.
# Dentro de cada giant cluster, recoges símbolos únicos y repartes uniformemente.
# Propiedad: elimina el sesgo por tamaño de small clusters. Solo importa el “macro-cluster”.
# Riesgo: Si por ejemplo tienes ds Giant, uno con 8 small y otro con solo 1 small
# ambos reciben un 50% del capital
"""
# def weight_distribution(
#     clustered_symbols: List[List[List[str]]],
#     max_weight: float = 0.03
# ) -> Dict[str, float]:
#     if not clustered_symbols:
#         return {}

#     G = len(clustered_symbols)
#     if G == 0:
#         return {}

#     weights: Dict[str, float] = {}

#     #v1
#     # for giant in clustered_symbols:
#     #     # símbolos únicos dentro del giant cluster
#     #     unique: List[str] = []
#     #     seen: Set[str] = set()
#     #     for small in giant:
#     #         for s in small:
#     #             if s not in seen:
#     #                 seen.add(s)
#     #                 unique.append(s)

#     #     if not unique:
#     #         continue

#     #     giant_share = 1.0 / G
#     #     per_symbol = giant_share / len(unique)

#     #     for s in unique:
#     #         weights[s] = weights.get(s, 0.0) + per_symbol

    
#     #v2
#     for giant in clustered_symbols:
#         # contar apariciones dentro del giant
#         appearances: Dict[str, int] = {}
#         for small in giant:
#             for s in small:
#                 appearances[s] = appearances.get(s, 0) + 1

#         if not appearances:
#             continue

#         giant_share = 1.0 / G

#         # A = total de apariciones (suma de counts)
#         A = sum(appearances.values())
#         if A <= 0:
#             continue

#         # Reparto que conserva masa: sum_s (giant_share * appearances[s]/A) = giant_share
#         for s, c in appearances.items():
#             weights[s] = weights.get(s, 0.0) + giant_share * (c / A)

#     # Renormaliza (por si había giant clusters vacíos)
#     total = sum(weights.values())
#     if total <= 0:
#         return {}

#     weights = {k: v / total for k, v in weights.items()}
#     #return apply_cap_and_renormalize(weights, max_weight=max_weight)

#     return weights

"""
# 3) Distribución de pesos a partir de la estructura anidada, VARIANTE 3
#
# Aquí sí mantenemos la idea de “repartir en árbol”, pero en cada nivel:
# en vez de share_per_child = parent_share / n,
# hacemos share_child = parent_share * (leaf_count(child) / leaf_count(parent)).
# Eso garantiza que un subcluster con 20 símbolos recibe (aprox) 10x más
# presupuesto que uno con 2 símbolos.
"""
Nested = Union[str, List["Nested"]]

def leaf_count(node: Nested) -> int:
    """Cuenta hojas únicas bajo un nodo (evita duplicados)."""
    seen: Set[str] = set()

    def walk(x: Nested):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            seen.add(x)

    walk(node)
    return len(seen)

def weight_distribution(
    clustered_symbols: List[List[List[str]]],
    max_weight: float = 0.03,
    depth_penalty: bool = True
) -> Dict[str, float]:
    """
    Reparte parent_share proporcional al número de hojas únicas de cada hijo.
    Si depth_penalty=True, añade penalización suave por profundidad (opcional).
    """
    if not clustered_symbols:
        return {}

    weights: Dict[str, float] = {}

    def assign(node: Nested, parent_share: float, level: int = 1):
        if parent_share <= 0:
            return

        if not isinstance(node, list):
            # hoja
            w = parent_share
            if depth_penalty:
                w = w / (2 ** (level - 1))
            weights[node] = weights.get(node, 0.0) + w
            return

        if len(node) == 0:
            return

        # calcula leaf counts por hijo
        counts = []
        for child in node:
            c = leaf_count(child)
            counts.append(c)

        total_leaves = sum(counts)
        if total_leaves <= 0:
            return

        for child, c in zip(node, counts):
            child_share = parent_share * (c / total_leaves)
            assign(child, child_share, level + 1)

    assign(clustered_symbols, parent_share=1.0, level=1)

    total = sum(weights.values())
    if total <= 0:
        return {}

    weights = {k: v / total for k, v in weights.items()}
    #return apply_cap_and_renormalize(weights, max_weight=max_weight)
    return weights

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def cap_weights_strict(w: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap estricto con redistribución iterativa (water-filling)."""
    if not w:
        return {}
    cap = float(cap)
    w = {k: float(v) for k, v in w.items() if v > 0}
    if not w:
        return {}

    # normaliza inicial
    s = sum(w.values())
    w = {k: v / s for k, v in w.items()}

    while True:
        over = {k for k, v in w.items() if v > cap}
        if not over:
            return w

        fixed_mass = cap * len(over)
        if fixed_mass >= 1.0 - 1e-12:
            # si el cap es demasiado bajo para N activos, devuelve equal-weight capado (lo más cercano)
            n = len(w)
            base = min(cap, 1.0 / n)
            out = {k: base for k in w}
            ss = sum(out.values())
            return {k: v / ss for k, v in out.items()}

        under = [k for k in w if k not in over]
        if not under:
            n = len(w)
            base = min(cap, 1.0 / n)
            out = {k: base for k in w}
            ss = sum(out.values())
            return {k: v / ss for k, v in out.items()}

        # fija los over al cap
        w_fixed = {k: cap for k in over}
        rem_mass = 1.0 - fixed_mass

        # redistribuye rem_mass proporcional a los pesos originales de los under (antes de rescalar)
        under_sum = sum(w[k] for k in under)
        if under_sum <= 0:
            # fallback uniforme en under
            per = rem_mass / len(under)
            w_under = {k: per for k in under}
        else:
            w_under = {k: rem_mass * (w[k] / under_sum) for k in under}

        w = {**w_fixed, **w_under}


from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def weight_distribution_portfolio(
    clustered_symbols: List[List[List[str]]],
    max_weight: float = 0.03,
    gamma_giant: float = 1.0,
    gamma_node: float = 1.0,
    overlap_correction: bool = True,
    returns_window: Optional[pd.DataFrame] = None,
    min_periods_score: int = 12,
) -> Dict[str, float]:
    """
    Reparto de pesos con scoring (opcional) por giant y por nodo.

    Base:
      - giant_share ∝ |G|^gamma_giant
      - node_share  ∝ |N|^gamma_node
      - ticker_share uniforme dentro del nodo
      - overlap_correction: divide contribución por nº de nodos del giant donde aparece el ticker
      - cap estricto opcional

    Extensión (Score por Sharpe histórico, sin rf):
      Para cualquier cluster C (giant o nodo):
        Score(C) = mu_C / sigma_C
      donde mu_C y sigma_C se calculan sobre el retorno equal-weight del cluster por periodo:
        r_C(t) = mean_i r_i(t)

      Giant:
        giant_score = |giant|^gamma_giant * max(Score(giant), 0)

      Nodo:
        node_score  = |node|^gamma_node  * max(Score(node), 0)

      Si returns_window es None: vuelve al esquema de tamaño puro.
      Si con Score todo queda 0 (p.ej. Score<=0 o falta de datos): fallback a tamaño puro.
    """
    if not clustered_symbols:
        return {}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _sharpe_eqw(cols: List[str]) -> float:
        """Sharpe por periodo (mu/sigma) del retorno equal-weight del conjunto cols."""
        if returns_window is None or len(cols) == 0:
            return 0.0
        sub = returns_window[cols]
        r = sub.mean(axis=1, skipna=True).dropna()
        if len(r) < int(min_periods_score):
            return 0.0
        mu = float(r.mean())
        sig = float(r.std(ddof=0))
        if sig <= 0.0:
            return 0.0
        return mu / sig

    # -----------------------------
    # Precompute estructura
    # -----------------------------
    giant_uniqs: List[Set[str]] = []
    membership_counts: List[Dict[str, int]] = []
    node_sets_by_giant: List[List[Set[str]]] = []

    for giant in clustered_symbols:
        uniq = set()
        counts: Dict[str, int] = {}
        node_sets: List[Set[str]] = []

        for node in giant:
            ns = set(node)
            if not ns:
                continue
            node_sets.append(ns)
            uniq |= ns
            for s in ns:
                counts[s] = counts.get(s, 0) + 1

        giant_uniqs.append(uniq)
        membership_counts.append(counts)
        node_sets_by_giant.append(node_sets)

    # Si hay returns_window, ordena y trabaja con las columnas disponibles
    if returns_window is not None:
        returns_window = returns_window.sort_index()

    # -----------------------------
    # Giant scores (tamaño * calidad)
    # -----------------------------
    giant_scores: List[float] = []
    for uniq in giant_uniqs:
        sz = len(uniq)
        if sz <= 0:
            giant_scores.append(0.0)
            continue

        base = float(sz) ** float(gamma_giant)

        if returns_window is None:
            giant_scores.append(base)
        else:
            cols = [c for c in uniq if c in returns_window.columns]
            sh = _sharpe_eqw(cols)
            giant_scores.append(base * max(sh, 0.0))

    # Fallback si todo se anula por Score<=0 / falta de datos
    if sum(giant_scores) <= 0:
        giant_scores = [(len(uniq) ** float(gamma_giant)) if len(uniq) > 0 else 0.0 for uniq in giant_uniqs]

    total_g = float(sum(giant_scores))
    if total_g <= 0:
        return {}

    # -----------------------------
    # Asignación
    # -----------------------------
    weights: Dict[str, float] = {}

    for g_idx, (giant, g_score) in enumerate(zip(clustered_symbols, giant_scores)):
        if g_score <= 0:
            continue

        g_share = g_score / total_g
        node_sets = node_sets_by_giant[g_idx]
        if not node_sets:
            continue

        # Node scores (tamaño * calidad)
        node_scores: List[float] = []
        for ns in node_sets:
            nsz = len(ns)
            if nsz <= 0:
                node_scores.append(0.0)
                continue

            nbase = float(nsz) ** float(gamma_node)

            if returns_window is None:
                node_scores.append(nbase)
            else:
                cols = [c for c in ns if c in returns_window.columns]
                nsh = _sharpe_eqw(cols)
                node_scores.append(nbase * max(nsh, 0.0))

        # Fallback si en este giant los nodos se anulan
        if sum(node_scores) <= 0:
            node_scores = [(len(ns) ** float(gamma_node)) if len(ns) > 0 else 0.0 for ns in node_sets]

        total_n = float(sum(node_scores))
        if total_n <= 0:
            continue

        counts = membership_counts[g_idx]

        for ns, n_score in zip(node_sets, node_scores):
            if n_score <= 0 or not ns:
                continue

            node_share = g_share * (n_score / total_n)
            per = node_share / len(ns)

            for s in ns:
                contrib = per
                if overlap_correction:
                    m = max(1, counts.get(s, 1))
                    contrib /= m
                weights[s] = weights.get(s, 0.0) + contrib

    # Normaliza
    s = float(sum(weights.values()))
    if s <= 0:
        return {}
    weights = {k: v / s for k, v in weights.items()}

    # Cap estricto
    if max_weight is not None and float(max_weight) > 0:
        weights = cap_weights_strict(weights, cap=float(max_weight))

    return weights
