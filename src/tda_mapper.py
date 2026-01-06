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
from sklearn.cluster import DBSCAN
from umap import UMAP


# -------------------------------------------------------------------
# 1) Parámetros del "pipeline" de TDA (Mapper)
# -------------------------------------------------------------------
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

    min_assets: int = 5


# -------------------------------------------------------------------
# 2) Construcción de clusters vía Mapper a partir de precios
# -------------------------------------------------------------------
def build_clusters_from_prices(
    prices_window: pd.DataFrame, #informativo, no imone el tipo en tiempo de ejecución
    params: MapperParams
) -> Optional[List[List[List[str]]]]: #Tipo de retorno, optional significa que puededevover None
    """
    Entrada:
      prices_window: DataFrame con index=fechas y columnas=tickers; valores=precios (idealmente Adj Close).
        Nota: El adjusted close es una versió modifcada del precio de cierre de una ación que tiene en cuenta 
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

    # ---------------------------------------------------------------
    # 2.1) Preparación de datos: ordenar y rellenar hacia delante
    # ---------------------------------------------------------------
    # ffill: rellena NaNs usando el último valor conocido hacia adelante.
    # Importante: NO usamos bfill (relleno hacia atrás) para no "usar" valores futuros
    # en periodos anteriores (aunque sea solo imputación, metodológicamente es mejor evitarlo).
    #
    # dropna(axis=1, how="any"): elimina tickers que tengan algún NaN residual en la ventana.
    # Esto fuerza que todos los activos usados tengan serie completa para el periodo.
    prices = prices_window.sort_index().ffill().dropna(axis=1, how="any")

    # Si quedan muy pocos activos, no intentamos clusterizar
    if prices.shape[1] < params.min_assets:
        return None

    # ---------------------------------------------------------------
    # 2.2) Pasar de precios a retornos (log-returns)
    # ---------------------------------------------------------------
    # log_returns(t) = log(P_t / P_{t-1})
    # - reduce efecto de escala (una acción de 20$ y otra de 2000$)
    # - para correlación es más natural usar retornos que precios
    log_ret_t = np.log(prices / prices.shift(1)).dropna()

    # Si tenemos muy pocas observaciones temporales, la correlación será inestable
    if log_ret_t.shape[0] < 10:
        return None

    # Transponemos para que cada fila sea un ticker (un "punto" en Mapper)
    # y cada columna sea el tiempo (features).
    # Luego lo que hacemos es tratar caa ticker como un "punto" con muchas
    # features(los retornos a lo largo del tiempo)
    log_returns = log_ret_t.T  # filas=tickers, columnas=tiempo

    if log_returns.shape[0] < params.min_assets:
        return None

    # ---------------------------------------------------------------
    # 2.3) Lens / proyección: PCA -> UMAP
    # ---------------------------------------------------------------
    # KeplerMapper trabaja así:
    #   1) proyecta cada punto a un espacio (lens)
    #   2) crea un cover sobre ese lens (intervalos/celdas con solapamiento)
    #   3) en cada celda hace clustering local en el espacio original
    #
    # Aquí usamos un pipeline de proyección:
    #   - PCA conserva ~80% varianza: reduce ruido y dimensión antes de UMAP
    #   - UMAP a 1D: nos da un lens 1D, más simple y estable para el cover
    mapper = km.KeplerMapper()

    projected = mapper.fit_transform(
        log_returns,
        projection=[
            PCA(n_components=params.pca_var, random_state=params.random_state),
            UMAP(n_components=params.umap_dim, random_state=params.random_state, n_jobs=1),
        ]
    )

    # ---------------------------------------------------------------
    # 2.4) Cover: define cómo troceamos el lens
    # ---------------------------------------------------------------
    # n_cubes: cuántas celdas/intervalos
    # perc_overlap: cuánto se solapan
    # El solapamiento es lo que permite que un ticker aparezca en varios nodos
    # (y eso genera aristas entre nodos).
    cover = km.Cover(n_cubes=params.n_cubes, perc_overlap=params.perc_overlap)

    # ---------------------------------------------------------------
    # 2.5) Clustering local dentro de cada celda (DBSCAN)
    # ---------------------------------------------------------------
    # DBSCAN agrupa puntos según densidad y puede dejar "ruido" (no asignados).
    # metric="correlation":
    #   la distancia depende de correlación entre series de retornos,
    #   es decir, agrupa tickers que se mueven de forma parecida (co-movimiento).
    #
    # eps/min_samples controlan qué tan exigente es DBSCAN.
    graph = mapper.map(
        projected,
        log_returns,
        cover=cover,
        clusterer=DBSCAN(
            metric="correlation",
            eps=params.dbscan_eps,
            min_samples=params.dbscan_min_samples,
            n_jobs=1
        )
    )

    # ---------------------------------------------------------------
    # 2.6) Extraer nodos y links del grafo de Mapper
    # ---------------------------------------------------------------
    # graph["nodes"]: dict node_id -> lista de índices (filas) de log_returns
    # graph["links"]: dict node_id -> lista de vecinos (por solapamiento)
    nodes = graph.get("nodes", {})
    links = graph.get("links", {})

    # Si no hay nodos, falló el clustering / cover (o todo quedó como ruido)
    if not nodes:
        return None

    # ---------------------------------------------------------------
    # 2.7) Convertir el grafo de Mapper a un grafo NetworkX
    # ---------------------------------------------------------------
    # Esto lo hacemos para calcular "componentes conexas".
    # Componentes conexas = grupos de nodos conectados por aristas.
    #
    # Interpretación:
    #   - un grupo grande de nodos conectados implica que hay solapamientos
    #     consistentes entre clusters locales, lo que sugiere una "macro-estructura".
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    for n, nbrs in links.items():
        for m in nbrs:
            G.add_edge(n, m)

    components = list(nx.connected_components(G))
    tickers = list(log_returns.index)  # nombres de fila = tickers

    # ---------------------------------------------------------------
    # 2.8) Convertir componentes -> estructura anidada de tickers
    # clustered acaba siendo algo como:
    # clustered = [
    #      [["AAPL", "MSFT"], ["GOOG"]],
    #      [["TSLA"], ["AMZN", "META"]]
    # ]
    # ---------------------------------------------------------------
    clustered: List[List[List[str]]] = []
    for comp in components:
        giant: List[List[str]] = []  # componente conexa (giant cluster)

        for node_id in comp:
            idxs = nodes.get(node_id, [])
            if not idxs:
                continue

            # Pasamos de índices (posiciones) a tickers reales
            small = [tickers[i] for i in idxs if 0 <= i < len(tickers)]

            # Dedupe: en un nodo podría repetirse (raro, pero mejor robustez)
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

    # ---------------------------------------------------------------
    # 2.9) Cobertura: asegurar que ningún ticker se pierda
    # ---------------------------------------------------------------
    # Según cover+DBSCAN, algunos tickers podrían no acabar en ningún nodo útil.
    # Si los ignoras, desaparecen del reparto de pesos => sesgo.
    #
    # Aquí los añadimos como singletons (clusters de 1 elemento).
    covered = set()
    for giant in clustered:
        for small in giant:
            #update añade todo los elementos de small a covered elimando duplicados 
            covered.update(small) 

    #Ahora covered tiene todos los tickers de cada small sin duplicados
    missing = [s for s in tickers if s not in covered]
    if missing: # Una lista vacía es false, una lista con al menos un elemento es true
        # Se produce una lista como:
        # [
        #    ["AAPL"],
        #    ["TSLA"],
        #    ["META"]
        #]
        # y se añade ese bloque a clustered
        clustered.append([[s] for s in missing])

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
def weight_distribution(clustered_symbols: List[List[List[str]]]) -> Dict[str, float]:
    """
    Convierte la estructura clustered_symbols en un diccionario {ticker: peso}.

    Intuición:
    - Recorremos recursivamente la estructura.
    - En cada nivel repartimos el "share" del padre entre los hijos.
    - Penalizamos por profundidad con (2 ** (level - 1)):
        => cuanto más profundo, menor contribución marginal.

    Importante:
    - En Mapper, un ticker puede aparecer en varios nodos (por solapamiento).
      Como aquí acumulamos pesos (weights[ticker] += ...), un ticker que aparece
      en varios nodos puede acabar con más peso final (soft membership).
    """
    weights: Dict[str, float] = {}

    def assign(nested_list, level: int = 1, parent_share: float = 1.0):
        # Caso base: lista vacía
        if not nested_list:
            return

        n = len(nested_list)
        if n == 0:
            return

        # Reparto igualitario del share del padre entre los hijos de este nivel
        share_per_child = parent_share / n

        for item in nested_list:
            if isinstance(item, list):
                # Si el hijo es otra lista, bajamos un nivel.
                # Penalizamos por profundidad: / (2 ** (level - 1))
                assign(item, level + 1, share_per_child / (2 ** (level - 1)))
            else:
                # Si el hijo no es lista, asumimos que es un ticker (hoja).
                # Sumamos contribución (acumulativa) al peso del ticker.
                weights[item] = weights.get(item, 0.0) + share_per_child / (2 ** (level - 1))

    # Iniciamos con parent_share=1.0 (100% del presupuesto de pesos)
    assign(clustered_symbols)

    # Normalización final para asegurar suma(weights)=1
    total = float(sum(weights.values()))
    if total <= 0:
        return {}

    return {k: v / total for k, v in weights.items()}


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
    # Para TFG, esta versión simple suele ser suficiente.
    return capped

"""
# 3) Distribución de pesos a partir de la estructura anidada, VARIANTE 2
#
# Definición:
# Sea G el nº de giant clusters (componentes conexas).
# Cada giant cluster recibe 1/G.
# Dentro de cada giant cluster, recoges símbolos únicos y repartes uniformemente.
# Propiedad: elimina el sesgo por tamaño de small clusters. Solo importa el “macro-cluster”.
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

#     for giant in clustered_symbols:
#         # símbolos únicos dentro del giant cluster
#         unique: List[str] = []
#         seen: Set[str] = set()
#         for small in giant:
#             for s in small:
#                 if s not in seen:
#                     seen.add(s)
#                     unique.append(s)

#         if not unique:
#             continue

#         giant_share = 1.0 / G
#         per_symbol = giant_share / len(unique)

#         for s in unique:
#             weights[s] = weights.get(s, 0.0) + per_symbol

#     # Renormaliza (por si había giant clusters vacíos)
#     total = sum(weights.values())
#     if total <= 0:
#         return {}

#     weights = {k: v / total for k, v in weights.items()}
#     return apply_cap_and_renormalize(weights, max_weight=max_weight)

"""
# 3) Distribución de pesos a partir de la estructura anidada, VARIANTE 3
#
# Aquí sí mantenemos la idea de “repartir en árbol”, pero en cada nivel:
# en vez de share_per_child = parent_share / n,
# hacemos share_child = parent_share * (leaf_count(child) / leaf_count(parent)).
# Eso garantiza que un subcluster con 20 símbolos recibe (aprox) 10x más
# presupuesto que uno con 2 símbolos.
"""
# Nested = Union[str, List["Nested"]]

# def leaf_count(node: Nested) -> int:
#     """Cuenta hojas únicas bajo un nodo (evita duplicados)."""
#     seen: Set[str] = set()

#     def walk(x: Nested):
#         if isinstance(x, list):
#             for y in x:
#                 walk(y)
#         else:
#             seen.add(x)

#     walk(node)
#     return len(seen)

# def weight_distribution(
#     clustered_symbols: List[List[List[str]]],
#     max_weight: float = 0.03,
#     depth_penalty: bool = False
# ) -> Dict[str, float]:
#     """
#     Reparte parent_share proporcional al número de hojas únicas de cada hijo.
#     Si depth_penalty=True, añade penalización suave por profundidad (opcional).
#     """
#     if not clustered_symbols:
#         return {}

#     weights: Dict[str, float] = {}

#     def assign(node: Nested, parent_share: float, level: int = 1):
#         if parent_share <= 0:
#             return

#         if not isinstance(node, list):
#             # hoja
#             w = parent_share
#             if depth_penalty:
#                 w = w / (2 ** (level - 1))
#             weights[node] = weights.get(node, 0.0) + w
#             return

#         if len(node) == 0:
#             return

#         # calcula leaf counts por hijo
#         counts = []
#         for child in node:
#             c = leaf_count(child)
#             counts.append(c)

#         total_leaves = sum(counts)
#         if total_leaves <= 0:
#             return

#         for child, c in zip(node, counts):
#             child_share = parent_share * (c / total_leaves)
#             assign(child, child_share, level + 1)

#     assign(clustered_symbols, parent_share=1.0, level=1)

#     total = sum(weights.values())
#     if total <= 0:
#         return {}

#     weights = {k: v / total for k, v in weights.items()}
#     return apply_cap_and_renormalize(weights, max_weight=max_weight)
