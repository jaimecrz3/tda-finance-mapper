import kmapper as km
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 1) Datos (ejemplo)
X, y = datasets.load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

# 2) Crear objeto Mapper
mapper = km.KeplerMapper(verbose=1)

# 3) Elegir "lens" (función filtro). Aquí: PCA a 2D (solo para el lens)
lens = mapper.fit_transform(X, projection=PCA(n_components=2, random_state=42))

# 4) Cover: número de cubos (intervalos) y solape
cover = km.Cover(n_cubes=15, perc_overlap=0.3)

# 5) Clustering local dentro de cada cubo
clusterer = DBSCAN(metric="cosine")

# 6) Construcción del grafo
graph = mapper.map(lens, X, cover=cover, clusterer=clusterer)

# 7) Visualización (HTML)
mapper.visualize(
    graph,
    path_html="mapper_digits.html",
    title="Mapper (Digits)",
    color_values=y,               # colorear por etiqueta real (para inspección)
    color_function_name="label"
)

print("Generado mapper_digits.html")


