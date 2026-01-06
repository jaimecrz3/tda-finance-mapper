# TDA in Finance: Mapper (Bachelor's Thesis)

This repository contains the code and experiments for my Bachelor's thesis on **Topological Data Analysis (TDA)** applied to **high-dimensional financial data**, with a focus on the **Mapper** algorithm (and optional extensions with persistent homology / Ball Mapper).

## Project goal
Build an end-to-end, reproducible pipeline to:
- construct rolling-window representations of market data (returns, volatility, etc.),
- apply **Mapper** to visualize and summarize market structure,
- identify **market regimes** and transitions,
- compare with standard baselines (e.g., PCA + clustering).

## Method overview
1. **Data**: daily prices (indices and/or a basket of stocks).
2. **Preprocessing**: log-returns, rolling features, scaling.
3. **Lens (filter)**: PCA/UMAP projections or market indicators.
4. **Cover**: overlapping intervals (n_intervals, overlap).
5. **Local clustering**: DBSCAN or single-linkage (MST cut).
6. **Mapper graph**: nodes = local clusters, edges = shared points.
7. **Evaluation**: stability across parameters, regime coherence, baselines.

## Repository structure
src/tda/
mapper_pipeline.py
clustering.py
features.py
metrics.py
notebooks/
01_data_download.ipynb
02_feature_engineering.ipynb
03_mapper_experiments.ipynb
data/
processed/ # generated features
results/
figures/
tables/
tests/


## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```
## Reproducibility
- Random seeds are fixed where applicable.
- Main outputs are exported to results/ (figures + CSV/JSON).

## Data sources
- Datasets are downloaded from public sources (e.g., Kaggle / Yahoo Finance) and are used for academic purposes only.

## Disclaimer
- This project is for academic research and does not provide investment advice. Results may be affected by non-stationarity, noise, and overfitting. Avoid look-ahead bias in all experiments.

Ideas para comparar:

1) Métricas de DBSCAN
2) Evaluar si un método de construcción de cartera basado en Topological Data Analysis (TDA) mejora el desempeño ajustado por riesgo y/o la diversificación respecto a dos baselines estándar, manteniendo constantes:

Universo (Top 200 constituyentes de SPY por peso, en cada fecha)

Rango temporal (2020-03-01 a 2025-03-01)

Frecuencia de rebalance (misma para todos)

Costes (mismo modelo de comisiones/slippage)

Tipo de cuenta/brokerage model

Resolución de datos y warm-up

Variable independiente (lo único que cambia): método de asignación de pesos.

Estrategias a comparar (3 backtests controlados)
Estrategia A — Baseline 1: Equal-weight (EW-200)

Definición: cada rebalance, repartir el 100% del capital equitativamente entre los 200 activos seleccionados.

Universo: Top 200 SPY constituents

Pesos: 
𝑤
𝑖
=
1
/
200
w
i
	​

=1/200

Rebalance: cada recalibrate_days (igual que en TDA)

Propósito: baseline “simple” y muy usado; mide si TDA aporta algo más allá de diversificación trivial.

Estrategia B — Baseline 2: Market-cap weight proxy (MCW-200 ≈ SPY)

Definición: aproximar el comportamiento de SPY ponderando cada acción por su “peso en el ETF” en esa fecha.

Universo: Top 200 SPY constituents

Pesos: 
𝑤
𝑖
∝
ETF constituent weight
𝑖
w
i
	​

∝ETF constituent weight
i
	​

, normalizado a sumar 1

Rebalance: cada recalibrate_days

Propósito: comparar contra un benchmark “tipo SPY” pero restringido a los mismos 200 activos (para que la comparación sea justa y “ceteris paribus”).

Nota metodológica: esto no replica exactamente SPY (porque SPY tiene ~500 y tú usas 200), pero es un proxy razonable y, sobre todo, controlado.

Estrategia C — Método propuesto: TDA clustering + equal cluster weights (TDA-ECW)

Definición: construir clústeres de similitud (basados en correlación de retornos y el pipeline TDA) y asignar el capital de forma balanceada entre clústeres (y subclústeres), en vez de balancearlo por acción.

Universo: Top 200 SPY constituents

Clustering: KeplerMapper + (PCA/UMAP) + DBSCAN (distancia por correlación)

Pesos:

capital se reparte por igual entre “clusters grandes”

dentro de cada cluster grande, por igual entre subclusters

dentro de cada subcluster, por igual entre acciones
(o tu regla exacta weight_distribution, pero debe quedar descrita con claridad)

Rebalance: cada recalibrate_days

Hipótesis: la asignación por clúster reduce concentración implícita (acciones altamente correlacionadas) y mejora eficiencia riesgo/retorno frente a EW y MCW.

3) u pipeline (retornos → correlaciones/distancias → UMAP/DBSCAN/Mapper) escala mal con el número de activos:

La matriz de correlación/distancias es de tamaño 
𝑁
×
𝑁
N×N.

Con 200: 
200
2
=
40,000
200
2
=40,000 pares.

Con 500: 
500
2
=
250,000
500
2
=250,000 pares.

Eso es 6,25 veces más solo en pares, y en la práctica el tiempo/memoria también suben mucho.

Algoritmos como UMAP y DBSCAN, y el propio Mapper, suelen depender fuertemente de cálculos de vecindades/distancias: con 500 se vuelve mucho más pesado y más sensible a detalles numéricos.

Para un TFG, 200 es un compromiso muy habitual: suficientemente grande para que haya estructura (clusters), pero manejable para iterar, depurar y repetir backtests.

En vez de fijar 200 arbitrariamente, define el universo por cobertura de peso del ETF, por ejemplo:

“Selecciono el mínimo número de constituyentes que acumulan el 90% / 95% del peso del ETF”.
