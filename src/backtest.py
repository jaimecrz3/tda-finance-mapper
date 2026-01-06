from typing import Dict, Tuple
import numpy as np
import pandas as pd

# Importamos lo que construye clústeres (Mapper) y lo que convierte clústeres en pesos
from tda_mapper import MapperParams, build_clusters_from_prices, weight_distribution


# ============================================================
# 1) BACKTEST TDA: rebalance periódico + pesos por TDA
# ============================================================
def backtest_tda(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    params: MapperParams,
    tc_bps: float = 0.0
) -> pd.DataFrame: # Devolvemos un DataFrame
    """
    Backtest que simula una estrategia long-only:
      - Cada rebalance_days, recalcula clústeres usando los últimos lookback_days
      - Convierte clústeres en pesos (weight_distribution)
      - Mantiene esos pesos constantes hasta el siguiente rebalance
      - Calcula el retorno diario del portfolio como suma ponderada de retornos diarios
      - Aplica un coste proporcional al turnover (cambio de pesos) el primer día tras rebalance

    prices:
      DataFrame (Date x assets) con precios (Adj Close recomendado).
    lookback_days:
      Ventana histórica para construir clústeres (similar al History lookback en QC).
    rebalance_days:
      Frecuencia de rebalanceo (similar a recalibrate_days en QC).
    tc_bps:
      Coste de transacción en basis points (bps).
      Nota: El turnover indica qué porcentaje de la cartera se compra y vende durante un periodo 
      (día, mes, año). Cuanto mayor es el turnover: Más operaciones, más cambios de posiciones,
      mayores costes de transacción (comisiones, slippage), mayor impacto fiscal,
      Ej: Si en un rebalanceo la carera rota un 100%, es decir, un turnover=1,
      y tc_bps = 5 bps(1 bp = 0.01% = 0.0001) el coste que se aplica:
      coste = turnover*(tc_bps/10000) = 1*(5/10000) = 0.05%
      Es decir, se descuenta 0.05% del valor del portfolio ese día.
    """

    # 1) Ordenar fechas por si acaso
    prices = prices.sort_index()

    # 2) Retornos simples diarios 
    # pct_change calcula el cambio porcentual entre cada fila y la anterior. Formula:
    # r_t = P_t/P_(t-1) - 1
    # fillna(0): el primer día no tiene retorno; lo ponemos a 0 para no romper cálculos
    rets = prices.pct_change().fillna(0.0)

    dates = prices.index

    # 3) Validación mínima: necesitamos suficiente histórico
    # Si no hay al menos lookback_days + margen, no se puede hacer el primer rebalance correctamente
    if len(dates) < lookback_days + 10:
        raise ValueError("Serie demasiado corta para lookback_days.")

    # 4) Índices donde vamos a rebalancear
    # Empezamos en lookback_days porque antes no existe una ventana completa
    # y vamos saltando cada rebalance_days
    rebalance_idx = list(range(lookback_days, len(dates) - 1, rebalance_days))

    # 5) Series donde guardamos:
    # - port_ret: retorno diario de la estrategia
    # - turnover: turnover (aprox) solo el día que se aplica coste
    port_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)

    # 6) Pesos anteriores (para calcular turnover frente al nuevo peso)
    w_prev: Dict[str, float] = {}

    # ============================================================
    # Loop de rebalanceos
    # ============================================================
    for k, idx in enumerate(rebalance_idx):

        # 7) Ventana de precios usada para construir clústeres (lookback)
        # Incluye idx (día de rebalance) para usar hasta el día actual
        #
        # prices es un DataFrame de pandas:
        # filas -> tiempo (fechas)
        # columnas -> activos
        # .iloc[] selecciona filas por posición numérica (no por fecha).
        # Resultado: Una ventana con lookback_days + 1 filas
        window = prices.iloc[idx - lookback_days: idx + 1]

        # 8) Construir clústeres via Mapper (devuelve estructura anidada o None)
        clusters = build_clusters_from_prices(window, params)

        # 9) Convertir clústeres en pesos
        # Si falla el clustering, fallback: equal-weight en el panel disponible
        if not clusters:
            cols = list(window.columns)
            w = {c: 1.0 / len(cols) for c in cols}
        else:
            # weights sobre tickers 
            w = weight_distribution(clusters)

            # Seguridad: quedarnos solo con tickers realmente presentes en la ventana
            panel_set = set(window.columns)
            w = {s: float(v) for s, v in w.items() if s in panel_set and v > 0}

            # Normalización por si algo quedó mal (suma distinta a 1 o vacío)
            tot = sum(w.values())
            if tot <= 0:
                cols = list(window.columns)
                w = {c: 1.0 / len(cols) for c in cols}
            else:
                w = {s: v / tot for s, v in w.items()}

        # ============================================================
        # 10) Turnover aproximado
        # ============================================================
        # Turnover = 0.5 * sum(|w_new - w_old|) donde w_new son los 
        # nuevos pesos de cada ticker y w_old los antiguos
        # - se usa mucho como aproximación de “cuánto rebalancing haces”
        # - 0.5 hace que pasar de 100% A a 100% B sea turnover=1 (100%)
        # Interpretación del resultado to
        # to = 0.0 -> No hubo rebalance
        # to = 0.1 -> 10% de la cartera rotada
        # to = 1.0 -> 100% de rotación
        # to >1.0 -> Rotación múltiple (muy agresiva)
        keys = set(w_prev) | set(w) #union de activos de ambas carteras
        to = 0.5 * sum(
            abs(w.get(a, 0.0) - w_prev.get(a, 0.0)) 
            for a in keys
        )

        # ============================================================
        # 11) Definir tramo en el que estos pesos se aplican
        # ============================================================
        # Desde el día siguiente al rebalance (idx+1) hasta el siguiente rebalance.
        # Nota: esto replica la idea de “calculas pesos hoy, operas al próximo día/bar”.
        end_idx = rebalance_idx[k + 1] if k + 1 < len(rebalance_idx) else (len(dates) - 1)
        hold_dates = dates[idx + 1: end_idx + 1]

        # ============================================================
        # 12) Costes de transacción
        # ============================================================
        # tc_bps está en basis points (1 bp = 0.01% = 0.0001)
        # coste = turnover*(tc_bps/10000)
        # Se aplica solo el primer día del tramo como aproximación simple
        cost = to * (tc_bps / 10000.0) if tc_bps > 0 else 0.0

        # ============================================================
        # 13) Calcular retorno diario del portfolio durante el tramo
        # ============================================================
        for j, d in enumerate(hold_dates):
            # rets: DataFrame con retornos diarios de cada activo
            # seleccionamos la fila de retornos del dia d
            day_rets = rets.loc[d]

            # w.items() -> pesos actuales ({activo: peso})
            # day_rets.get(a, 0.0) -> retorno del activo a ese día (0 si no hay dato)
            # Multiplica peso × retorno y suma para todos los activos
            # Resultado -> retorno diario del portfolio
            r = sum(
                wgt * float(day_rets.get(a, 0.0)) 
                for a, wgt in w.items()
            )

            # Aplicar coste solo el primer día tras rebalance
            if j == 0 and cost > 0:
                r -= cost
                turnover.loc[d] = to

            # Guardamos el retorno del dia d
            port_ret.loc[d] = r

        # 14) Guardamos pesos actuales como “anteriores” para el próximo rebalance
        w_prev = dict(w)

    # 15) NAV (Net Asset Value): muestra cómo evoluciona el valor del portfolio a lo largo del tiempo
    nav = (1.0 + port_ret).cumprod()

    # 16) Devolvemos DataFrame con retorno, nav y turnover
    return pd.DataFrame({"port_ret": port_ret, "port_nav": nav, "turnover": turnover})


# ============================================================
# 2) BASELINE: equal-weight "constante" (sin rebalance explícito)
# ============================================================
def backtest_equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline simple:
      - Equal-weight fijo: cada día el retorno = media de retornos de todos los activos.
    Nota:
      - Esto no modela rebalanceo ni costes.
      - Es útil como baseline “idealizado”, pero para comparación justa con TDA
        es mejor usar backtest_equal_weight_rebalanced con mismos rebalance_days y tc_bps.
    """
    prices = prices.sort_index()
    rets = prices.pct_change().fillna(0.0)

    # Peso igual para todos los activos disponibles
    w = 1.0 / prices.shape[1]

    # retorno diario = promedio de retornos
    port_ret = rets.sum(axis=1) * w

    nav = (1.0 + port_ret).cumprod()
    return pd.DataFrame({"port_ret": port_ret, "port_nav": nav})


# ============================================================
# 3) MÉTRICAS: retorno anualizado, vol anualizada, Sharpe y max drawdown
# ============================================================
def perf_summary(port_ret: pd.Series, trading_days: int = 252) -> Dict[str, float]:
    """
    Calcula métricas típicas:
      - ann_return: retorno anualizado
      - ann_vol: volatilidad anualizada
      - sharpe: (ann_return / ann_vol) sin risk-free
      - max_drawdown: drawdown mínimo (negativo)

    trading_days=252: aproximación típica para días de mercado en un año.
    """
    r = port_ret.dropna()
    if len(r) == 0:
        return {}

    # Retorno anualizado geométrico:
    # (1+R_total)^(252/N) - 1
    ann_ret = (1.0 + r).prod() ** (trading_days / len(r)) - 1.0

    # Vol anualizada:
    # std diaria * sqrt(252)
    ann_vol = r.std(ddof=0) * np.sqrt(trading_days)

    # Sharpe simplificado (sin rf)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # NAV y drawdown
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    max_dd = dd.min()

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd)
    }


# ============================================================
# 4) BASELINE JUSTO: equal-weight + rebalance periódico + mismos costes
# ============================================================
def backtest_equal_weight_rebalanced(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    tc_bps: float = 0.0
) -> pd.DataFrame:
    """
    Equal-weight pero:
      - rebalancea cada rebalance_days (igual que TDA)
      - empieza en lookback_days (igual que TDA, para comparabilidad)
      - aplica costes con el mismo esquema (tc_bps * turnover)

    Esto es el baseline recomendado para comparar con TDA.
    """
    prices = prices.sort_index()
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index

    if len(dates) < lookback_days + 10:
        raise ValueError("Serie demasiado corta para lookback_days.")

    rebalance_idx = list(range(lookback_days, len(dates) - 1, rebalance_days))

    port_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)

    w_prev = {}

    for k, idx in enumerate(rebalance_idx):
        window = prices.iloc[idx - lookback_days: idx + 1]
        cols = list(window.columns)

        # Pesos equal-weight en el panel disponible
        w = {c: 1.0 / len(cols) for c in cols}

        # Turnover vs pesos previos
        keys = set(w_prev) | set(w)
        to = 0.5 * sum(abs(w.get(a, 0.0) - w_prev.get(a, 0.0)) for a in keys)

        # Tramo hasta el siguiente rebalance
        end_idx = rebalance_idx[k + 1] if k + 1 < len(rebalance_idx) else (len(dates) - 1)
        hold_dates = dates[idx + 1: end_idx + 1]

        cost = (tc_bps / 10000.0) * to if tc_bps > 0 else 0.0

        for j, d in enumerate(hold_dates):
            day_rets = rets.loc[d]
            r = sum(wgt * float(day_rets.get(a, 0.0)) for a, wgt in w.items())

            if j == 0 and cost > 0:
                r -= cost
                turnover.loc[d] = to

            port_ret.loc[d] = r

        w_prev = dict(w)

    nav = (1.0 + port_ret).cumprod()
    return pd.DataFrame({"port_ret": port_ret, "port_nav": nav, "turnover": turnover})
