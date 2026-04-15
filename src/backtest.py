from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd

# Mapper
from tda_mapper import MapperParams, build_clusters_from_prices, weight_distribution

# PH modules
from tdapersistence import PHParams, compute_persistence_diagrams_from_returns
from tdaphfeatures import ph_summary_features
from tdaregime import TopologicalAnomalyDetector, compute_landscape_norm


# BACKTEST TDA
def backtest_tda(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    params: MapperParams,
    tc_bps: float = 0.0,
    use_ph_control: bool = False,
    ph_params: Optional[PHParams] = None,
    ph_history_len: int = 12,  # Tamaño de la ventana histórica 
    ph_diagnostics_csv: Optional[str] = None,
) -> pd.DataFrame:
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
      Ventana histórica para construir clústeres 
    rebalance_days:
      Frecuencia de rebalanceo 
    tc_bps:
      Coste de transacción en basis points (bps).
      Nota: El turnover indica qué porcentaje de la cartera se compra y vende durante un periodo 
      (día, mes, año). Cuanto mayor es el turnover: Más operaciones, más cambios de posiciones,
      mayores costes de transacción (comisiones, slippage), mayor impacto fiscal,
      Ej: Si en un rebalanceo la carera rota un 100%, es decir, un turnover=1,
      y tc_bps = 5 bps(1 bp = 0.01% = 0.0001) el coste que se aplica:
      coste = turnover*(tc_bps/10000) = 1*(5/10000) = 0.05%
      Es decir, se descuenta 0.05% del valor del portfolio ese día.

    Extensión (opcional): PH regime control
      - Calcula PH en paralelo (sobre retornos) por ventana
      - Score_t = bottleneck(H1_{t-1}, H1_t)
      - Alpha_t = función online de score_t (sin look-ahead)
      - Mezcla final: w = alpha_t*w_tda + (1-alpha_t)*w_eq
    """

    # 1) Nos aseguramos que las fechas van estrictamente del pasado al futuro
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
    # Nota: Usar series de pandas nos permite indexar por fecha
    port_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)

    # Series para diagnosticar PH (se rellenan solo si use_ph_control=True)
    landscape_norm = pd.Series(np.nan, index=dates)
    market_safe_flag = pd.Series(1.0, index=dates) # 1.0 = Safe, 0.0 = Cash

    # 6) Pesos anteriores (para calcular turnover frente al nuevo peso)
    w_prev: Dict[str, float] = {}

    # Config PH por defecto
    if ph_params is None:
        ph_params = PHParams(
            maxdim=1,
            corr_method="pearson",
            dist_variant="sqrt",
            winsor_q=0.01
        )

    # Controlador de régimen 
    detector = TopologicalAnomalyDetector(
        history_len=ph_history_len,
        danger_quantile=0.95, 
        min_history=max(3, int(ph_history_len * 0.25)) 
    )

    diagnostics_rows = []


    # Loop de rebalanceos
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
        panel = list(window.columns) # tickers

        # 8) Construir clústeres via Mapper (devuelve estructura anidada o None)
        # Tomamos los n activos y construimos una nube de puntos formada por cada
        # uno de estos activos. Cada uno de estos puntos tiene una dimension w, donde
        # w es el tamaño de la ventana.
        # Esto nos permite clusterizar activos para asignar pesos. El output del clustering/Mapper 
        # se interpreta directamente: grupos de activos similares en ese periodo.
        # Posteriormente con weight_distribution_portfolio asignamos peso por cluster.
        #
        # Si quisiesemos detectar estados del mercado / regímenes, definiríamos w puntos y cada uno
        # de ellos de dimensión n, donde n es el número de activos considerados.
        clusters = build_clusters_from_prices(window, params)

        # 9) Convertir clústeres en pesos
        # Si falla el clustering, fallback: equal-weight en el panel disponible
        if not clusters:
            w_tda = {c: 1.0 / len(panel) for c in panel} if len(panel) > 0 else {}
        else:
            # weights sobre tickers
            #w_tda = weight_distribution(clusters)
            w_tda = weight_distribution(
                clusters,
                max_weight=0.1
            )

            # Seguridad: quedarnos solo con tickers realmente presentes en la ventana
            panel_set = set(panel)
            w_tda = {s: float(v) for s, v in w_tda.items() if s in panel_set and v > 0.0}

            # Normalización por si algo quedó mal (suma distinta a 1 o vacío)
            tot = sum(w_tda.values())
            if tot <= 0:
                w_tda = {c: 1.0 / len(panel) for c in panel} if len(panel) > 0 else {}
            else:
                w_tda = {s: v / tot for s, v in w_tda.items()}

        
        # PH control 
        norma_t = None
        market_safe = True

        if use_ph_control and len(panel) > 0:
            # para detectar crisis topológicas, dividimos el tamaño de la ventana (lookback_days) esa ventana entre 3, esto es tener un "Slow Signal" 
            # y un "Fast Signal".Queremos que Mapper sea lento y estable: necesitamos 5 años de datos para asegurarnos de que agrupar la tecnología con el consumo 
            # no sea una casualidad, sino un clúster estructural real.
            # En cambio la Homología Persistente queremos que sea mas rapida: si viene un "crash" financiero, no podemos esperar 5 años para reaccionar. 
            # Al recortar la ventana a 20 meses, el cálculo de Vietoris-Rips es muchísimo más sensible a lo que está pasando hoy en el mercado.
            ph_lookback_days = round(lookback_days/3)
            ph_window = prices.iloc[idx - ph_lookback_days : idx + 1]
            ph_returns_window = ph_window.pct_change().dropna(how="all")

            ph_out = compute_persistence_diagrams_from_returns(ph_returns_window, ph_params)
            dgms = ph_out.get("dgms", [])
            symbols_used = ph_out.get("symbols", [])

            # Calculamos la norma L2 y evaluamos seguridad
            if dgms and len(dgms) > 1:
                norma_t = compute_landscape_norm(dgms, dimension=1)
                market_safe = detector.is_market_safe(norma_t)

            feats = ph_summary_features(dgms) if dgms else {}
            # Con esto se hacer un gráfico de líneas en Python o Excel con la serie histórica del S&P 500 de fondo, y superponer encima la línea del
            # landscape_norm_L2. Ej de resultado: " El pico del L2 coincide exactamente con los meses previos a la caída de Lehman Brothers"
            diagnostics_rows.append({
                "rebalance_date": dates[idx],
                "n_assets_panel": len(panel),
                "n_assets_used_ph": len(symbols_used),
                "landscape_norm_L2": norma_t if norma_t is not None else np.nan,
                "market_safe": market_safe,
                **feats
            })

        # APLICACIÓN DE PESOS
        if use_ph_control and not market_safe:
            w = {} # Cash (Sin exposición a mercado)
            print(f"[{dates[idx].date()}] ALERTA TDA: Anomalía L2={norma_t:.2f}. Pasando a LIQUIDEZ (Cash).")
        else:
            # MERCADO SEGURO: Procedemos con Mapper
            w = w_tda

            # Con esto comprobamos lo siguiente:
            # Si la norma L1 (valor absoluto) es muy bajito, el modelo no se diferencia de repartir pesos equitativamente
            w_eq = {s: 1.0 / len(panel) for s in panel}
            l1 = sum(abs(w.get(s, 0.0) - w_eq[s]) for s in panel)
            wvals = np.array([w.get(s, 0.0) for s in panel]) if w else np.array([0.0])
            print(f"[{dates[idx].date()}] Seguro. L1_to_eq={l1:.6f}  max_weight={wvals.max():.4f}")

        
        # 10) Turnover aproximado
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
        to = 0.5 * sum(abs(w.get(a, 0.0) - w_prev.get(a, 0.0)) for a in keys)

        # 11) Definir tramo en el que estos pesos se aplican
        # Desde el día siguiente al rebalance (idx+1) hasta el siguiente rebalance.
        # Nota: esto replica la idea de “calculas pesos hoy, operas al próximo día/bar”.
        end_idx = rebalance_idx[k + 1] if k + 1 < len(rebalance_idx) else (len(dates) - 1)
        hold_dates = dates[idx + 1: end_idx + 1]

        # 12) Costes de transacción
        # tc_bps está en basis points (1 bp = 0.01% = 0.0001)
        # coste = turnover*(tc_bps/10000)
        # Se aplica solo el primer día del tramo como aproximación simple
        #
        # Hay veces que los algoritmos matemáticos compran y venden sin parar. Sin costes de transacción, 
        # el backtest puede parecer un éxito, pero en la vida real las comisiones lo arruinarían. Al incluirlo, validamos que
        # la señal topológica sea lo bastante fuerte como para ser rentable incluso pagando peajes al bróker.
        cost = to * (tc_bps / 10000.0) if tc_bps > 0 else 0.0

        # 13) Calcular retorno diario del portfolio durante el tramo
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

            # rellenar diagnóstico por día (constante en el tramo)
            if use_ph_control:
                landscape_norm.loc[d] = (norma_t if norma_t is not None else np.nan)
                market_safe_flag.loc[d] = 1.0 if market_safe else 0.0

        # 14) Guardamos pesos actuales como “anteriores” para el próximo rebalance
        w_prev = dict(w)

    # 15) NAV (Net Asset Value): muestra cómo evoluciona el valor del portfolio a lo largo del tiempo
    nav = (1.0 + port_ret).cumprod()

    # 16) Devolvemos DataFrame con retorno, nav y turnover
    out = pd.DataFrame({
        "port_ret": port_ret,
        "port_nav": nav,
        "turnover": turnover
    })

    if use_ph_control:
        out["landscape_norm_L2"] = landscape_norm
        out["market_safe_flag"] = market_safe_flag

    # Guardar CSV de diagnósticos por rebalance si se pide
    if ph_diagnostics_csv is not None and len(diagnostics_rows) > 0:
        pd.DataFrame(diagnostics_rows).to_csv(ph_diagnostics_csv, index=False)

    return out


# MÉTRICAS
def perf_summary(
    port_ret: pd.Series,
    periods_per_year: int = 12,
    rf: Optional[Union[float, pd.Series]] = 0.0, 
    mar: float = 0.0,
) -> Dict[str, float]:
    """
    Parámetros
    ----------
    port_ret : pd.Series
        Retornos del portfolio (en decimales, p.ej. 0.01 = +1%).
    periods_per_year : int
        Frecuencia de anualización (252 diario, 12 mensual, etc.).
    rf : float o pd.Series
        Risk-free en la misma frecuencia que port_ret (en decimales). Si es float, se asume constante.
    mar : float
        Minimum Acceptable Return para Sortino, en la MISMA frecuencia que port_ret
        (por defecto 0.0). Si quieres MAR anual, usa mar_annual / periods_per_year.
    """
    import numpy as _np
    import pandas as _pd

    r = port_ret.dropna().copy()
    if len(r) == 0:
        return {}

    # 1) Retornos: aritmético vs geométrico
    n = len(r)
    total_return = (1.0 + r).prod() - 1.0
    ann_return_geo = (1.0 + r).prod() ** (periods_per_year / n) - 1.0

    # 2) Riesgo
    vol_period = float(r.std(ddof=0))
    ann_vol = vol_period * _np.sqrt(periods_per_year)

    # 3) Excess returns
    if isinstance(rf, (int, float, _np.floating)):
        rf_series = _pd.Series(float(rf), index=r.index)
    else:
        rf_series = _pd.Series(rf).reindex(r.index).fillna(0.0)

    excess = (r - rf_series).dropna()

    # 4) Sharpe 
    ex_std = float(excess.std(ddof=0))
    sharpe = _np.sqrt(periods_per_year) * float(excess.mean()) / ex_std if ex_std > 0 else _np.nan

    # 5) Sortino
    mar_series = _pd.Series(float(mar), index=excess.index)
    downside = (excess - mar_series).copy()
    downside[downside > 0] = 0.0
    downside_dev = float(_np.sqrt((downside ** 2).mean()))
    sortino = _np.sqrt(periods_per_year) * float((excess - mar_series).mean()) / downside_dev if downside_dev > 0 else _np.nan

    # 6) Max Drawdown
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    max_dd = float(dd.min())

    return {
        "total_return": total_return,
        "ann_return_geo": ann_return_geo,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd
    }


# BASELINE
"""
Paper para justificar por qué uso esta estrategia como baseline es:
Autores: DeMiguel, V., Garlappi, L., & Uppal, R.
Año: 2009
Título: Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?
Revista: The Review of Financial Studies, 22(5), 1915-1953.

Demostró matemáticamente que los modelos hipercomplejos de optimización de carteras (que ganaron el Premio Nobel, como el de Markowitz) a menudo 
rinden peor en el mundo real que simplemente repartir el dinero a partes iguales (1/N). El motivo es que los modelos complejos acumulan 
muchísimo error de estimación al calcular matrices de covarianza, mientras que 1/N tiene un error de estimación de cero.
"""
def backtest_equal_weight_rebalanced(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int
) -> pd.DataFrame:
    """
    Baseline Equal-weight puro:
      - Rebalancea cada rebalance_days (igual que TDA).
      - Empieza en lookback_days (igual que TDA, garantizando comparabilidad temporal).
    """
    prices = prices.sort_index()
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index

    if len(dates) < lookback_days + 10:
        raise ValueError("Serie demasiado corta para lookback_days.")

    # Índices de rebalanceo
    rebalance_idx = list(range(lookback_days, len(dates) - 1, rebalance_days))

    port_ret = pd.Series(0.0, index=dates)

    for k, idx in enumerate(rebalance_idx):
        # 1. Definir el universo disponible en la ventana actual
        window = prices.iloc[idx - lookback_days: idx + 1]
        cols = list(window.columns)

        # 2. Asignar pesos 1/N
        w = {c: 1.0 / len(cols) for c in cols} if len(cols) > 0 else {}

        # 3. Definir el tramo de mantenimiento (Hold period)
        end_idx = rebalance_idx[k + 1] if k + 1 < len(rebalance_idx) else (len(dates) - 1)
        hold_dates = dates[idx + 1: end_idx + 1]

        # 4. Calcular el retorno diario durante el tramo
        for d in hold_dates:
            day_rets = rets.loc[d]
            r = sum(wgt * float(day_rets.get(a, 0.0)) for a, wgt in w.items())
            port_ret.loc[d] = r

    # Curva de capital final (Net Asset Value)
    nav = (1.0 + port_ret).cumprod()
    
    return pd.DataFrame({
        "port_ret": port_ret, 
        "port_nav": nav
    })
