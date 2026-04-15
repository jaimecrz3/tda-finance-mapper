from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd

# Mapper
from tda_mapper import MapperParams, build_clusters_from_prices, weight_distribution

# PH modules
from tdapersistence import PHParams, compute_persistence_diagrams_from_returns
from tdaphfeatures import ph_summary_features
from tdaregime import TopologicalAnomalyDetector, compute_landscape_norm



def mix_with_equal_weight(
    w_tda: Dict[str, float],
    universe: List[str],
    alpha: float
) -> Dict[str, float]:
    """
    Mezcla convexa: alpha * w_tda + (1-alpha) * w_eq
    Garantiza pesos para todo el universo.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    n = len(universe)
    if n == 0:
        return {}

    w_eq = {s: 1.0 / n for s in universe}

    w = {}
    for s in universe:
        w[s] = alpha * w_tda.get(s, 0.0) + (1.0 - alpha) * w_eq[s]

    tot = sum(w.values())
    return {s: v / tot for s, v in w.items()} if tot > 0 else w_eq


# ============================================================
# 1) BACKTEST TDA: rebalance periódico + pesos por TDA
# ============================================================
def backtest_tda(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    params: MapperParams,
    tc_bps: float = 0.0,
    use_ph_control: bool = False,
    ph_params: Optional[PHParams] = None,
    # regime_controller: Optional[RegimeController] = None,
    ph_history_len: int = 12,  # Tamaño de la ventana histórica 
    ph_diagnostics_csv: Optional[str] = None,
    periods_per_year: int = 252,
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

    Extensión (opcional): PH regime control
      - Calcula PH en paralelo (sobre retornos) por ventana
      - Score_t = bottleneck(H1_{t-1}, H1_t)
      - Alpha_t = función online de score_t (sin look-ahead)
      - Mezcla final: w = alpha_t*w_tda + (1-alpha_t)*w_eq
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
        panel = list(window.columns) # tickers
        returns_window = window.pct_change().dropna(how="all") # retornos de window

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

        # ----------------------------
        # PH control (NUEVO ENFOQUE: LANDSCAPE + CASH)
        # ----------------------------
        norma_t = None
        market_safe = True

        if use_ph_control and len(panel) > 0:
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
            diagnostics_rows.append({
                "rebalance_date": dates[idx],
                "n_assets_panel": len(panel),
                "n_assets_used_ph": len(symbols_used),
                "landscape_norm_L2": norma_t if norma_t is not None else np.nan,
                "market_safe": market_safe,
                **feats
            })

        # --- APLICACIÓN DE PESOS ---
        if use_ph_control and not market_safe:
            # ALARMA: El mercado se ha roto topológicamente.
            w = {} # Cash (Sin exposición a mercado)
            print(f"[{dates[idx].date()}] ALERTA TDA: Anomalía L2={norma_t:.2f}. Pasando a LIQUIDEZ (Cash).")
        else:
            # MERCADO SEGURO: Procedemos con Mapper
            covered_by_tda = [s for s in panel if w_tda.get(s, 0.0) > 0.0]
            coverage = len(covered_by_tda) / max(1, len(panel))
            alpha_cov = float(np.clip(coverage, 0.0, 1.0))

            w = mix_with_equal_weight(w_tda, panel, alpha=alpha_cov)

            w_eq = {s: 1.0 / len(panel) for s in panel}
            l1 = sum(abs(w.get(s, 0.0) - w_eq[s]) for s in panel)
            wvals = np.array([w.get(s, 0.0) for s in panel]) if w else np.array([0.0])
            print(f"[{dates[idx].date()}] Seguro. L1_to_eq={l1:.6f}  max_weight={wvals.max():.4f}")

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
        to = 0.5 * sum(abs(w.get(a, 0.0) - w_prev.get(a, 0.0)) for a in keys)

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
# 3) MÉTRICAS
# ============================================================
def perf_summary(
    port_ret: pd.Series,
    periods_per_year: int = 252,
    rf: Optional[Union[float, pd.Series]] = 0.0, 
    mar: float = 0.0,
    var_levels: Tuple[float, ...] = (0.95, 0.99),
    market_ret: Optional[pd.Series] = None,
    factors: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Panel ampliado de métricas de performance.

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
    var_levels : tuple
        Niveles de confianza para VaR/ES, p.ej. (0.95, 0.99).
    market_ret : pd.Series, opcional
        Retorno del mercado (misma frecuencia) para CAPM alpha/beta.
        CAPM(Capital Asset Pricing Model) es un modelo matemático qu estima el retorno
        esperado de una inversión basado en su riesgo relativo al resto del mercado
    factors : pd.DataFrame, opcional
        DataFrame de factores (misma frecuencia). Formatos soportados:
          - Columnas tipo Ken French: ["Mkt-RF","SMB","HML","RMW","CMA"] y opcional "RF"
          - Si existe "RF" aquí y rf se deja a 0.0/None, se usará este RF.

        Importante: asegúrate de que los factores estén en decimales (no en %).
    """
    import numpy as _np
    import pandas as _pd

    r = port_ret.dropna().copy()
    if len(r) == 0:
        return {}

    # 1) Retornos: aritmético vs geométrico
    n = len(r)
    total_return = (1.0 + r).prod() - 1.0
    geo_mean_period = (1.0 + r).prod() ** (1.0 / n) - 1.0
    arith_mean_period = float(r.mean())

    ann_return_geo = (1.0 + r).prod() ** (periods_per_year / n) - 1.0
    ann_return_arith = arith_mean_period * periods_per_year

    # 2) Riesgo
    vol_period = float(r.std(ddof=0))
    ann_vol = vol_period * _np.sqrt(periods_per_year)

    # 3) Excess returns
    if rf is None:
        rf_series = _pd.Series(0.0, index=r.index)
    elif isinstance(rf, (int, float, _np.floating)):
        rf_series = _pd.Series(float(rf), index=r.index)
    else:
        rf_series = _pd.Series(rf).reindex(r.index)

    if factors is not None and "RF" in factors.columns:
        if rf is None or (isinstance(rf, (int, float, _np.floating)) and float(rf) == 0.0):
            rf_series = _pd.Series(factors["RF"]).reindex(r.index)

    excess = (r - rf_series).dropna()

    # 4) Sharpe (canónico)
    ex_std = float(excess.std(ddof=0))
    sharpe = _np.sqrt(periods_per_year) * float(excess.mean()) / ex_std if ex_std > 0 else _np.nan

    # 5) Sortino
    mar_series = _pd.Series(float(mar), index=excess.index)
    downside = (excess - mar_series).copy()
    downside[downside > 0] = 0.0
    downside_dev = float(_np.sqrt((downside ** 2).mean()))
    sortino = _np.sqrt(periods_per_year) * float((excess - mar_series).mean()) / downside_dev if downside_dev > 0 else _np.nan

    # 6) NAV, HWM, Drawdown, duración, Calmar
    nav = (1.0 + r).cumprod()
    hwm_nav = float(nav.cummax().max())
    hwm_return = float(hwm_nav - 1.0)

    dd = nav / nav.cummax() - 1.0
    max_dd = float(dd.min())  # negativo

    in_dd = nav < nav.cummax()
    max_dd_duration = 0
    cur = 0
    for v in in_dd.values:
        if v:
            cur += 1
            max_dd_duration = max(max_dd_duration, cur)
        else:
            cur = 0

    calmar = float(ann_return_geo / abs(max_dd)) if max_dd < 0 else _np.nan

    # 7) VaR/ES históricos
    var_es = {}
    for lvl in var_levels:
        a = 1.0 - float(lvl)
        q = float(r.quantile(a))
        es = float(r[r <= q].mean()) if (r <= q).any() else _np.nan
        var_es[f"VaR_{int(lvl*100)}"] = q
        var_es[f"ES_{int(lvl*100)}"] = es

    # 8) Momentos
    skew = float(r.skew()) if n > 2 else _np.nan
    kurt = float(r.kurtosis()) if n > 3 else _np.nan

    # 9) Regressions: CAPM y FF5
    capm_stats = {}
    ff5_stats = {}

    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        sm = None

    if sm is not None and market_ret is not None:
        mkt = _pd.Series(market_ret).reindex(r.index).dropna()
        idx = excess.index.intersection(mkt.index)
        y = excess.reindex(idx).dropna()
        x_mkt_ex = (mkt.reindex(idx) - rf_series.reindex(idx)).dropna()

        idx2 = y.index.intersection(x_mkt_ex.index)
        y = y.reindex(idx2)
        x_mkt_ex = x_mkt_ex.reindex(idx2)

        if len(y) >= 20 and x_mkt_ex.std(ddof=0) > 0:
            X = sm.add_constant(x_mkt_ex.values)
            model = sm.OLS(y.values, X).fit()
            alpha = float(model.params[0])
            beta = float(model.params[1])
            capm_stats = {
                "capm_alpha_period": alpha,
                "capm_alpha_ann": alpha * periods_per_year,
                "capm_alpha_tstat": float(model.tvalues[0]),
                "capm_beta": beta,
                "capm_beta_tstat": float(model.tvalues[1]),
                "capm_r2": float(model.rsquared),
                "capm_nobs": float(model.nobs),
            }

    if sm is not None and factors is not None:
        f = factors.copy()
        cols = []
        if "Mkt-RF" in f.columns: cols.append("Mkt-RF")
        if "SMB" in f.columns: cols.append("SMB")
        if "HML" in f.columns: cols.append("HML")
        if "RMW" in f.columns: cols.append("RMW")
        if "CMA" in f.columns: cols.append("CMA")

        if len(cols) >= 1:
            f = f[cols].reindex(r.index)
            idx = excess.index.intersection(f.dropna().index)
            y = excess.reindex(idx).dropna()
            Xdf = f.reindex(idx).dropna()

            idx2 = y.index.intersection(Xdf.index)
            y = y.reindex(idx2)
            Xdf = Xdf.reindex(idx2)

            if len(y) >= (len(cols) + 20):
                X = sm.add_constant(Xdf.values)
                model = sm.OLS(y.values, X).fit()
                alpha = float(model.params[0])
                ff5_stats = {
                    "ff_alpha_period": alpha,
                    "ff_alpha_ann": alpha * periods_per_year,
                    "ff_alpha_tstat": float(model.tvalues[0]),
                    "ff_r2": float(model.rsquared),
                    "ff_nobs": float(model.nobs),
                }
                for j, c in enumerate(cols, start=1):
                    ff5_stats[f"ff_beta_{c}"] = float(model.params[j])
                    ff5_stats[f"ff_tstat_{c}"] = float(model.tvalues[j])

    out = {
        "n_periods": float(n),
        "total_return": float(total_return),

        "geo_mean_period": float(geo_mean_period),
        "arith_mean_period": float(arith_mean_period),

        "ann_return_geo": float(ann_return_geo),
        "ann_return_arith": float(ann_return_arith),

        "ann_return": float(ann_return_geo),  # compat
        "ann_vol": float(ann_vol),

        "sharpe": float(sharpe),
        "sortino": float(sortino),

        "hwm_nav": float(hwm_nav),
        "hwm_return": float(hwm_return),

        "max_drawdown": float(max_dd),
        "max_drawdown_duration": float(max_dd_duration),

        "calmar": float(calmar),

        "skew": float(skew),
        "kurtosis": float(kurt),
    }
    out.update(var_es)
    out.update(capm_stats)
    out.update(ff5_stats)
    return out



# ============================================================
# 4) BASELINE JUSTO: equal-weight + rebalance periódico + mismos costes
# ============================================================
"""
Este baseline sigue sin ser correcto del todo, ¿Qué ocurre en el mundo real?

En la práctica, incluso si no realizas operaciones, los pesos de una
cartera cambian debido a los retornos de los activos.

Si en el día t tienes pesos w_t y al día siguiente los activos obtienen
retornos r_{t+1,i}, los pesos "drifteados" antes de reequilibrar son:

    w~_{t+1,i} = w_{t,i} (1 + r_{t+1,i})
                ---------------------------------
                sum_j w_{t,j} (1 + r_{t+1,j})

Estos pesos reflejan la evolución natural de la cartera sin trading.

--------------------------------------------------
Turnover correcto en un rebalanceo
--------------------------------------------------

Al reequilibrar, el turnover debe medirse como la distancia entre:

- Pesos actuales drifted: w~
- Pesos objetivo:         w*

Definición estándar:

    turnover = 0.5 * sum_i | w*_i - w~_i |

El factor 0.5 hace que pasar de 100% en A a 100% en B implique turnover = 1
(100% del capital rotado).

--------------------------------------------------
Implicación importante
--------------------------------------------------

Incluso una estrategia equal-weight tiene turnover > 0, aunque los pesos
objetivo w* sean siempre los mismos, porque los pesos reales w~ cambian
cada día debido a los retornos.

Por tanto, ignorar el drift de pesos subestima sistemáticamente el
turnover y los costes de transacción.
"""
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

        w = {c: 1.0 / len(cols) for c in cols} if len(cols) > 0 else {}

        keys = set(w_prev) | set(w)
        to = 0.5 * sum(abs(w.get(a, 0.0) - w_prev.get(a, 0.0)) for a in keys)

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
