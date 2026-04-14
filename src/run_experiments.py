import pandas as pd
import numpy as np

from tda_mapper import MapperParams
from backtest import (
    backtest_tda,
    backtest_equal_weight,
    perf_summary,
    backtest_equal_weight_rebalanced
)

from prepare_prices_49_IP_datareader import load_kf49_prices_from_returns


# Este archivo pone a prueba si agrupar activos finacieros usando TDA genera mejores retornos
# que estrategias de inversion mas simples
def main():
    use_data_nasdaq = False
    use_data_49_IP = True
    
    if use_data_nasdaq:
        prices = pd.read_parquet("data/prices_nasdaq100_adjclose.parquet").sort_index()
    elif use_data_49_IP:
        # El fichero de Ken French trae retornos (en %), no precios.
        # 1. parsear el bloque “Average Value Weighted Returns -- Monthly” (o el equal-weighted),
        # 2. limpiar missings (-99.99/-999),
        # 3. convertir retornos -> índice de precios sintético por capitalización compuesta.
        rets = pd.read_csv( # Lo convertimos a data frame
            "data/49_industries_portfolios_monthly.csv",
            index_col=0, # Indicamos que la columna que vamos a usar como indice es la 0, que es en la que vienen las fechas
            parse_dates=True # Parseamos index_col como un Datatimeindex, necesario para hacer time series analysis
        )

        # Asegura índice a fin de mes
        # pd.to_datetime(rets.index) toma el índice original: 197001 y lo traduce a un objeto de fecha de Pandas: 1970-01-01
        # .to_period("M") toma esa fecha exacta (1970-01-01) y lo convierte a un PeriodIndex Ahora la fecha no es un día concreto, sino que representa conceptualmente a "Todo enero de 1970"
        # .to_timestamp("M") vuelve a convertir en un DatatimeIndex, al pasarle la letra "M" (que en Pandas significa Month-end), empuja la fecha al último día del mes.
        # Resultado: El índice se convierte en 1970-01-31
        rets.index = pd.to_datetime(rets.index).to_period("M").to_timestamp("M")

        # A partir de los retornos generados, obtenemos los precios
        prices = load_kf49_prices_from_returns(
            rets=rets,
            require_complete_panel=True
        ).sort_index()

        # Factores Ken French (mensual): RF y Market Return
        # Este archivo contiene el Modelo de 3 Factores de Fama y French
        # Cada fila es un mes, y los valores están en formato decimal (por ejemplo, 0.05 es un 5% de retorno en ese mes). Las columnas son:
        # RF (Risk-Free Rate): Es el retorno que habrías obtenido si hubieras dejado tu dinero en el activo más seguro del mundo ese mes (usualmente Bonos del Tesoro de EE. UU. a un mes). 
        # Si tu estrategia de inversión no supera a esta columna, mejor dejas el dinero en el banco.
        # Mkt-RF (Market Risk Premium): Es el retorno total de toda la bolsa de valores junta, restándole la tasa libre de riesgo (RF). 
        # Responde a la pregunta: "¿Cuánto pagó la bolsa este mes por asumir el riesgo de invertir en acciones en lugar de dejar el dinero seguro?"
        # SMB (Small Minus Big): Mide el rendimiento de las empresas de pequeña capitalización (Small Caps) frente a las gigantes (Large Caps). Si el número es positivo, significa que a las
        # empresas pequeñas les fue mejor que a las grandes ese mes.
        # HML (High Minus Low): Mide el rendimiento de las empresas "Value" (compañia cuyo valor bursatil es menor que su valor intrinseco) frente a las empresas "Growth".
        # 
        # Este archivo nos sirve para calcular el Alpha y el Beta de tu estrategia. Se cruza la rentabilidad del portafolio con estos 3 factores para responder a:
        # ¿Ese 15% de ganancia se debe a que el algoritmo topológico es brillante (Alpha), o simplemente tuviste suerte porque durante esa década la bolsa en general subió muchísimo (Mkt-RF) y las empresas pequeñas estaban de moda (SMB)?
        # Sin este archivo solo se sabria el dinero ganado pero no el por qué.
        ff3 = pd.read_csv(
            "data/ff3_monthly.csv",
            index_col=0,
            parse_dates=True
        )

        # Normaliza índice a fin de mes
        ff3.index = pd.to_datetime(ff3.index).to_period("M").to_timestamp("M")

        rf = ff3["RF"]            # monthly risk-free (decimals)
        mkt = ff3["Mkt-RF"] + rf  # market TOTAL return (decimals)

        # Toma la lista de fechas de los precios (prices), la cruza con las fechas de la tasa libre de riesgo (rf) y 
        # la vuelve a cruzar con las fechas del mercado (mkt). El resultado (common) es una nueva lista maestra de fechas que 
        # solo incluye los meses que existen simultáneamente en los tres archivos.
        common = prices.index.intersection(rf.index).intersection(mkt.index)
        prices = prices.loc[common]
        rf = rf.loc[common]
        mkt = mkt.loc[common]
        ff3 = ff3.loc[common]
    else:
        raise ValueError("Se debe incluir un dataset")

    # Parámetros
    # lookback_days: Es el tamaño de la ventana histórica para construir clústeres.
    # Con datos diarios: lookback_days = 750 significa “750 días de trading”.
    # Con datos mensuales: lookback_days = 60 significa “60 meses”.
    #
    # rebalance_days: Es la frecuencia de rebalanceo en número de observaciones.
    # Con datos diarios: rebalance_days = 63 ≈ trimestral.
    # Con datos mensuales: rebalance_days = 3 = rebalance cada 3 meses (trimestral).
    if use_data_nasdaq:
        lookback_weeks = 150 # Unos tres años
        lookback_days = lookback_weeks * 5 # aprox 5 días de mercado/semana
        rebalance_days = 63 # trimestral
    elif use_data_49_IP:
        lookback_months = 60 # 5 años
        lookback_days = lookback_months # decisión práctica para reutilizar el mismo pipeline sin renombrar funcione
        rebalance_days = 3       # trimestral en frecuencia mensual (3 observaciones)
    else:
        raise ValueError("Se debe incluir un dataset")

    if use_data_nasdaq:
        params = MapperParams(
            pca_var=0.80,
            umap_dim=1,
            n_cubes=7,
            perc_overlap=0.40,
            dbscan_eps=0.60,
            dbscan_min_samples=4,
            random_state=1
        )
    elif use_data_49_IP:
        params = MapperParams(
            pca_var=0.80,
            umap_dim=1,
            n_cubes=12,
            perc_overlap=0.25,
            dbscan_eps=0.40,
            dbscan_min_samples=2,
            random_state=1,

            # activar HACA:
            clusterer="haca",
            # Es el umbral de distancia que determina hasta que punto dos clusters
            # pueden unirse. Si la distancia entre los dos clusters es mayor que
            # este umbral, no se unirán. Si es menor se consideraran suficiente
            # similares para fusionarse.
            # 
            # Un valor bajo da lugar a clusters más pequeños y específicos.
            # un valor alto da lugar a clusters más grandes y generales .
            haca_distance_threshold=0.6,
            # El método linkage define como se calcula la distancia entre dos clusters.
            # Con average, la distancia entre dos clusters es la media de todas las 
            # distancias entre cada punto de un cluster y cada punto del otro.
            haca_linkage="average",
        )
    else:
        raise ValueError("Se debe incluir un dataset")

    if use_data_nasdaq:
        intervals = [
            ("2013-01-01", "2016-12-31"),
            ("2016-01-01", "2019-12-31"),
            ("2020-01-01", "2021-12-31"),
            ("2013-01-01", "2021-12-31"),
        ]
    elif use_data_49_IP:
        intervals = [
            ("1975-01-31", "1984-12-31"), # Era de hiperinflación y subidas de tipos de interés extremas de Paul Volcker
            ("1985-01-31", "1994-12-31"),
            ("1995-01-31", "2004-12-31"), # la burbuja de las Punto Com (2000)
            ("2005-01-31", "2014-12-31"), # Gran Crisis Financiera de 2008
            ("2015-01-31", "2025-11-30"), # Covid-19
        ]
    else:
        raise ValueError("Se debe incluir un dataset")

    rows = []
    out_rows = []

    if use_data_nasdaq:
        periods_per_year = 252 
    elif use_data_49_IP:
        periods_per_year = 12
    else:
        raise ValueError("Se debe incluir un dataset")

    if use_data_nasdaq:
        for a, b in intervals:
            start = pd.Timestamp(a)
            end = pd.Timestamp(b)

            # 1) Warm-up: extendemos el panel hacia atrás para poder calcular clústeres
            # desde el primer día del tramo evaluado.
            warm_start = start - pd.tseries.offsets.BDay(lookback_days + 5)

            # 2) Subpanel con warm-up incluido
            sub = prices.loc[warm_start:end].dropna(axis=1, how="any")
            #sub = prices.loc[warm_start:end]

            # 3) Checks de tamaño mínimo
            if sub.shape[0] < lookback_days + 50 or sub.shape[1] < 10:
                continue

            # 4) Ejecutamos backtests (en sub que incluye warm-up)
            # ----------------------------
            # TDA sin PH
            # ----------------------------
            tda_mapper_full = backtest_tda(
                sub, lookback_days, rebalance_days, params, tc_bps=5.0,
                use_ph_control=False
            )

            # ----------------------------
            # TDA con PH control
            # ----------------------------
            diag_name = f"results_ph_diagnostics_{a}_{b}.csv".replace(":", "-")
            tda_ph_full = backtest_tda(
                sub, lookback_days, rebalance_days, params, tc_bps=5.0,
                use_ph_control=True,
                ph_diagnostics_csv=None
            )

            # Baselines
            eqw_full = backtest_equal_weight(sub)
            eqw_reb_full = backtest_equal_weight_rebalanced(sub, lookback_days, rebalance_days, tc_bps=5.0)

            # 5) Evaluación SOLO en el tramo [start, end] (quitamos warm-up de métricas)
            tda_mapper = tda_mapper_full.loc[start:end]
            tda_ph = tda_ph_full.loc[start:end]
            eqw = eqw_full.loc[start:end]
            eqw_reb = eqw_reb_full.loc[start:end]

            rows.append({
                "start": a, "end": b,
                "tda_mapper": perf_summary(tda_mapper["port_ret"], periods_per_year=periods_per_year, rf=0.0),
                "tda_ph": perf_summary(tda_ph["port_ret"], periods_per_year=periods_per_year, rf=0.0),
                "eqw": perf_summary(eqw["port_ret"], periods_per_year=periods_per_year, rf=0.0),
                "eqw_reb": perf_summary(eqw_reb["port_ret"], periods_per_year=periods_per_year, rf=0.0),
                "final_nav_tda_mapper": float(tda_mapper["port_nav"].iloc[-1]),
                "final_nav_tda_ph": float(tda_ph["port_nav"].iloc[-1]),
                "final_nav_eqw": float(eqw["port_nav"].iloc[-1]),
                "final_nav_eqw_reb": float(eqw_reb["port_nav"].iloc[-1]),
            })

            out_rows.append({
                "start": a, "end": b,

                "tda_mapper_ann_return": rows[-1]["tda_mapper"]["ann_return"],
                "tda_mapper_sharpe": rows[-1]["tda_mapper"]["sharpe"],
                "tda_mapper_max_dd": rows[-1]["tda_mapper"]["max_drawdown"],

                "tda_ph_ann_return": rows[-1]["tda_ph"]["ann_return"],
                "tda_ph_sharpe": rows[-1]["tda_ph"]["sharpe"],
                "tda_ph_max_dd": rows[-1]["tda_ph"]["max_drawdown"],

                "eqw_ann_return": rows[-1]["eqw"]["ann_return"],
                "eqw_sharpe": rows[-1]["eqw"]["sharpe"],
                "eqw_max_dd": rows[-1]["eqw"]["max_drawdown"],

                "eqw_reb_ann_return": rows[-1]["eqw_reb"]["ann_return"],
                "eqw_reb_sharpe": rows[-1]["eqw_reb"]["sharpe"],
                "eqw_reb_max_dd": rows[-1]["eqw_reb"]["max_drawdown"],
            })
    elif use_data_49_IP:
        for a, b in intervals:
            # Tomamos el texto de la lista de intervalos y lo convertimos en una fecha real que Python entiende.
            start = pd.Timestamp(a) 
            end = pd.Timestamp(b) 

            # Para saber en que invertir en una fecha, necesitamos informacion previa.
            # Aqui retrocedemos lookback_days + 1 desde la fecha de inicio.
            # Por ejemplo, si lookback_days = 60, retrocedemos 5 años mas un mes de seguridad
            warm_start = start - pd.offsets.MonthEnd(lookback_days + 1)

            # Tomamos la table de precios y nos quedamos con el periodo que va desde el inicio del calentamiento hasta el final de la década
            # Eliminamos todas las columnas (industrias) que tengan algun valor nulo
            sub = prices.loc[warm_start:end].dropna(axis=1, how="any")

            # sub.shape[0] es el número de filas (meses).
            # sub.shape[1] es el número de columnas (industrias/activos).
            # Si los meses disponibles son menores a los necesarios para el calentamiento MÁS un año de operativa real (12 meses) ignoramos este periodo
            # Si después de limpiar los activos defectuosos nos quedamos con menos de 10 industrias para invertir ignoramos este periodo.
            if sub.shape[0] < lookback_days + 12 or sub.shape[1] < 10:
                continue

            # SIMULAMOS LAS ESTRATEGIAS DE INVERSION (sobre el periodo completo, incluyendo el calentamiento)
            tda_mapper_full = backtest_tda(
                sub, lookback_days, rebalance_days, params, tc_bps=5.0,
                use_ph_control=False, periods_per_year=12
            )

            tda_ph_full = backtest_tda(
                sub, lookback_days, rebalance_days, params, tc_bps=5.0,
                use_ph_control=True,
                ph_diagnostics_csv=None, periods_per_year=12
            )

            eqw_full = backtest_equal_weight(sub)
            eqw_reb_full = backtest_equal_weight_rebalanced(
                sub, lookback_days, rebalance_days, tc_bps=5.0
            )
            
            # Tomamos los resultados de los backtests y quitamos el periodo de calentamiento
            tda_mapper = tda_mapper_full.loc[start:end]
            tda_ph = tda_ph_full.loc[start:end]
            eqw = eqw_full.loc[start:end]
            eqw_reb = eqw_reb_full.loc[start:end]

            # COMPROBACION (luego se quitara)
            # Resta mes a mes los rendimientos de la estrategia TDA Mapper ylos de Equal Weight Rebalanced
            # Si la diferencia máxima es 0.000000, significa que el modelo complejo de Topología está haciendo exactamente 
            # lo mismo que la estrategia tonta.
            diff = (tda_mapper["port_ret"] - eqw_reb["port_ret"]).abs()
            print("Max abs daily diff:", diff.max())
            print("Mean abs daily diff:", diff.mean())

            # quitamos el periodo de calentamiento de los factores (RF y mercado)
            rf_win = rf.loc[start:end]
            mkt_win = mkt.loc[start:end]
            ff3_win = ff3.loc[start:end]

            # Obtenems las metricas para cada algoritmo
            rows.append({
                "start": a, "end": b,
                "tda_mapper": perf_summary(tda_mapper["port_ret"], periods_per_year=periods_per_year, rf=rf_win, market_ret=mkt_win, factors=ff3_win),
                "tda_ph": perf_summary(tda_ph["port_ret"], periods_per_year=periods_per_year, rf=rf_win, market_ret=mkt_win, factors=ff3_win),
                "eqw": perf_summary(eqw["port_ret"], periods_per_year=periods_per_year, rf=rf_win, market_ret=mkt_win, factors=ff3_win),
                "eqw_reb": perf_summary(eqw_reb["port_ret"], periods_per_year=periods_per_year, rf=rf_win, market_ret=mkt_win, factors=ff3_win),
                "final_nav_tda_mapper": float(tda_mapper["port_nav"].iloc[-1]),
                "final_nav_tda_ph": float(tda_ph["port_nav"].iloc[-1]),
                "final_nav_eqw": float(eqw["port_nav"].iloc[-1]),
                "final_nav_eqw_reb": float(eqw_reb["port_nav"].iloc[-1]),
            })

            last = rows[-1] # Ultimo periodo analizado

            tda = last["tda_mapper"]
            ph  = last["tda_ph"]
            eqw = last["eqw"]
            eqr = last["eqw_reb"]

            # Funcion para extraer los resultados del último periodo analizado (rows[-1]) para luego pasarlos a las columnas del Excel final
            # Si algun dato no existe, lo pone como nan(not a number) 
            def _g(d: dict, k: str, default=np.nan):
                return d.get(k, default)

            out_rows.append({
                "start": a, "end": b,

                # --------- TDA Mapper ----------
                "tda_mapper_total_return": _g(tda, "total_return"),
                "tda_mapper_ann_return_geo": _g(tda, "ann_return_geo"),
                "tda_mapper_ann_return_arith": _g(tda, "ann_return_arith"),
                "tda_mapper_geo_mean_period": _g(tda, "geo_mean_period"),
                "tda_mapper_arith_mean_period": _g(tda, "arith_mean_period"),

                "tda_mapper_ann_vol": _g(tda, "ann_vol"),
                "tda_mapper_sharpe": _g(tda, "sharpe"),
                "tda_mapper_sortino": _g(tda, "sortino"),
                "tda_mapper_calmar": _g(tda, "calmar"),

                "tda_mapper_hwm_return": _g(tda, "hwm_return"),
                "tda_mapper_max_dd": _g(tda, "max_drawdown"),
                "tda_mapper_max_dd_duration": _g(tda, "max_drawdown_duration"),

                "tda_mapper_VaR_95": _g(tda, "VaR_95"),
                "tda_mapper_ES_95": _g(tda, "ES_95"),
                "tda_mapper_VaR_99": _g(tda, "VaR_99"),
                "tda_mapper_ES_99": _g(tda, "ES_99"),

                "tda_mapper_capm_alpha_ann": _g(tda, "capm_alpha_ann"),
                "tda_mapper_capm_alpha_tstat": _g(tda, "capm_alpha_tstat"),
                "tda_mapper_capm_beta": _g(tda, "capm_beta"),
                "tda_mapper_capm_r2": _g(tda, "capm_r2"),

                "tda_mapper_ff_alpha_ann": _g(tda, "ff_alpha_ann"),
                "tda_mapper_ff_alpha_tstat": _g(tda, "ff_alpha_tstat"),
                "tda_mapper_ff_r2": _g(tda, "ff_r2"),

                # Con FF3 tendrás estas tres si factors contiene Mkt-RF/SMB/HML:
                "tda_mapper_ff_beta_MktRF": _g(tda, "ff_beta_Mkt-RF"),
                "tda_mapper_ff_tstat_MktRF": _g(tda, "ff_tstat_Mkt-RF"),
                "tda_mapper_ff_beta_SMB": _g(tda, "ff_beta_SMB"),
                "tda_mapper_ff_tstat_SMB": _g(tda, "ff_tstat_SMB"),
                "tda_mapper_ff_beta_HML": _g(tda, "ff_beta_HML"),
                "tda_mapper_ff_tstat_HML": _g(tda, "ff_tstat_HML"),

                # --------- TDA + PH ----------
                "tda_ph_total_return": _g(ph, "total_return"),
                "tda_ph_ann_return_geo": _g(ph, "ann_return_geo"),
                "tda_ph_ann_return_arith": _g(ph, "ann_return_arith"),
                "tda_ph_ann_vol": _g(ph, "ann_vol"),
                "tda_ph_sharpe": _g(ph, "sharpe"),
                "tda_ph_sortino": _g(ph, "sortino"),
                "tda_ph_calmar": _g(ph, "calmar"),
                "tda_ph_hwm_return": _g(ph, "hwm_return"),
                "tda_ph_max_dd": _g(ph, "max_drawdown"),
                "tda_ph_max_dd_duration": _g(ph, "max_drawdown_duration"),
                "tda_ph_VaR_95": _g(ph, "VaR_95"),
                "tda_ph_ES_95": _g(ph, "ES_95"),
                "tda_ph_VaR_99": _g(ph, "VaR_99"),
                "tda_ph_ES_99": _g(ph, "ES_99"),
                "tda_ph_capm_alpha_ann": _g(ph, "capm_alpha_ann"),
                "tda_ph_capm_alpha_tstat": _g(ph, "capm_alpha_tstat"),
                "tda_ph_capm_beta": _g(ph, "capm_beta"),
                "tda_ph_capm_r2": _g(ph, "capm_r2"),
                "tda_ph_ff_alpha_ann": _g(ph, "ff_alpha_ann"),
                "tda_ph_ff_alpha_tstat": _g(ph, "ff_alpha_tstat"),
                "tda_ph_ff_r2": _g(ph, "ff_r2"),

                # --------- EQW ----------
                "eqw_total_return": _g(eqw, "total_return"),
                "eqw_ann_return_geo": _g(eqw, "ann_return_geo"),
                "eqw_ann_return_arith": _g(eqw, "ann_return_arith"),
                "eqw_ann_vol": _g(eqw, "ann_vol"),
                "eqw_sharpe": _g(eqw, "sharpe"),
                "eqw_sortino": _g(eqw, "sortino"),
                "eqw_calmar": _g(eqw, "calmar"),
                "eqw_hwm_return": _g(eqw, "hwm_return"),
                "eqw_max_dd": _g(eqw, "max_drawdown"),
                "eqw_max_dd_duration": _g(eqw, "max_drawdown_duration"),
                "eqw_VaR_95": _g(eqw, "VaR_95"),
                "eqw_ES_95": _g(eqw, "ES_95"),
                "eqw_VaR_99": _g(eqw, "VaR_99"),
                "eqw_ES_99": _g(eqw, "ES_99"),

                # --------- EQW Rebalanced ----------
                "eqw_reb_total_return": _g(eqr, "total_return"),
                "eqw_reb_ann_return_geo": _g(eqr, "ann_return_geo"),
                "eqw_reb_ann_return_arith": _g(eqr, "ann_return_arith"),
                "eqw_reb_ann_vol": _g(eqr, "ann_vol"),
                "eqw_reb_sharpe": _g(eqr, "sharpe"),
                "eqw_reb_sortino": _g(eqr, "sortino"),
                "eqw_reb_calmar": _g(eqr, "calmar"),
                "eqw_reb_hwm_return": _g(eqr, "hwm_return"),
                "eqw_reb_max_dd": _g(eqr, "max_drawdown"),
                "eqw_reb_max_dd_duration": _g(eqr, "max_drawdown_duration"),
                "eqw_reb_VaR_95": _g(eqr, "VaR_95"),
                "eqw_reb_ES_95": _g(eqr, "ES_95"),
                "eqw_reb_VaR_99": _g(eqr, "VaR_99"),
                "eqw_reb_ES_99": _g(eqr, "ES_99"),

                # --------- NAV finales (ya los guardabas) ----------
                "final_nav_tda_mapper": float(last["final_nav_tda_mapper"]),
                "final_nav_tda_ph": float(last["final_nav_tda_ph"]),
                "final_nav_eqw": float(last["final_nav_eqw"]),
                "final_nav_eqw_reb": float(last["final_nav_eqw_reb"]),
            })

    else:
        raise ValueError("Se debe incluir un dataset")

    summary = pd.DataFrame(out_rows)
    if use_data_nasdaq:
        summary.to_csv("results_Nasdaq100/prueba_results_summary_mapper_vs_ph_control.csv", index=False)
    elif use_data_49_IP:
        summary.to_csv(
            "results_49_Industry_Portfolios/results_haca_landscape_summary_mapper_vs_ph_control.csv", index=False)
    else:
        raise ValueError("Se debe incluir un dataset")
    
    print(summary)

#__name__ es una variable especial que se establece en "__main__" únicamente cuando 
# el intérprete ejecuta el script directamente; en caso contrario, toma el nombre del módulo.
#Cómo funciona
#
#Cuando ejecutas: python your_script.py
# Python asigna a __name__ el valor "__main__" dentro de your_script.py, por lo que el 
# código dentro del bloque if se ejecuta.
#
#Importado como módulo, Cuando otro script hace: import your_script
# Python asigna a __name__ el valor "your_script" dentro de ese archivo, por lo que el bloque if se omite.
if __name__ == "__main__":
    main()