import pandas as pd

from tda_mapper import MapperParams
from backtest import backtest_tda, backtest_equal_weight, perf_summary, backtest_equal_weight_rebalanced


def main():
    prices = pd.read_parquet("data/prices_adjclose.parquet").sort_index()

    # Parámetros 
    lookback_weeks = 150
    lookback_days = lookback_weeks * 5  # aprox 5 días de mercado/semana
    rebalance_days = 125

    params = MapperParams(
        pca_var=0.80,
        umap_dim=1,
        n_cubes=10,
        perc_overlap=0.2,
        dbscan_eps=0.3,
        dbscan_min_samples=2,
        random_state=1
    )

    intervals = [
        ("2013-01-01", "2016-12-31"),
        ("2016-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2013-01-01", "2021-12-31"),
    ]

    rows = []
    for a, b in intervals:
        start = pd.Timestamp(a)
        end = pd.Timestamp(b)

        # 1) Warm-up: extendemos el panel hacia atrás para poder calcular clústeres
        # desde el primer día del tramo evaluado.
        warm_start = start - pd.tseries.offsets.BDay(lookback_days + 5)

        # 2) Subpanel con warm-up incluido
        sub = prices.loc[warm_start:end].dropna(axis=1, how="any")

        # 3) Checks de tamaño mínimo
        if sub.shape[0] < lookback_days + 50 or sub.shape[1] < 10:
            continue

        # 4) Ejecutamos backtests (en sub que incluye warm-up)
        tda_full = backtest_tda(sub, lookback_days, rebalance_days, params, tc_bps=5.0)

        # Baseline A: equal-weight buy&hold (sin rebalance) - opcional mantenerlo
        eqw_full = backtest_equal_weight(sub)

        # Baseline B (recomendado): equal-weight con rebalance cada 125 días y mismos costes
        eqw_reb_full = backtest_equal_weight_rebalanced(
            sub, lookback_days, rebalance_days, tc_bps=5.0
        )

        # 5) Evaluación SOLO en el tramo [start, end] (quitamos warm-up de métricas)
        tda = tda_full.loc[start:end]
        eqw = eqw_full.loc[start:end]
        eqw_reb = eqw_reb_full.loc[start:end]

        rows.append({
            "start": a, "end": b,
            "tda": perf_summary(tda["port_ret"]),
            "eqw": perf_summary(eqw["port_ret"]),
            "eqw_reb": perf_summary(eqw_reb["port_ret"]),
            "final_nav_tda": float(tda["port_nav"].iloc[-1]),
            "final_nav_eqw": float(eqw["port_nav"].iloc[-1]),
            "final_nav_eqw_reb": float(eqw_reb["port_nav"].iloc[-1]),
        })

        # 6) Guardamos outputs completos (incluyen warm-up) y también evaluados
        #tda_full.to_csv(f"results_tda_full_{a}_{b}.csv", index=True)
        #eqw_full.to_csv(f"results_eqw_full_{a}_{b}.csv", index=True)
        #eqw_reb_full.to_csv(f"results_eqw_reb_full_{a}_{b}.csv", index=True)

        #tda.to_csv(f"results_tda_eval_{a}_{b}.csv", index=True)
        #eqw.to_csv(f"results_eqw_eval_{a}_{b}.csv", index=True)
        #eqw_reb.to_csv(f"results_eqw_reb_eval_{a}_{b}.csv", index=True)

    # Resumen plano
    out = []
    for r in rows:
        out.append({
            "start": r["start"], "end": r["end"],
            "tda_ann_return": r["tda"]["ann_return"],
            "tda_sharpe": r["tda"]["sharpe"],
            "tda_max_dd": r["tda"]["max_drawdown"],

            "eqw_ann_return": r["eqw"]["ann_return"],
            "eqw_sharpe": r["eqw"]["sharpe"],
            "eqw_max_dd": r["eqw"]["max_drawdown"],

            "eqw_reb_ann_return": r["eqw_reb"]["ann_return"],
            "eqw_reb_sharpe": r["eqw_reb"]["sharpe"],
            "eqw_reb_max_dd": r["eqw_reb"]["max_drawdown"],
        })

    summary = pd.DataFrame(out)
    summary.to_csv("results_summary.csv", index=False)
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
