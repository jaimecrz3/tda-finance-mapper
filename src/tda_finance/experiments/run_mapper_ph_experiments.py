"""Run final Mapper and Mapper+PH portfolio experiments.

This script compares three strategies on the selected dataset:

1. Mapper-based portfolio.
2. Mapper-based portfolio with persistent-homology regime control.
3. Equal-weight rebalanced benchmark.

The script is intended to be executed from the project root with:

    python -m tda_finance.experiments.run_mapper_ph_experiments
"""

from __future__ import annotations

import os
from typing import Dict, Literal, Tuple

import pandas as pd

from tda_finance.data_preprocessing.preprocess_kf49 import (
    load_kf49_prices_from_returns,
)
from tda_finance.data_preprocessing.preprocess_sp500_crsp import (
    load_sp500_prices_from_monthly_returns,
)
from tda_finance.portfolio.backtest_engine import (
    backtest_equal_weight_rebalanced,
    backtest_tda,
    perf_summary,
)
from tda_finance.tda.mapper_clustering import MapperParams


DatasetName = Literal["kf49", "sp500"]


def _get_metric(
    metrics: Dict[str, float],
    key: str,
    default: float = float("nan"),
) -> float:
    """Return a metric value or a default value if the key is missing."""
    return metrics.get(key, default)


def main() -> None:
    """Run the final experiment for the selected dataset."""
    # Select the dataset to evaluate. The same experimental protocol is used
    # for both universes.
    dataset: DatasetName = "sp500"

    # Common experimental configuration.
    periods_per_year = 12
    lookback_months = 60
    lookback_days = lookback_months
    rebalance_days = 3
    tc_bps = 5.0

    rf = pd.Series(dtype=float)

    # Load prices and risk-free rate. Dates are converted to month-end so that
    # all datasets follow the same monthly convention.
    if dataset == "kf49":
        rets = pd.read_csv(
            "data/49_industries_portfolios_monthly.csv",
            index_col=0,
            parse_dates=True,
        )
        rets.index = pd.to_datetime(rets.index).to_period("M").to_timestamp("M")

        prices = load_kf49_prices_from_returns(
            rets=rets,
            require_complete_panel=True,
        ).sort_index()

        ff3 = pd.read_csv(
            "data/ff3_monthly.csv",
            index_col=0,
            parse_dates=True,
        )
        ff3.index = pd.to_datetime(ff3.index).to_period("M").to_timestamp("M")

        rf = ff3["RF"]

        common = prices.index.intersection(rf.index)
        prices = prices.loc[common]
        rf = rf.loc[common]

        results_dir = "results_49_Industry_Portfolios"
        output_csv = (
            "results_49_Industry_Portfolios/"
            "results_kf49_mapper_vs_ph__vs_baseline.csv"
        )

    elif dataset == "sp500":
        prices = load_sp500_prices_from_monthly_returns(
            path="data/sp500_crsp_monthly_clean.parquet",
            file_format="parquet",
            start_price=100.0,
        ).sort_index()

        ff3 = pd.read_csv(
            "data/ff3_monthly.csv",
            index_col=0,
            parse_dates=True,
        )
        ff3.index = pd.to_datetime(ff3.index).to_period("M").to_timestamp("M")

        rf = ff3["RF"]

        common = prices.index.intersection(rf.index)
        prices = prices.loc[common]
        rf = rf.loc[common]

        results_dir = "results_SP500_CRSP"
        output_csv = "results_SP500_CRSP/results_sp500_mapper_vs_ph_vs_baseline.csv"

    else:
        raise ValueError("dataset must be either 'kf49' or 'sp500'.")

    os.makedirs(results_dir, exist_ok=True)

    # Final Mapper configuration selected from the previous experimental
    # comparisons. The cover parameters are dataset-specific because KF49 and
    # S&P 500 have different universe sizes and dependency structures.
    if dataset == "kf49":
        params = MapperParams(
            pca_var=0.80,
            umap_dim=1,
            n_cubes=12,
            perc_overlap=0.25,
            dbscan_eps=0.40,
            dbscan_min_samples=2,
            random_state=1,
            clusterer="haca",
            haca_distance_threshold=0.6,
            haca_linkage="average",
        )

    elif dataset == "sp500":
        params = MapperParams(
            pca_var=0.80,
            umap_dim=1,
            n_cubes=16,
            perc_overlap=0.15,
            dbscan_eps=0.40,
            dbscan_min_samples=2,
            random_state=1,
            clusterer="haca",
            haca_distance_threshold=0.6,
            haca_linkage="average",
            min_assets=30,
        )

    else:
        raise ValueError("dataset must be either 'kf49' or 'sp500'.")

    # Evaluation is performed by decades. Each period includes a warm-up window
    # before the evaluation start date, so the first portfolio can be built
    # causally using only past information.
    intervals: list[Tuple[str, str]] = [
        ("1975-01-31", "1984-12-31"),
        ("1985-01-31", "1994-12-31"),
        ("1995-01-31", "2004-12-31"),
        ("2005-01-31", "2014-12-31"),
        ("2015-01-31", "2025-11-30"),
    ]

    out_rows = []

    for start_text, end_text in intervals:
        start = pd.Timestamp(start_text).to_period("M").to_timestamp("M")
        end = pd.Timestamp(end_text).to_period("M").to_timestamp("M")

        # Keep warm-up data for model construction, but compute performance
        # metrics only on the actual evaluation period.
        warm_start = start - pd.offsets.MonthEnd(lookback_days + 1)
        sub = prices.loc[warm_start:end]

        # KF49 is treated as a complete-panel dataset. In S&P 500, assets can
        # enter or leave the available universe, so only fully empty columns are
        # removed at this stage.
        if dataset == "kf49":
            sub = sub.dropna(axis=1, how="any")

        elif dataset == "sp500":
            sub = sub.dropna(axis=1, how="all")

        if sub.shape[0] < lookback_days + 12:
            print(f"[SKIP] {dataset} {start_text}-{end_text}: too few months.")
            continue

        if sub.shape[1] < 10:
            print(f"[SKIP] {dataset} {start_text}-{end_text}: too few assets.")
            continue

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset}")
        print(f"Evaluation period: {start_text} -> {end_text}")
        print(f"Warm start: {warm_start.date()}")
        print(f"Subpanel shape: {sub.shape}")
        print("=" * 80)

        # Strategy 1: Mapper portfolio without PH regime control.
        tda_mapper_full = backtest_tda(
            sub,
            lookback_days,
            rebalance_days,
            params,
            tc_bps=tc_bps,
            use_ph_control=False,
        )

        diag_path = os.path.join(
            results_dir,
            f"ph_diagnostics_{dataset}_{start_text}_{end_text}.csv".replace(
                ":",
                "-",
            ),
        )

        # Strategy 2: Mapper portfolio with PH regime control. When the PH
        # signal flags an anomalous regime, the portfolio temporarily switches
        # to the equal-weight benchmark.
        tda_ph_full = backtest_tda(
            sub,
            lookback_days,
            rebalance_days,
            params,
            tc_bps=tc_bps,
            use_ph_control=True,
            ph_diagnostics_csv=diag_path,
            regime_action="equal_weight",
        )

        # Strategy 3: equal-weight benchmark, evaluated with the same rebalance
        # frequency and transaction-cost convention.
        eqw_reb_full = backtest_equal_weight_rebalanced(
            sub,
            lookback_days,
            rebalance_days,
            tc_bps=tc_bps,
        )

        # Remove the warm-up part before computing metrics and saving curves.
        tda_mapper = tda_mapper_full.loc[start:end]
        tda_ph = tda_ph_full.loc[start:end]
        eqw_reb = eqw_reb_full.loc[start:end]
        rf_win = rf.loc[start:end]

        tda = perf_summary(
            tda_mapper["port_ret"],
            periods_per_year=periods_per_year,
            rf=rf_win,
        )

        ph = perf_summary(
            tda_ph["port_ret"],
            periods_per_year=periods_per_year,
            rf=rf_win,
        )

        eqr = perf_summary(
            eqw_reb["port_ret"],
            periods_per_year=periods_per_year,
            rf=rf_win,
        )

        # Save NAV curves for later plotting and inspection.
        curves = pd.DataFrame(
            {
                "tda_mapper_ret": tda_mapper["port_ret"],
                "tda_mapper_nav": tda_mapper["port_nav"],
                "tda_ph_ret": tda_ph["port_ret"],
                "tda_ph_nav": tda_ph["port_nav"],
                "eqw_reb_ret": eqw_reb["port_ret"],
                "eqw_reb_nav": eqw_reb["port_nav"],
            }
        )

        curves_path = os.path.join(
            results_dir,
            f"nav_curves_{dataset}_{start_text}_{end_text}.csv".replace(
                ":",
                "-",
            ),
        )
        curves.to_csv(curves_path)

        out_rows.append(
            {
                "dataset": dataset,
                "start": start_text,
                "end": end_text,
                "n_months_total_with_warmup": int(sub.shape[0]),
                "n_assets_total_in_subpanel": int(sub.shape[1]),
                "tda_mapper_total_return": _get_metric(tda, "total_return"),
                "tda_mapper_ann_return_geo": _get_metric(tda, "ann_return_geo"),
                "tda_mapper_ann_vol": _get_metric(tda, "ann_vol"),
                "tda_mapper_sharpe": _get_metric(tda, "sharpe"),
                "tda_mapper_sortino": _get_metric(tda, "sortino"),
                "tda_mapper_max_dd": _get_metric(tda, "max_dd"),
                "tda_ph_total_return": _get_metric(ph, "total_return"),
                "tda_ph_ann_return_geo": _get_metric(ph, "ann_return_geo"),
                "tda_ph_ann_vol": _get_metric(ph, "ann_vol"),
                "tda_ph_sharpe": _get_metric(ph, "sharpe"),
                "tda_ph_sortino": _get_metric(ph, "sortino"),
                "tda_ph_max_dd": _get_metric(ph, "max_dd"),
                "eqw_reb_total_return": _get_metric(eqr, "total_return"),
                "eqw_reb_ann_return_geo": _get_metric(eqr, "ann_return_geo"),
                "eqw_reb_ann_vol": _get_metric(eqr, "ann_vol"),
                "eqw_reb_sharpe": _get_metric(eqr, "sharpe"),
                "eqw_reb_sortino": _get_metric(eqr, "sortino"),
                "eqw_reb_max_dd": _get_metric(eqr, "max_dd"),
                "final_nav_tda_mapper": float(tda_mapper["port_nav"].iloc[-1]),
                "final_nav_tda_ph": float(tda_ph["port_nav"].iloc[-1]),
                "final_nav_eqw_reb": float(eqw_reb["port_nav"].iloc[-1]),
            }
        )

    summary = pd.DataFrame(out_rows)
    summary.to_csv(output_csv, index=False)

    print("\nSummary saved to:")
    print(output_csv)
    print(summary)


if __name__ == "__main__":
    main()
