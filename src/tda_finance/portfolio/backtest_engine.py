"""Backtesting utilities for Mapper-based portfolio strategies.

This module contains the functions used to simulate long-only portfolio
strategies, compute performance metrics and compare Mapper-based portfolios
against an equal-weight benchmark.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from tda_finance.tda.mapper_clustering import (
    MapperParams,
    build_clusters_from_prices,
    weight_distribution,
)
from tda_finance.tda.persistence_diagrams import (
    PHParams,
    compute_persistence_diagrams_from_returns,
)
from tda_finance.tda.persistence_features import ph_summary_features
from tda_finance.tda.regime_detection import (
    TopologicalAnomalyDetector,
    compute_landscape_norm,
)


WeightMethod = Literal["node_overlap", "macro_cluster"]
RegimeAction = Literal["cash", "equal_weight"]


def backtest_tda(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    params: MapperParams,
    tc_bps: float = 0.0,
    use_ph_control: bool = False,
    ph_params: Optional[PHParams] = None,
    ph_history_len: int = 12,
    ph_diagnostics_csv: Optional[str] = None,
    weight_method: WeightMethod = "node_overlap",
    regime_action: RegimeAction = "cash",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run a causal long-only backtest using Mapper-based weights.

    At each rebalance date, the function builds a Mapper representation from a
    historical price window, converts the resulting clusters into portfolio
    weights and applies those weights until the next rebalance. Optionally, a
    persistent-homology signal can be used as a regime-control mechanism.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price matrix indexed by date, with one column per asset.
    lookback_days : int
        Number of past observations used to build each Mapper graph.
    rebalance_days : int
        Number of observations between consecutive rebalances.
    params : MapperParams
        Parameters used in the Mapper construction.
    tc_bps : float, default=0.0
        Transaction cost in basis points, applied proportionally to turnover.
    use_ph_control : bool, default=False
        Whether to activate the persistent-homology regime control.
    ph_params : PHParams, optional
        Parameters used to compute persistence diagrams. If None, default
        parameters are used.
    ph_history_len : int, default=12
        Number of past PH scores used by the anomaly detector.
    ph_diagnostics_csv : str, optional
        Path where PH diagnostics are saved. If None, diagnostics are not
        written to disk.
    weight_method : {"node_overlap", "macro_cluster"}, default="node_overlap"
        Rule used to transform Mapper clusters into portfolio weights.
    regime_action : {"cash", "equal_weight"}, default="cash"
        Action applied when PH control detects an anomalous regime.
    verbose : bool, default=False
        Whether to print rebalance diagnostics.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with portfolio returns, NAV and turnover.
        If PH control is active, additional diagnostic columns are included.

    Raises
    ------
    ValueError
        If the input time series is too short or if regime_action is not
        supported.

    Notes
    -----
    The backtest is causal: weights computed at a rebalance date are applied
    only to subsequent observations.
    """
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)
    dates = prices.index

    if len(dates) < lookback_days + 10:
        raise ValueError("Time series is too short for the selected lookback.")

    # Rebalance dates start after the lookback window. The last available date
    # is not used as a rebalance date because weights are applied afterwards.
    rebalance_idx = list(range(lookback_days, len(dates) - 1, rebalance_days))

    port_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)

    landscape_norm = pd.Series(np.nan, index=dates)
    market_safe_flag = pd.Series(1.0, index=dates)

    previous_weights: Dict[str, float] = {}

    if ph_params is None:
        ph_params = PHParams(
            maxdim=1,
            corr_method="pearson",
            dist_variant="sqrt",
            winsor_q=0.01,
        )

    detector = TopologicalAnomalyDetector(
        history_len=ph_history_len,
        danger_quantile=0.95,
        min_history=max(3, int(ph_history_len * 0.25)),
    )

    diagnostics_rows = []

    for k, idx in enumerate(rebalance_idx):
        # The window includes only information available at the rebalance date.
        window_raw = prices.iloc[idx - lookback_days: idx + 1]

        # Forward filling is allowed inside the historical window. Backward
        # filling is avoided to prevent look-ahead bias.
        window = window_raw.sort_index().ffill().dropna(axis=1, how="any")
        panel = list(window.columns)

        if len(panel) == 0:
            continue

        clusters = build_clusters_from_prices(window, params)

        # If Mapper cannot produce a valid graph, the strategy falls back to an
        # equal-weight portfolio over the available panel.
        if not clusters:
            mapper_weights = _equal_weight(panel)
        else:
            mapper_weights = weight_distribution(
                clusters,
                max_weight=0.1,
                method=weight_method,
            )
            mapper_weights = _filter_and_normalize_weights(
                mapper_weights,
                valid_assets=panel,
            )

            if not mapper_weights:
                mapper_weights = _equal_weight(panel)

        norm_t: Optional[float] = None
        market_safe = True

        # PH control uses a shorter recent window. Mapper builds the portfolio,
        # while PH only decides whether the current regime looks anomalous.
        if use_ph_control and len(panel) > 0:
            ph_lookback_days = round(lookback_days / 3)
            ph_window_raw = prices.iloc[idx - ph_lookback_days: idx + 1]
            ph_window = ph_window_raw.sort_index().ffill().dropna(axis=1, how="any")
            ph_returns_window = ph_window.pct_change(fill_method=None).dropna(how="all")

            ph_out = compute_persistence_diagrams_from_returns(
                ph_returns_window,
                ph_params,
            )
            dgms = ph_out.get("dgms", [])
            symbols_used = ph_out.get("symbols", [])

            # The H1 landscape norm is used as a scalar topological regime
            # indicator. Large values relative to recent history flag anomalies.
            if dgms and len(dgms) > 1:
                norm_t = compute_landscape_norm(dgms, dimension=1)
                market_safe = detector.is_market_safe(norm_t)

            features = ph_summary_features(dgms) if dgms else {}

            diagnostics_rows.append(
                {
                    "rebalance_date": dates[idx],
                    "n_assets_panel": len(panel),
                    "n_assets_used_ph": len(symbols_used),
                    "landscape_norm_L2": norm_t if norm_t is not None else np.nan,
                    "market_safe": market_safe,
                    **features,
                }
            )

        # If PH flags an anomalous regime, replace the Mapper weights according
        # to the selected defensive action.
        if use_ph_control and not market_safe:
            if regime_action == "cash":
                weights: Dict[str, float] = {}

            elif regime_action == "equal_weight":
                weights = _equal_weight(panel)

            else:
                raise ValueError(
                    "regime_action must be either 'cash' or 'equal_weight'."
                )

            if verbose:
                print(
                    f"[{dates[idx].date()}] PH alert. "
                    f"norm={norm_t}. Action={regime_action}."
                )

        else:
            weights = mapper_weights

            if verbose:
                eq_weights = _equal_weight(panel)
                l1_distance = sum(
                    abs(weights.get(asset, 0.0) - eq_weights[asset])
                    for asset in panel
                )
                max_weight = max(weights.values()) if weights else 0.0
                print(
                    f"[{dates[idx].date()}] Safe regime. "
                    f"L1_to_equal_weight={l1_distance:.6f}, "
                    f"max_weight={max_weight:.4f}."
                )

        turnover_value = _turnover(previous_weights, weights)

        end_idx = (
            rebalance_idx[k + 1]
            if k + 1 < len(rebalance_idx)
            else len(dates) - 1
        )

        # Weights computed at the rebalance date are applied only from the next
        # observation onward, preserving causality.
        hold_dates = dates[idx + 1: end_idx + 1]

        # Transaction costs are charged once, at the beginning of the holding
        # period, in proportion to portfolio turnover.
        cost = turnover_value * (tc_bps / 10000.0) if tc_bps > 0 else 0.0

        for j, date in enumerate(hold_dates):
            day_rets = rets.loc[date]
            daily_return = _portfolio_return(day_rets, weights)

            if j == 0 and cost > 0:
                daily_return -= cost
                turnover.loc[date] = turnover_value

            port_ret.loc[date] = daily_return

            if use_ph_control:
                landscape_norm.loc[date] = norm_t if norm_t is not None else np.nan
                market_safe_flag.loc[date] = 1.0 if market_safe else 0.0

        previous_weights = dict(weights)

    nav = (1.0 + port_ret).cumprod()

    out = pd.DataFrame(
        {
            "port_ret": port_ret,
            "port_nav": nav,
            "turnover": turnover,
        }
    )

    if use_ph_control:
        out["landscape_norm_L2"] = landscape_norm
        out["market_safe_flag"] = market_safe_flag

    if ph_diagnostics_csv is not None and diagnostics_rows:
        pd.DataFrame(diagnostics_rows).to_csv(ph_diagnostics_csv, index=False)

    return out


def perf_summary(
    port_ret: pd.Series,
    periods_per_year: int = 12,
    rf: Optional[Union[float, pd.Series]] = 0.0,
    mar: float = 0.0,
) -> Dict[str, float]:
    """Compute performance and risk metrics for a return series.

    Parameters
    ----------
    port_ret : pandas.Series
        Portfolio returns in decimal form.
    periods_per_year : int, default=12
        Number of return observations per year.
    rf : float or pandas.Series, optional
        Risk-free rate in the same frequency as port_ret. If a scalar is
        provided, it is treated as constant.
    mar : float, default=0.0
        Minimum acceptable return used in the Sortino ratio, in the same
        frequency as port_ret.

    Returns
    -------
    dict of str to float
        Dictionary containing total return, annualized return, annualized
        volatility, Sharpe ratio, Sortino ratio and maximum drawdown.
    """
    returns = pd.to_numeric(port_ret, errors="coerce").dropna().astype(float)

    if len(returns) == 0:
        return {}

    n_obs = len(returns)
    returns_array = returns.to_numpy(dtype=float)

    growth = float(np.prod(1.0 + returns_array))
    total_return = growth - 1.0
    ann_return_geo = growth ** (periods_per_year / n_obs) - 1.0

    vol_period = float(np.std(returns_array, ddof=0))
    ann_vol = float(vol_period * np.sqrt(periods_per_year))

    if isinstance(rf, (int, float, np.floating)):
        rf_series = pd.Series(float(rf), index=returns.index)
    else:
        rf_series = pd.Series(rf).reindex(returns.index).fillna(0.0)

    excess = (returns - rf_series).dropna()

    excess_std = float(excess.std(ddof=0))
    sharpe = (
        float(np.sqrt(periods_per_year) * excess.mean() / excess_std)
        if excess_std > 0
        else float(np.nan)
    )

    mar_series = pd.Series(float(mar), index=excess.index)
    downside = (excess - mar_series).copy()
    downside[downside > 0] = 0.0
    downside_dev = float(np.sqrt((downside**2).mean()))

    sortino = (
        float(np.sqrt(periods_per_year) * (excess - mar_series).mean() / downside_dev)
        if downside_dev > 0
        else float(np.nan)
    )

    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    max_dd = float(drawdown.min())

    return {
        "total_return": total_return,
        "ann_return_geo": ann_return_geo,
        "ann_vol": float(ann_vol),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
    }


def _equal_weight(assets: list[str]) -> Dict[str, float]:
    """Return equal weights for a list of assets."""
    if len(assets) == 0:
        return {}

    weight = 1.0 / len(assets)
    return {asset: weight for asset in assets}


def _filter_and_normalize_weights(
    weights: Dict[str, float],
    valid_assets: list[str],
) -> Dict[str, float]:
    """Keep valid positive weights and normalize them to sum to one."""
    valid_set = set(valid_assets)
    filtered = {
        asset: float(weight)
        for asset, weight in weights.items()
        if asset in valid_set and weight > 0.0
    }

    total = sum(filtered.values())

    if total <= 0:
        return {}

    return {asset: weight / total for asset, weight in filtered.items()}


def _turnover(
    previous_weights: Dict[str, float],
    new_weights: Dict[str, float],
) -> float:
    """Compute one-way portfolio turnover between two weight vectors."""
    assets = set(previous_weights) | set(new_weights)

    return 0.5 * sum(
        abs(new_weights.get(asset, 0.0) - previous_weights.get(asset, 0.0))
        for asset in assets
    )


def _portfolio_return(
    asset_returns: pd.Series,
    weights: Dict[str, float],
) -> float:
    """Compute the weighted portfolio return for one date."""
    portfolio_return = 0.0

    for asset, weight in weights.items():
        asset_return = asset_returns.get(asset, np.nan)

        if pd.notna(asset_return):
            portfolio_return += weight * float(asset_return)

    return portfolio_return


def backtest_equal_weight_rebalanced(
    prices: pd.DataFrame,
    lookback_days: int,
    rebalance_days: int,
    tc_bps: float = 0.0,
) -> pd.DataFrame:
    """Run an equal-weight rebalanced benchmark.

    The strategy starts after the same lookback period used by the Mapper
    strategy. At each rebalance date, all available assets in the current window
    receive the same weight and the portfolio is held until the next rebalance.

    Transaction costs are applied proportionally to turnover, using the same
    convention as in backtest_tda.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price matrix indexed by date, with one column per asset.
    lookback_days : int
        Number of past observations used to define the available asset universe.
    rebalance_days : int
        Number of observations between consecutive rebalances.
    tc_bps : float, default=0.0
        Transaction cost in basis points, applied proportionally to turnover.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with portfolio returns, NAV and turnover.
    """
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)
    dates = prices.index

    if len(dates) < lookback_days + 10:
        raise ValueError("Time series is too short for the selected lookback.")

    # Equal-weight is evaluated with the same rebalance schedule and the same
    # transaction-cost convention as the Mapper strategies.
    rebalance_idx = list(range(lookback_days, len(dates) - 1, rebalance_days))

    port_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)

    previous_weights: Dict[str, float] = {}

    for k, idx in enumerate(rebalance_idx):
        window_raw = prices.iloc[idx - lookback_days: idx + 1]
        window = window_raw.sort_index().ffill().dropna(axis=1, how="any")
        assets = list(window.columns)

        if len(assets) == 0:
            continue

        weights = _equal_weight(assets)
        turnover_value = _turnover(previous_weights, weights)

        end_idx = (
            rebalance_idx[k + 1]
            if k + 1 < len(rebalance_idx)
            else len(dates) - 1
        )
        hold_dates = dates[idx + 1: end_idx + 1]

        cost = turnover_value * (tc_bps / 10000.0) if tc_bps > 0 else 0.0

        for j, date in enumerate(hold_dates):
            day_rets = rets.loc[date]
            daily_return = _portfolio_return(day_rets, weights)

            if j == 0 and cost > 0:
                daily_return -= cost
                turnover.loc[date] = turnover_value

            port_ret.loc[date] = daily_return

        previous_weights = dict(weights)

    nav = (1.0 + port_ret).cumprod()

    return pd.DataFrame(
        {
            "port_ret": port_ret,
            "port_nav": nav,
            "turnover": turnover,
        }
    )
