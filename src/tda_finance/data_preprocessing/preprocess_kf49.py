"""Preprocessing utilities for Kenneth French 49 Industry Portfolios.

This module provides helper functions to convert monthly returns into synthetic
price indices and to download the datasets used in the experiments.
"""

from __future__ import annotations

import pandas as pd
from pandas_datareader.data import get_data_famafrench


def returns_to_price_index(
    rets: pd.DataFrame,
    base: float = 100.0,
) -> pd.DataFrame:
    """Convert return series into synthetic price indices.

    Each column is treated independently. If an asset has missing values at the
    beginning of the sample, its synthetic price index starts at the first valid
    return observation.

    Parameters
    ----------
    rets : pandas.DataFrame
        Return matrix indexed by date, with one column per asset. Returns must
        be expressed in decimal form.
    base : float, default=100.0
        Initial value used for each synthetic price index.

    Returns
    -------
    pandas.DataFrame
        Synthetic price matrix with the same index and columns as the input.
    """
    prices = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)

    for column in rets.columns:
        asset_returns = rets[column]
        first_valid_date = asset_returns.first_valid_index()

        if first_valid_date is None:
            continue

        # The price path represents the value of investing base units and
        # reinvesting returns from the first valid observation onward.
        path = (1.0 + asset_returns.loc[first_valid_date:]).cumprod() * base
        prices.loc[first_valid_date:, column] = path

    return prices


def load_kf49_prices_from_returns(
    rets: pd.DataFrame,
    base: float = 100.0,
    require_complete_panel: bool = True,
) -> pd.DataFrame:
    """Build synthetic prices from Kenneth French 49 industry returns.

    Parameters
    ----------
    rets : pandas.DataFrame
        Monthly industry returns in decimal form.
    base : float, default=100.0
        Initial value used for each synthetic price index.
    require_complete_panel : bool, default=True
        If True, industries with any missing value are removed before building
        prices.

    Returns
    -------
    pandas.DataFrame
        Synthetic monthly price matrix sorted by date.
    """
    # For the KF49 experiments, a complete panel can be required because the
    # universe is small and stable.
    if require_complete_panel:
        rets = rets.dropna(axis=1, how="any")

    return returns_to_price_index(rets, base=base).sort_index()


def main() -> None:
    """Download KF49 industry returns and Fama-French 3 factors."""
    # Kenneth French returns are provided in percentages, so they are converted
    # to decimal form before being saved.
    dataset = get_data_famafrench("49_Industry_Portfolios", start="1970-01")
    ind49_monthly = (dataset[0] / 100.0).round(5)
    ind49_monthly.to_csv("data/49_industries_portfolios_monthly.csv")

    # The risk-free rate is later aligned with the portfolio return series.
    ff3 = get_data_famafrench("F-F_Research_Data_Factors", start="1970-01")[0]
    ff3 = (ff3 / 100.0).round(5)
    ff3.to_csv("data/ff3_monthly.csv")

    print("Download completed.")


if __name__ == "__main__":
    main()
