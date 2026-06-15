"""Preprocessing utilities for S&P 500 CRSP monthly data.

This module loads the cleaned CRSP/WRDS monthly file used in the experiments
and converts monthly stock returns into synthetic price indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_sp500_prices_from_monthly_returns(
    path: str = "data/sp500_crsp_monthly_clean.parquet",
    file_format: str = "parquet",
    start_price: float = 100.0,
    min_price: float = 1e-6,
) -> pd.DataFrame:
    """Load monthly S&P 500 returns and convert them into synthetic prices.

    The expected input file contains at least the following columns:
    PERMNO, Month and MonthlyRet. PERMNO is used as the asset identifier because
    it is more stable than the ticker.

    Parameters
    ----------
    path : str, default="data/sp500_crsp_monthly_clean.parquet"
        Path to the cleaned CRSP monthly file.
    file_format : {"parquet", "csv"}, default="parquet"
        File format used to read the input data.
    start_price : float, default=100.0
        Initial value used for each synthetic price index.
    min_price : float, default=1e-6
        Lower bound applied to gross returns to avoid zero synthetic prices.

    Returns
    -------
    pandas.DataFrame
        Synthetic price matrix indexed by month-end date, with one column per
        PERMNO.

    Raises
    ------
    ValueError
        If the file format is unsupported, required columns are missing, or
        duplicated PERMNO-month observations are found.
    """
    if file_format == "parquet":
        df = pd.read_parquet(path)
    elif file_format == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("file_format must be either 'parquet' or 'csv'.")

    required_columns = {"PERMNO", "Month", "MonthlyRet"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["Month"] = pd.to_datetime(df["Month"])
    df["PERMNO"] = df["PERMNO"].astype(str)
    df["MonthlyRet"] = pd.to_numeric(df["MonthlyRet"], errors="coerce")

    df = df.dropna(subset=["Month", "PERMNO", "MonthlyRet"]).copy()

    # Ensure that each asset-month pair appears only once before pivoting.
    df = df.sort_values(["PERMNO", "Month"])
    duplicated = df.duplicated(["PERMNO", "Month"]).sum()

    if duplicated > 0:
        raise ValueError(f"Duplicated PERMNO-Month observations: {duplicated}")

    returns = (
        df
        .pivot(index="Month", columns="PERMNO", values="MonthlyRet")
        .sort_index()
    )

    # Missing returns are kept as NaN. This preserves periods before an asset
    # enters the universe and after it leaves it.
    gross_returns = 1.0 + returns

    # A return of -100% would create a zero price and later cause numerical
    # issues when log-returns are computed.
    gross_returns = gross_returns.clip(lower=min_price)

    prices = gross_returns.copy()

    for column in prices.columns:
        asset_gross_returns = gross_returns[column].dropna()

        if asset_gross_returns.empty:
            prices[column] = np.nan
            continue

        synthetic_prices = start_price * asset_gross_returns.cumprod()
        prices[column] = synthetic_prices.reindex(prices.index)

    prices.index = pd.to_datetime(prices.index).to_period("M").to_timestamp("M")

    return prices.sort_index()


def load_sp500_returns_matrix(
    path: str = "data/sp500_crsp_monthly_clean.parquet",
    file_format: str = "parquet",
) -> pd.DataFrame:
    """Load monthly S&P 500 returns as a Month x PERMNO matrix.

    Parameters
    ----------
    path : str, default="data/sp500_crsp_monthly_clean.parquet"
        Path to the cleaned CRSP monthly file.
    file_format : {"parquet", "csv"}, default="parquet"
        File format used to read the input data.

    Returns
    -------
    pandas.DataFrame
        Monthly return matrix indexed by month-end date, with one column per
        PERMNO.

    Raises
    ------
    ValueError
        If the file format is unsupported.
    """
    if file_format == "parquet":
        df = pd.read_parquet(path)
    elif file_format == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("file_format must be either 'parquet' or 'csv'.")

    df["Month"] = pd.to_datetime(df["Month"])
    df["PERMNO"] = df["PERMNO"].astype(str)
    df["MonthlyRet"] = pd.to_numeric(df["MonthlyRet"], errors="coerce")

    returns = (
        df
        .dropna(subset=["Month", "PERMNO", "MonthlyRet"])
        .pivot(index="Month", columns="PERMNO", values="MonthlyRet")
        .sort_index()
    )

    returns.index = pd.to_datetime(returns.index).to_period("M").to_timestamp("M")

    return returns
