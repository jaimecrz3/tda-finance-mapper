"""Create monthly CRSP/WRDS S&P 500 returns from daily data.

This script aggregates daily returns into monthly compounded returns for each
PERMNO. The resulting file is the input used by the package preprocessing
function load_sp500_prices_from_monthly_returns.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


InputFormat = Literal["csv", "parquet"]

INPUT_FORMAT: InputFormat = "csv"

INPUT_CSV_PATH = "data/sp500_crsp_clean.csv"
INPUT_PARQUET_PATH = "data/sp500_crsp_clean.parquet"

OUTPUT_CSV_PATH = "data/sp500_crsp_monthly_clean.csv"
OUTPUT_PARQUET_PATH = "data/sp500_crsp_monthly_clean.parquet"


def compound_return(returns: pd.Series) -> float:
    """Compute the compounded return of a daily return series."""
    clean_returns = pd.to_numeric(returns, errors="coerce").dropna().astype(float)

    if len(clean_returns) == 0:
        return float(np.nan)

    returns_array = clean_returns.to_numpy(dtype=float)
    growth = float(np.prod(1.0 + returns_array))

    return growth - 1.0


def load_daily_data(input_format: InputFormat) -> pd.DataFrame:
    """Load the cleaned daily CRSP file."""
    if input_format == "csv":
        return pd.read_csv(
            INPUT_CSV_PATH,
            parse_dates=["DlyCalDt"],
            dtype={
                "PERMNO": "int32",
                "Ticker": "category",
                "DlyPrc": "float32",
                "DlyRet": "float32",
            },
        )

    if input_format == "parquet":
        return pd.read_parquet(INPUT_PARQUET_PATH)

    raise ValueError("INPUT_FORMAT must be either 'csv' or 'parquet'.")


def main() -> None:
    """Aggregate daily returns into monthly compounded returns."""
    df = load_daily_data(INPUT_FORMAT)

    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df["DlyRet"] = pd.to_numeric(df["DlyRet"], errors="coerce")
    df["DlyPrc"] = pd.to_numeric(df["DlyPrc"], errors="coerce")

    df = df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)
    df["Month"] = df["DlyCalDt"].dt.to_period("M")

    # Monthly returns are computed by compounding all valid daily returns within
    # each asset-month pair.
    monthly = (
        df
        .groupby(["PERMNO", "Month"], observed=True)
        .agg(
            Ticker=("Ticker", "last"),
            MonthlyRet=("DlyRet", compound_return),
            LastPrice=("DlyPrc", "last"),
            NDays=("DlyRet", "count"),
        )
        .reset_index()
    )

    monthly["Month"] = monthly["Month"].dt.to_timestamp()
    monthly = monthly.sort_values(["PERMNO", "Month"]).reset_index(drop=True)

    # Months without valid returns are not useful for the monthly backtest.
    monthly = monthly.dropna(subset=["MonthlyRet"])
    monthly = monthly[monthly["NDays"] > 0].copy()

    print(monthly.head())
    print(monthly.info(memory_usage="deep"))

    print("\nNumber of rows:", len(monthly))
    print("Number of unique PERMNO:", monthly["PERMNO"].nunique())
    print("Start date:", monthly["Month"].min())
    print("End date:", monthly["Month"].max())

    print("\nDuplicated PERMNO-Month observations:")
    print(monthly.duplicated(["PERMNO", "Month"]).sum())

    print("\nMissing values:")
    print(monthly.isna().sum())

    print("\nMonthlyRet statistics:")
    print(monthly["MonthlyRet"].describe())

    print("\nNumber of assets per month:")
    print(monthly.groupby("Month")["PERMNO"].nunique().describe())

    monthly.to_csv(OUTPUT_CSV_PATH, index=False)
    monthly.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("\nFiles saved:")
    print(f"- {OUTPUT_CSV_PATH}")
    print(f"- {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()
