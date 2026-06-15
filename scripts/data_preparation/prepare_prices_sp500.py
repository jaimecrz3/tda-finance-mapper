"""Prepare raw CRSP/WRDS S&P 500 daily data.

This script reads the raw WRDS export, keeps only the columns required by the
project and saves a lighter cleaned file. It is intended to be run manually
before creating the monthly dataset.
"""

from __future__ import annotations

import pandas as pd


RAW_CSV_PATH = "data/SP500_data_sin_preprocesar.csv"
OUTPUT_CSV_PATH = "data/sp500_crsp_clean.csv"
OUTPUT_PARQUET_PATH = "data/sp500_crsp_clean.parquet"

USECOLS = ["PERMNO", "Ticker", "DlyCalDt", "DlyPrc", "DlyRet"]

DTYPES = {
    "PERMNO": "int32",
    "Ticker": "category",
    "DlyPrc": "float32",
    "DlyRet": "float32",
}


def main() -> None:
    """Load raw WRDS data and save a cleaned daily file."""
    df = pd.read_csv(
        RAW_CSV_PATH,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["DlyCalDt"],
    )

    # Sorting by asset and date makes later aggregations deterministic.
    df = df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)

    print(df.head())
    print(df.info(memory_usage="deep"))

    df.to_csv(OUTPUT_CSV_PATH, index=False)

    # Parquet is usually faster and lighter than CSV. Keep this line active if
    # pyarrow is installed in the environment.
    df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("\nFiles saved:")
    print(f"- {OUTPUT_CSV_PATH}")
    print(f"- {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    main()
