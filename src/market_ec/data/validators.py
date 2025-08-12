"""Validation utilities for downloaded price data."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def check_price_columns(df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLUMNS) -> None:
    """Ensure all required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def check_date_continuity(df: pd.DataFrame) -> None:
    """Check that dates for each ticker are unique and reasonably continuous."""
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("date column must be datetime-like")

    for ticker, grp in df.groupby("ticker"):
        dates = grp["date"].sort_values()
        if dates.duplicated().any():
            raise ValueError(f"Duplicate dates found for ticker {ticker}")
        gaps = dates.diff().dropna().dt.days
        if (gaps > 5).any():
            raise ValueError(f"Date gaps detected for ticker {ticker}")


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Run standard validations on a prices DataFrame."""
    check_price_columns(df)
    check_date_continuity(df)
    return df
