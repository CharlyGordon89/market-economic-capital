"""Return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Compute log returns for each ticker.

    Parameters
    ----------
    df:
        Price DataFrame containing ``ticker``, ``date`` and ``price_col`` columns.
    price_col:
        Column to use for price (defaults to ``adj_close``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ticker``, ``date`` and ``log_return``.
    """

    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    df_sorted = df.sort_values(["ticker", "date"]).copy()
    df_sorted["log_return"] = df_sorted.groupby("ticker")[price_col].transform(
        lambda s: np.log(s / s.shift(1))
    )
    rets = df_sorted.dropna(subset=["log_return"])
    return rets[["ticker", "date", "log_return"]]
