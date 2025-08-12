"""Download price data from Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import yfinance as yf


@dataclass
class PriceRequest:
    """Request parameters for downloading prices."""

    tickers: Sequence[str]
    start: str | pd.Timestamp
    end: str | pd.Timestamp
    interval: str = "1d"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case."""
    return df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))


def download_prices(req: PriceRequest) -> pd.DataFrame:
    """Download OHLCV data for the requested tickers.

    Parameters
    ----------
    req:
        Request describing tickers and date range.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, date, open, high, low, close,
        adj_close, volume.
    """

    data = yf.download(
        tickers=list(req.tickers),
        start=req.start,
        end=req.end,
        interval=req.interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        )

    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in req.tickers:
            if ticker not in data.columns.levels[0]:
                continue
            df_t = data[ticker].copy()
            df_t = _clean_columns(df_t)
            df_t["ticker"] = ticker
            frames.append(df_t.reset_index())
    else:
        df_t = _clean_columns(data)
        df_t["ticker"] = list(req.tickers)[0]
        frames.append(df_t.reset_index())

    df = pd.concat(frames, ignore_index=True)
    df = _clean_columns(df)
    df = df[[
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]]
    return df
