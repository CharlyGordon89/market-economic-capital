from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


@dataclass(frozen=True)
class GBMParams:
    """Parameters of a geometric Brownian motion."""

    mu_annual: float
    sigma_annual: float

    def to_dict(self) -> dict:
        return {
            "mu_annual": float(self.mu_annual),
            "sigma_annual": float(self.sigma_annual),
        }


def fit_gbm_from_log_returns(log_returns: pd.Series) -> GBMParams:
    """Fit GBM parameters from a series of (daily) log returns.

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns. Missing values are ignored.

    Returns
    -------
    GBMParams
        Dataclass containing annualized drift and volatility.
    """
    r = pd.Series(log_returns).dropna().to_numpy(dtype=float)
    if r.size < 2:
        raise ValueError("Need at least two observations to fit GBM")

    mu_d = float(np.mean(r))
    sigma_d = float(np.std(r, ddof=1))

    mu_ann = mu_d * TRADING_DAYS
    sigma_ann = sigma_d * np.sqrt(TRADING_DAYS)

    return GBMParams(mu_annual=mu_ann, sigma_annual=sigma_ann)
