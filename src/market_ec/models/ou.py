from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gbm import TRADING_DAYS


@dataclass(frozen=True)
class OUParams:
    """Parameters for the Ornstein–Uhlenbeck process."""

    kappa: float
    theta: float
    sigma: float
    dt: float

    def to_dict(self) -> dict:
        return {
            "kappa": float(self.kappa),
            "theta": float(self.theta),
            "sigma": float(self.sigma),
            "dt": float(self.dt),
        }


def fit_ou_from_log_price(log_price: pd.Series, dt: float | None = None) -> OUParams:
    """Fit continuous-time OU parameters from a log-price series.

    The discrete-time dynamics are assumed to follow an AR(1) process:
    ``x_{t+1} = a + b x_t + eps_t``.  Parameters are then mapped to the
    continuous-time OU process ``dX_t = kappa (theta - X_t) dt + sigma dW_t``.

    Parameters
    ----------
    log_price : pd.Series
        Series of log prices. Missing values are ignored.
    dt : float, optional
        Time step in years. Defaults to ``1/TRADING_DAYS``.
    """
    x = pd.Series(log_price).dropna().to_numpy(dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two observations to fit OU")

    if dt is None:
        dt = 1.0 / TRADING_DAYS

    x_t = x[:-1]
    x_tp1 = x[1:]
    design = np.column_stack([np.ones_like(x_t), x_t])
    # Ordinary least squares: x_{t+1} = a + b x_t
    beta, _, _, _ = np.linalg.lstsq(design, x_tp1, rcond=None)
    a, b = beta
    resid = x_tp1 - (a + b * x_t)
    s2 = np.var(resid, ddof=2)

    # Map to continuous-time parameters
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = np.sqrt(s2 * 2.0 * kappa / (1.0 - b**2))

    return OUParams(kappa=float(kappa), theta=float(theta), sigma=float(sigma), dt=float(dt))
