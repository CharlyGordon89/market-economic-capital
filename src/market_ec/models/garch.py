from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class GARCH11Params:
    """Parameters for a GARCH(1,1) model."""

    mu: float
    omega: float
    alpha1: float
    beta1: float
    unconditional_var: float
    persistence: float

    def to_dict(self) -> dict:
        return {
            "mu": float(self.mu),
            "omega": float(self.omega),
            "alpha1": float(self.alpha1),
            "beta1": float(self.beta1),
            "unconditional_var": float(self.unconditional_var),
            "persistence": float(self.persistence),
        }


def _garch11_negloglike(params: np.ndarray, r: np.ndarray) -> float:
    mu, omega, alpha, beta = params
    # Enforce parameter constraints; penalize invalid sets
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e12
    eps = r - mu
    n_obs = eps.size
    h = np.empty(n_obs)
    # start variance at unconditional variance
    h0 = omega / (1.0 - alpha - beta)
    h[0] = h0
    for t in range(1, n_obs):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    ll = 0.5 * (np.log(2 * np.pi) + np.log(h) + (eps**2) / h)
    return float(np.sum(ll))


def fit_garch11(log_returns: pd.Series) -> tuple[GARCH11Params, pd.Series]:
    """Fit a GARCH(1,1) model to a log-return series.

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns. Missing values are ignored.

    Returns
    -------
    params : :class:`GARCH11Params`
        Estimated parameter dataclass.
    std_resid : pd.Series
        Standardized residuals ``(r_t - mu)/sqrt(h_t)`` from the fitted model.
    """
    r = pd.Series(log_returns).dropna().to_numpy(dtype=float)
    if r.size < 2:
        raise ValueError("Need at least two observations to fit GARCH")

    mu0 = float(np.mean(r))
    var0 = float(np.var(r, ddof=1))
    omega0 = 0.1 * var0
    alpha0 = 0.05
    beta0 = 0.9
    x0 = np.array([mu0, omega0, alpha0, beta0])

    res = minimize(_garch11_negloglike, x0, args=(r,), method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("GARCH(1,1) fit did not converge: " + res.message)
    mu, omega, alpha1, beta1 = res.x
    eps = r - mu
    n_obs = r.size
    h = np.empty(n_obs)
    h[0] = omega / (1.0 - alpha1 - beta1)
    for t in range(1, n_obs):
        h[t] = omega + alpha1 * eps[t - 1] ** 2 + beta1 * h[t - 1]

    std_resid = pd.Series(eps / np.sqrt(h), index=pd.Series(log_returns).dropna().index)

    uncond_var = float(omega / (1.0 - alpha1 - beta1))
    persistence = float(alpha1 + beta1)
    params = GARCH11Params(
        mu=float(mu),
        omega=float(omega),
        alpha1=float(alpha1),
        beta1=float(beta1),
        unconditional_var=uncond_var,
        persistence=persistence,
    )
    return params, std_resid
