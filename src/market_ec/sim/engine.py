from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from market_ec.sim.shocks import sample_gaussian_copula_normals, sample_t_copula_normals

TRADING_DAYS = 252

@dataclass
class GBMParam:
    mu_annual: float
    sigma_annual: float

@dataclass
class OUParam:
    kappa: float
    theta: float
    sigma: float
    dt: float

@dataclass
class GARCHParam:
    mu: float
    omega: float
    alpha1: float
    beta1: float
    unconditional_var: float
    persistence: float


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _load_model_params(models_dir: Path) -> Tuple[Dict[str, GBMParam], Dict[str, OUParam], Dict[str, GARCHParam]]:
    gbm, ou, garch = {}, {}, {}

    gbm_path = models_dir / "gbm_params.json"
    if gbm_path.exists():
        d = _read_json(gbm_path)
        for k, v in d.items():
            gbm[k] = GBMParam(mu_annual=float(v["mu_annual"]), sigma_annual=float(v["sigma_annual"]))

    ou_path = models_dir / "ou_params.json"
    if ou_path.exists():
        d = _read_json(ou_path)
        for k, v in d.items():
            ou[k] = OUParam(
                kappa=float(v["kappa"]),
                theta=float(v["theta"]),
                sigma=float(v["sigma"]),
                dt=float(v.get("dt", 1.0 / TRADING_DAYS)),
            )

    garch_path = models_dir / "garch11_params.json"
    if garch_path.exists():
        d = _read_json(garch_path)
        for k, v in d.items():
            garch[k] = GARCHParam(
                mu=float(v["mu"]),
                omega=float(v["omega"]),
                alpha1=float(v["alpha1"]),
                beta1=float(v["beta1"]),
                unconditional_var=float(v["unconditional_var"]),
                persistence=float(v["persistence"]),
            )
    return gbm, ou, garch


def _load_copula(models_dir: Path) -> Tuple[str, dict]:
    g = models_dir / "copula_gaussian.json"
    t = models_dir / "copula_t.json"
    if g.exists():
        data = _read_json(g)
        return "gaussian", data
    if t.exists():
        data = _read_json(t)
        return "t", data
    raise FileNotFoundError("No copula params found (copula_gaussian.json or copula_t.json).")


def _normalize_weights(weights: Dict[str, float], available: List[str]) -> Dict[str, float]:
    filt = {k: v for k, v in weights.items() if k in available}
    s = sum(filt.values())
    if s <= 0:
        raise ValueError("Sum of portfolio weights <= 0 after intersecting with available tickers.")
    return {k: v / s for k, v in filt.items()}


def _build_shocks(
    copula_kind: str,
    copula_params: dict,
    tickers: List[str],
    n_sims: int,
    T: int,
    rng: np.random.Generator,
) -> np.ndarray:
    corr = np.array(copula_params["corr"], dtype=float)
    d = len(tickers)
    if corr.shape != (d, d):
        raise ValueError(f"Copula corr shape {corr.shape} does not match n_assets={d}")

    if copula_kind == "gaussian":
        Z = sample_gaussian_copula_normals(corr, n_sims, T, d, rng)
    elif copula_kind == "t":
        df = int(copula_params.get("df", 6))
        Z = sample_t_copula_normals(corr, df, n_sims, T, d, rng)
    else:
        raise ValueError(f"Unknown copula kind: {copula_kind}")
    return Z  # shape (n_sims, T, d)


def simulate_paths(
    prices: pd.DataFrame,
    models_dir: Path,
    tickers: List[str],
    weights: Dict[str, float],
    n_sims: int,
    T: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    """
    Simulate daily log-returns for the requested tickers using:
    - GARCH if available, else GBM constants; OU simulated on log-price, converted to returns.
    Returns:
      R: shape (n_sims, T, d)  per-asset daily log-returns
      used_tickers: tickers actually simulated (intersection with models/copula).
    """
    # Load models
    gbm_params, ou_params, garch_params = _load_model_params(models_dir)
    copula_kind, copula_data = _load_copula(models_dir)
    copula_tickers = list(weights.keys())  # already intersected before call

    # Determine final asset order (must match copula corr order!)
    # We assume the copula was fitted on 'dependence.tickers' order in config,
    # and weights were intersected against those. We'll use that order filtered by weights.
    full_order = tickers  # dependence tickers, from config
    used = [t for t in full_order if t in weights]

    if not used:
        raise ValueError("No overlapping tickers between weights and dependence set.")

    # Build copula shocks for used assets
    # Extract corr submatrix in the same order
    corr_full = np.array(copula_data["corr"], dtype=float)
    idx = [full_order.index(t) for t in used]
    corr = corr_full[np.ix_(idx, idx)]
    Z = _build_shocks(copula_kind, {"corr": corr, **{k: v for k, v in copula_data.items() if k != "corr"}}, used, n_sims, T, rng)

    d = len(used)
    R = np.zeros((n_sims, T, d), dtype=float)

    # Precompute daily params
    for j, t in enumerate(used):
        z = Z[:, :, j]
        if t in garch_params:
            gp = garch_params[t]
            # GARCH recursion per simulation
            h0 = gp.unconditional_var if gp.unconditional_var > 0 else gp.omega / max(1e-8, (1 - gp.alpha1 - gp.beta1))
            h = np.full((n_sims,), h0, dtype=float)
            mu_d = gp.mu  # already daily mean (from fit)
            for tt in range(T):
                # returns: r_t = mu_d + sqrt(h_t) * z_t
                R[:, tt, j] = mu_d + np.sqrt(h) * z[:, tt]
                # variance update with today's shock
                eps_t = np.sqrt(h) * z[:, tt]
                h = gp.omega + gp.alpha1 * (eps_t ** 2) + gp.beta1 * h
        elif t in gbm_params:
            gp = gbm_params[t]
            mu_d = gp.mu_annual / TRADING_DAYS
            sig_d = gp.sigma_annual / np.sqrt(TRADING_DAYS)
            R[:, :, j] = mu_d + sig_d * z
        elif t in ou_params:
            op = ou_params[t]
            # simulate log-price then diff to returns
            dt = op.dt
            # start from last known price
            px = prices[prices["ticker"] == t].sort_values("date")["adj_close"].astype(float)
            if px.empty:
                raise ValueError(f"Missing price history for OU ticker {t}")
            x0 = np.log(float(px.iloc[-1]))
            X = np.empty((n_sims, T + 1), dtype=float)
            X[:, 0] = x0
            sigma_dt = op.sigma * np.sqrt(dt)
            for tt in range(T):
                X[:, tt + 1] = (
                    X[:, tt] + op.kappa * (op.theta - X[:, tt]) * dt + sigma_dt * z[:, tt]
                )
            # returns are log-diffs
            R[:, :, j] = X[:, 1:] - X[:, :-1]
        else:
            # Fallback: zero return (shouldn't happen in well-configured runs)
            R[:, :, j] = 0.0

    return R, used
