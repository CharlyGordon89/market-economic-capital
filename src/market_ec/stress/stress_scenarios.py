from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from market_ec.sim.engine import simulate_paths, _normalize_weights
from market_ec.sim.shocks import (
    sample_gaussian_copula_normals,
    sample_t_copula_normals,
)
# reuse PD projection from shocks to keep behavior consistent
from market_ec.sim.shocks import _to_corr_pd


@dataclass
class ParametricStress:
    name: str
    vol_scale: float = 1.0    # multiply asset vols
    rho_scale: float = 1.0    # multiply off-diagonal correlations


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _scale_corr(corr: np.ndarray, rho_scale: float) -> np.ndarray:
    """
    Scale off-diagonal correlations by rho_scale, keep diag=1,
    then project to nearest PD correlation matrix.
    """
    C = np.asarray(corr, dtype=float).copy()
    d = C.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j:
                C[i, j] = np.clip(C[i, j] * rho_scale, -0.999, 0.999)
            else:
                C[i, j] = 1.0
    C = 0.5 * (C + C.T)
    C = _to_corr_pd(C)
    return C


def run_parametric_stress(prices: pd.DataFrame,
                          models_dir: Path,
                          risk_dir: Path,
                          dep_tickers: list[str],
                          weights_cfg: dict[str, float],
                          horizon_days: int,
                          n_sims: int,
                          copula_kind: str,
                          copula_params: dict,
                          stresses: list[ParametricStress],
                          seed: int = 42) -> list[dict]:
    """
    For each stress: adjust correlations and vol scaling, simulate portfolio, record VaR changes.
    """
    rng = np.random.default_rng(seed)
    weights = _normalize_weights(weights_cfg, dep_tickers)

    d = len(dep_tickers)
    base_corr = _to_corr_pd(np.asarray(copula_params["corr"], dtype=float))

    # helper to simulate via engine with patched corr/vol_scale:
    # For transparency and consistency, we generate shocks here (as in Sprint 3)
    results = []
    for s in stresses:
        C = _scale_corr(base_corr, s.rho_scale)

        if copula_kind == "gaussian":
            Z = sample_gaussian_copula_normals(C, n_sims, horizon_days, d, rng)
        else:
            df = int(copula_params.get("df", 6))
            Z = sample_t_copula_normals(C, df, n_sims, horizon_days, d, rng)

        # --- Use the same return-generation logic as engine.simulate_paths ---
        from market_ec.sim.engine import _load_model_params, TRADING_DAYS
        gbm_params, ou_params, garch_params = _load_model_params(models_dir)

        used = [t for t in dep_tickers if t in weights]
        if not used:
            raise ValueError("No overlapping tickers between weights and dependence set.")
        idx = [dep_tickers.index(t) for t in used]
        w = np.array([weights[t] for t in used], dtype=float)
        w = w / w.sum()

        R = np.zeros((n_sims, horizon_days, len(used)), dtype=float)
        Z_used = Z[:, :, idx]

        for j, t in enumerate(used):
            z = Z_used[:, :, j] * s.vol_scale
            if t in garch_params:
                gp = garch_params[t]
                h0 = gp.unconditional_var if gp.unconditional_var > 0 else gp.omega / max(1e-8, (1 - gp.alpha1 - gp.beta1))
                h = np.full((n_sims,), h0, dtype=float)
                mu_d = gp.mu
                for tt in range(horizon_days):
                    R[:, tt, j] = mu_d + np.sqrt(h) * z[:, tt]
                    eps_t = np.sqrt(h) * z[:, tt]
                    h = gp.omega + gp.alpha1 * (eps_t ** 2) + gp.beta1 * h
            elif t in gbm_params:
                gp = gbm_params[t]
                mu_d = gp.mu_annual / TRADING_DAYS
                sig_d = gp.sigma_annual / np.sqrt(TRADING_DAYS)
                R[:, :, j] = mu_d + sig_d * z
            elif t in ou_params:
                op = ou_params[t]
                dt = op.dt
                px = prices[prices["ticker"] == t].sort_values("date")["adj_close"].astype(float)
                if px.empty:
                    raise ValueError(f"Missing price history for OU ticker {t}")
                x0 = np.log(float(px.iloc[-1]))
                X = np.empty((n_sims, horizon_days + 1), dtype=float)
                X[:, 0] = x0
                sigma_dt = op.sigma * np.sqrt(dt)
                for tt in range(horizon_days):
                    X[:, tt + 1] = X[:, tt] + op.kappa * (op.theta - X[:, tt]) * dt + sigma_dt * z[:, tt]
                R[:, :, j] = X[:, 1:] - X[:, :-1]
            else:
                R[:, :, j] = 0.0

        rp = (R * w[None, None, :]).sum(axis=2)
        total_log_ret = rp.sum(axis=1)
        losses = -np.expm1(total_log_ret)

        p995, p999 = np.percentile(losses, [99.5, 99.9])
        results.append({
            "name": s.name,
            "VaR_0.995": float(p995),
            "VaR_0.999": float(p999),
            "mean_loss": float(losses.mean()),
            "rho_scale": float(s.rho_scale),
            "vol_scale": float(s.vol_scale),
        })

    # save
    out = {"parametric_results": results}
    _save_json(out, risk_dir / "stress_parametric.json")
    return results


def run_historical_stress(prices: pd.DataFrame,
                          weights_cfg: dict[str, float],
                          windows: list[dict],
                          out_path: Path) -> list[dict]:
    from market_ec.sim.engine import _normalize_weights
    w_cfg = weights_cfg.copy()
    present = sorted(prices["ticker"].unique().tolist())
    weights = _normalize_weights(w_cfg, present)

    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    used = [t for t in wide.columns if t in weights]
    w = np.array([weights[t] for t in used], dtype=float)
    w = w / w.sum()

    ret = wide[used].pct_change().dropna()
    port_ret = (ret * w).sum(axis=1)

    out = []
    for win in windows:
        s = pd.to_datetime(win["start"])
        e = pd.to_datetime(win["end"])
        sl = port_ret.loc[(port_ret.index >= s) & (port_ret.index <= e)]
        if len(sl) == 0:
            continue
        cum = (1.0 + sl).prod() - 1.0
        peak = (sl + 1.0).cumprod().cummax()
        dd = ((sl + 1.0).cumprod() / peak - 1.0).min()
        out.append({"name": win["name"], "cum_return": float(cum), "max_drawdown": float(dd), "n_days": int(len(sl))})

    result = {"historical_results": out}
    _save_json(result, out_path)
    return out
