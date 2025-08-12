from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mlflow

from market_ec.utils.config import load_config
from market_ec.utils.logging import get_logger
from market_ec.utils.mlflow_utils import start_run
from market_ec.sim.engine import simulate_paths, _normalize_weights, TRADING_DAYS
from market_ec.risk.metrics import var_es

print("Script started")  # First line of the file

def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def run_simulation(cfg_path: str) -> None:
    logger = get_logger()
    cfg = load_config(cfg_path)

    data_dir = Path(cfg["artifacts"]["data_dir"])
    models_dir = Path(cfg["artifacts"]["models_dir"])
    risk_dir = Path(cfg["artifacts"].get("risk_dir", "risk"))
    risk_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    deps = cfg["dependence"]
    dep_tickers = list(deps["tickers"])          # reference order of the copula
    weights_cfg = cfg["portfolio"]["weights"]    # may include symbols that were dropped earlier
    weights = _normalize_weights(weights_cfg, dep_tickers)

    T = int(cfg["risk"]["horizon_days"])
    n_sims = int(cfg["risk"]["n_sims"])
    alphas = list(map(float, cfg["risk"]["alphas"]))
    seed = int(cfg.get("run", {}).get("random_seed", 42))
    save_samples = bool(cfg["risk"].get("save_samples", False))

    prices = pd.read_parquet(data_dir / "prices.parquet")

    rng = np.random.default_rng(seed)

    with start_run(cfg["mlflow"]["experiment"], run_name="simulate_portfolio", tags={"sprint": "3"}):
        # Simulate per-asset daily log-returns
        R, used = simulate_paths(
            prices=prices,
            models_dir=models_dir,
            tickers=dep_tickers,
            weights=weights,
            n_sims=n_sims,
            T=T,
            rng=rng,
        )

        # Align weights to used order
        w = np.array([weights[t] for t in used], dtype=float)
        w = w / w.sum()

        # portfolio daily log-return paths
        rp = (R * w[None, None, :]).sum(axis=2)  # (n_sims, T)

        # 1Y cumulative return from daily log-returns
        total_log_ret = rp.sum(axis=1)           # (n_sims,)
        total_simple_ret = np.expm1(total_log_ret)  # exp(sum) - 1
        losses = -total_simple_ret               # define loss as negative return

        # Risk metrics
        metrics = var_es(losses, alphas)

        # Diversification baseline: independent shocks (same marginals)
        # Shuffle assets' Z by permuting sims per asset (approx independence)
        # We recompute rp_indep by permuting along simulation axis differently per asset.
        n_sims_eff = R.shape[0]
        R_indep = np.empty_like(R)
        for j in range(R.shape[2]):
            perm = rng.permutation(n_sims_eff)
            R_indep[:, :, j] = R[perm, :, j]
        rp_indep = (R_indep * w[None, None, :]).sum(axis=2)
        total_log_ret_indep = rp_indep.sum(axis=1)
        losses_indep = -np.expm1(total_log_ret_indep)
        metrics_indep = var_es(losses_indep, alphas)

        # Save artifacts
        summary = {
            "used_tickers": used,
            "weights": {t: float(w[i]) for i, t in enumerate(used)},
            "horizon_days": T,
            "n_sims": n_sims,
            "alphas": alphas,
            "metrics": {str(a): {"VaR": v, "ES": e} for a, (v, e) in metrics.items()},
            "metrics_indep": {str(a): {"VaR": v, "ES": e} for a, (v, e) in metrics_indep.items()},
            "diversification_benefit": {
                str(a): {
                    "VaR_reduction": float(metrics_indep[a][0] - metrics[a][0]),
                    "ES_reduction": float(metrics_indep[a][1] - metrics[a][1]),
                }
                for a in alphas
            },
        }
        out_json = risk_dir / "portfolio_risk_summary.json"
        _save_json(summary, out_json)

        # Optionally save a small sample of losses for plots downstream
        if save_samples:
            df = pd.DataFrame({"loss": losses})
            df.to_parquet(risk_dir / "loss_samples.parquet", index=False)

        # MLflow logging
        mlflow.log_param("sim_used_tickers", ",".join(used))
        mlflow.log_param("sim_T", T)
        mlflow.log_param("sim_n_sims", n_sims)
        for a, (v, e) in metrics.items():
            mlflow.log_metric(f"VaR_{a}", v)
            mlflow.log_metric(f"ES_{a}", e)
        for a, (v, e) in metrics_indep.items():
            mlflow.log_metric(f"VaR_indep_{a}", v)
            mlflow.log_metric(f"ES_indep_{a}", e)
        mlflow.log_artifact(str(out_json))
        if save_samples:
            mlflow.log_artifact(str(risk_dir / "loss_samples.parquet"))
