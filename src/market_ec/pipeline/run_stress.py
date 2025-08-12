from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import mlflow

from market_ec.utils.config import load_config
from market_ec.utils.logging import get_logger
from market_ec.utils.mlflow_utils import start_run
from market_ec.stress.stress_scenarios import run_parametric_stress, run_historical_stress, ParametricStress
from market_ec.sim.engine import _load_copula

def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run historical and parametric stress tests")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logger = get_logger()
    cfg = load_config(args.config)

    data_dir = Path(cfg["artifacts"]["data_dir"])
    models_dir = Path(cfg["artifacts"]["models_dir"])
    risk_dir = Path(cfg["artifacts"]["risk_dir"])
    risk_dir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(data_dir / "prices.parquet")
    dep = cfg["dependence"]
    dep_tickers = list(dep["tickers"])
    weights = cfg["portfolio"]["weights"]
    T = int(cfg["risk"]["horizon_days"])
    n_sims = int(cfg["risk"]["n_sims"])
    seed = int(cfg.get("run", {}).get("random_seed", 42))

    # load copula params
    copula_kind, copula_params = _load_copula(models_dir)

    # parametric scenarios
    stress_cfgs = [ParametricStress(**s) for s in cfg["stress"].get("parametric", [])]
    # run
    with start_run(cfg["mlflow"]["experiment"], run_name="stress_tests", tags={"sprint": "4"}):
        # parametric
        parametric = run_parametric_stress(
            prices=prices,
            models_dir=models_dir,
            risk_dir=risk_dir,
            dep_tickers=dep_tickers,
            weights_cfg=weights,
            horizon_days=T,
            n_sims=n_sims,
            copula_kind=copula_kind,
            copula_params=copula_params,
            stresses=stress_cfgs,
            seed=seed,
        )
        for r in parametric:
            for k, v in r.items():
                if k != "name":
                    mlflow.log_metric(f"stress_{r['name']}_{k}", v)

        # historical
        hist_out_path = risk_dir / "stress_historical.json"
        historical = run_historical_stress(
            prices=prices,
            weights_cfg=weights,
            windows=cfg["stress"].get("historical", []),
            out_path=hist_out_path,
        )
        mlflow.log_artifact(str(hist_out_path))

    logger.info("Stress results saved to %s", risk_dir)

if __name__ == "__main__":
    main()
