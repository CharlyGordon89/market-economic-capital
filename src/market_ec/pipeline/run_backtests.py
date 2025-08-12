from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd
import mlflow

from market_ec.utils.config import load_config
from market_ec.utils.logging import get_logger
from market_ec.utils.mlflow_utils import start_run
from market_ec.risk.backtest import portfolio_returns_from_prices, rolling_hs_var, kupiec_pof_test, christoffersen_independence_test

def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run 1-day VaR backtests (historical method)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logger = get_logger()
    cfg = load_config(args.config)

    data_dir = Path(cfg["artifacts"]["data_dir"])
    risk_dir = Path(cfg["artifacts"]["risk_dir"])
    risk_dir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(data_dir / "prices.parquet")
    weights = cfg["portfolio"]["weights"]

    bt = cfg["backtest"]
    alpha = float(bt["alpha"])
    window = int(bt["window_days"])
    start = pd.to_datetime(bt.get("start"))
    end = pd.to_datetime(bt.get("end")) if bt.get("end") else None

    port_ret = portfolio_returns_from_prices(prices, weights)
    if start is not None:
        port_ret = port_ret.loc[port_ret.index >= start]
    if end is not None:
        port_ret = port_ret.loc[port_ret.index <= end]

    var_series = rolling_hs_var(port_ret, window=window, alpha=alpha)
    aligned = port_ret.to_frame("ret").join(var_series.rename("VaR"), how="inner")
    aligned = aligned.dropna()
    # breach if realized loss > VaR threshold
    aligned["loss"] = -aligned["ret"]
    aligned["breach"] = (aligned["loss"] > aligned["VaR"]).astype(int)

    pof = kupiec_pof_test(aligned["breach"].values, alpha=alpha)
    ind = christoffersen_independence_test(aligned["breach"].values)

    out = {
        "alpha": alpha,
        "window": window,
        "n_obs": int(len(aligned)),
        "POF": pof,
        "IND": ind,
        "breach_rate": float(aligned["breach"].mean()),
    }

    out_path = risk_dir / "backtest_results.json"
    _save_json(out, out_path)

    with start_run(cfg["mlflow"]["experiment"], run_name="backtest_var", tags={"sprint": "4"}):
        mlflow.log_metric("bt_breach_rate", out["breach_rate"])
        mlflow.log_metric("bt_pof_lr", out["POF"]["LR_POF"])
        mlflow.log_metric("bt_pof_p", out["POF"]["p_value"])
        mlflow.log_metric("bt_ind_lr", out["IND"]["LR_IND"])
        mlflow.log_metric("bt_ind_p", out["IND"]["p_value"])
        mlflow.log_artifact(str(out_path))

    logger.info("Backtest results saved to %s", out_path)

if __name__ == "__main__":
    main()
