from __future__ import annotations
from pathlib import Path
import argparse
from datetime import date, datetime  # NEW
import pandas as pd
import mlflow

from market_ec.utils.config import load_config
from market_ec.utils.logging import get_logger
from market_ec.utils.mlflow_utils import dataframe_hash, start_run
from market_ec.data.price_loader import PriceRequest, download_prices
from market_ec.data.validators import validate_prices
from market_ec.data.returns import compute_log_returns


def _get_data_window(data_cfg: dict) -> tuple[str | None, str | None]:
    start = data_cfg.get("start", data_cfg.get("start_date"))
    end = data_cfg.get("end", data_cfg.get("end_date"))

    def to_str(x):
        if isinstance(x, (date, datetime)):
            return x.isoformat()
        return x
    return to_str(start), to_str(end)


def build_dataset(cfg_path: str) -> None:
    logger = get_logger()
    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]

    out_dir = Path(cfg["artifacts"]["data_dir"])  # e.g., data/
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = _get_data_window(data_cfg)

    req = PriceRequest(
        tickers=list(data_cfg["tickers"]),
        start=start,
        end=end,
        interval=data_cfg.get("interval", "1d"),
        adj_close_only=True,
        source=data_cfg.get("source", "auto"),  # ← multi-source (auto|yahoo|stooq)
    )

    with start_run(experiment=cfg["mlflow"]["experiment"], run_name="build_dataset"):
        prices = download_prices(req)
        validate_prices(prices)

        present = sorted(prices["ticker"].unique().tolist())
        requested = list(req.tickers)
        missing = [t for t in requested if t not in present]
        if missing:
            logger.warning("Missing after multi-source fetch (skipped): %s", missing)
            mlflow.log_param("missing_tickers", ",".join(missing))

        logrets = compute_log_returns(prices, use_adj=True)

        # Save artifacts
        prices_path = out_dir / "prices.parquet"
        rets_path = out_dir / "log_returns.parquet"
        prices.to_parquet(prices_path, index=False)
        logrets.to_parquet(rets_path, index=False)

        # Track in MLflow
        mlflow.log_params({
            "tickers_requested": ",".join(requested),
            "tickers_present": ",".join(present),
            "start": start or "auto-15y",
            "end": end or "latest",
            "interval": req.interval,
            "source": req.source,
        })
        mlflow.log_artifact(str(prices_path))
        mlflow.log_artifact(str(rets_path))
        mlflow.log_metric("n_prices", int(len(prices)))
        mlflow.log_metric("n_log_returns", int(len(logrets)))
        mlflow.set_tags({"data_hash": dataframe_hash(prices)})

        logger.info("Saved prices to %s and returns to %s", prices_path, rets_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline dataset for market-ec")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    build_dataset(args.config)


if __name__ == "__main__":
    main()
