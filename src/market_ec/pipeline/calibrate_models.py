from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mlflow
from statsmodels.stats.diagnostic import acorr_ljungbox

from market_ec.utils.config import load_config
from market_ec.utils.logging import get_logger
from market_ec.utils.mlflow_utils import start_run
from market_ec.models.gbm import fit_gbm_from_log_returns
from market_ec.models.ou import fit_ou_from_log_price
from market_ec.models.garch import fit_garch11
from market_ec.dependence.copulas import fit_gaussian_copula, fit_t_copula


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _wide_returns(logret_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide matrix: index=date, columns=ticker."""
    w = logret_df.pivot(index="date", columns="ticker", values="log_ret").sort_index()
    return w


def run_calibration(cfg_path: str) -> None:
    logger = get_logger()
    cfg = load_config(cfg_path)

    data_dir = Path(cfg["artifacts"]["data_dir"])
    models_dir = Path(cfg["artifacts"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Sprint 1 artifacts
    prices = pd.read_parquet(data_dir / "prices.parquet")
    logrets = pd.read_parquet(data_dir / "log_returns.parquet")
    w_rets = _wide_returns(logrets)

    available_rets = set(w_rets.columns)
    available_prices = set(prices["ticker"].unique())

    def _filter(name: str, requested: list[str], on: str = "rets") -> tuple[list[str], list[str]]:
        base = available_rets if on == "rets" else available_prices
        ok = [t for t in requested if t in base]
        missing = [t for t in requested if t not in base]
        if missing:
            logger.warning("%s: skipping missing tickers %s (not in %s dataset)", name, missing, on)
        return ok, missing

    gbm_req = list(cfg["models"]["gbm_tickers"])
    ou_req = list(cfg["models"]["ou_tickers"])
    garch_req = list(cfg["models"]["garch_tickers"])

    # GBM/GARCH work on returns; OU fits on log-prices
    gbm_tickers, gbm_missing = _filter("GBM", gbm_req, on="rets")
    garch_tickers, garch_missing = _filter("GARCH", garch_req, on="rets")
    ou_tickers_prices, ou_missing_prices = _filter("OU", ou_req, on="prices")

    # Dependence (copula) works on returns
    dep_cfg = cfg["dependence"]
    dep_req = list(dep_cfg["tickers"])
    dep_tickers, dep_missing = _filter("COPULA", dep_req, on="rets")
    if len(dep_tickers) < 2:
        raise ValueError(
            f"COPULA: need at least 2 tickers present in returns; available={sorted(available_rets)}"
        )

    with start_run(cfg["mlflow"]["experiment"], run_name="calibrate_models", tags={"sprint": "2"}):
        # Log what we’re actually calibrating
        mlflow.log_params({
            "gbm_requested": ",".join(gbm_req),
            "gbm_used": ",".join(gbm_tickers),
            "garch_requested": ",".join(garch_req),
            "garch_used": ",".join(garch_tickers),
            "ou_requested": ",".join(ou_req),
            "ou_used": ",".join(ou_tickers_prices),
            "copula_requested": ",".join(dep_req),
            "copula_used": ",".join(dep_tickers),
        })

        # ---------- GBM ----------
        gbm_params: dict[str, dict] = {}
        for t in gbm_tickers:
            r = w_rets[t].dropna()
            if r.empty:
                logger.warning("GBM: %s has no returns after dropna; skipping", t)
                continue
            p = fit_gbm_from_log_returns(r)
            gbm_params[t] = p.to_dict()
        _save_json(gbm_params, models_dir / "gbm_params.json")
        mlflow.log_artifact(str(models_dir / "gbm_params.json"))

        # ---------- OU (on log-price level) ----------
        ou_params: dict[str, dict] = {}
        for t in ou_tickers_prices:
            px = (
                prices[prices["ticker"] == t]
                .sort_values("date")["adj_close"]
                .astype(float)
            )
            if px.isna().all() or len(px) < 5:
                logger.warning("OU: %s insufficient price history; skipping", t)
                continue
            x = np.log(px)
            op = fit_ou_from_log_price(x)
            ou_params[t] = op.to_dict()
        _save_json(ou_params, models_dir / "ou_params.json")
        mlflow.log_artifact(str(models_dir / "ou_params.json"))

        # ---------- GARCH(1,1) ----------
        garch_params: dict[str, dict] = {}
        garch_diag_rows: list[dict] = []
        for t in garch_tickers:
            r = w_rets[t].dropna()
            if len(r) < 200:
                logger.warning("GARCH: %s has <200 obs; skipping to avoid unstable fit", t)
                continue
            gp, std_resid = fit_garch11(r)
            garch_params[t] = gp.to_dict()

            # Ljung–Box on standardized residuals (lags 10, 20)
            sr = std_resid.dropna()
            lb10 = acorr_ljungbox(sr, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
            lb20 = acorr_ljungbox(sr, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
            garch_diag_rows.append({
                "ticker": t,
                "lb_pvalue_lag10": float(lb10),
                "lb_pvalue_lag20": float(lb20),
                "persistence": garch_params[t]["persistence"],
            })

        _save_json(garch_params, models_dir / "garch11_params.json")
        pd.DataFrame(garch_diag_rows).to_parquet(models_dir / "garch11_diagnostics.parquet", index=False)
        mlflow.log_artifact(str(models_dir / "garch11_params.json"))
        mlflow.log_artifact(str(models_dir / "garch11_diagnostics.parquet"))

        # ---------- Dependence (Copula) ----------
        dep_matrix = w_rets[dep_tickers].dropna()
        if dep_cfg["method"] == "gaussian":
            cp = fit_gaussian_copula(dep_matrix)
            dep_path = models_dir / "copula_gaussian.json"
            _save_json(cp.to_dict(), dep_path)
            mlflow.log_param("copula", "gaussian")
            mlflow.log_artifact(str(dep_path))
        elif dep_cfg["method"] == "t":
            df_t = int(dep_cfg.get("t_df", 6))
            cp = fit_t_copula(dep_matrix, df=df_t)
            dep_path = models_dir / "copula_t.json"
            _save_json(cp.to_dict(), dep_path)
            mlflow.log_param("copula", "t")
            mlflow.log_param("t_df", df_t)
            mlflow.log_artifact(str(dep_path))
        else:
            raise ValueError(f"Unknown dependence method: {dep_cfg['method']}")

        # ---------- Metrics & final logging ----------
        mlflow.log_param("gbm_n", len(gbm_params))
        mlflow.log_param("ou_n", len(ou_params))
        mlflow.log_param("garch_n", len(garch_params))
        mlflow.log_param("gbm_missing_n", len(gbm_missing))
        mlflow.log_param("ou_missing_n", len(ou_missing_prices))
        mlflow.log_param("garch_missing_n", len(garch_missing))
        mlflow.log_param("copula_missing_n", len(dep_missing))
        mlflow.log_metric("returns_rows", int(w_rets.shape[0]))
        mlflow.log_metric("returns_cols", int(w_rets.shape[1]))

        logger.info("Saved model params to %s", models_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate GBM/OU/GARCH and copula")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_calibration(args.config)


if __name__ == "__main__":
    main()
