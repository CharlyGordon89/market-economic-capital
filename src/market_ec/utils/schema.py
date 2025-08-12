from __future__ import annotations
from pydantic import BaseModel, conlist
from typing import Dict, List, Optional, Union
from datetime import date, datetime

class MLflowCfg(BaseModel):
    experiment: str

class ArtifactsCfg(BaseModel):
    data_dir: str
    models_dir: str
    risk_dir: str
    dashboard_dir: str

class DataCfg(BaseModel):
    tickers: conlist(str, min_length=1)
    start: Optional[Union[str, date, datetime]] = None
    end: Optional[Union[str, date, datetime]] = None
    interval: str = "1d"
    source: str = "auto"  # auto | yahoo | stooq

class ModelsCfg(BaseModel):
    gbm_tickers: List[str] = []
    ou_tickers: List[str] = []
    garch_tickers: List[str] = []

class DependenceCfg(BaseModel):
    method: str
    t_df: Optional[int] = 6
    tickers: conlist(str, min_length=2)

class PortfolioCfg(BaseModel):
    weights: Dict[str, float]

class RiskCfg(BaseModel):
    horizon_days: int = 252
    n_sims: int = 20000
    alphas: conlist(float, min_length=1)
    save_samples: bool = False

class StressCfg(BaseModel):
    historical: List[Dict] = []
    parametric: List[Dict] = []

class BacktestCfg(BaseModel):
    var_method: str = "historical"
    window_days: int = 250
    alpha: float = 0.99
    start: Optional[Union[str, date, datetime]] = None
    end: Optional[Union[str, date, datetime]] = None

class RunCfg(BaseModel):
    experiment_name: Optional[str] = None
    random_seed: Optional[int] = 42

class RootCfg(BaseModel):
    project: str
    mlflow: MLflowCfg
    artifacts: ArtifactsCfg
    data: DataCfg
    models: ModelsCfg
    dependence: DependenceCfg
    portfolio: PortfolioCfg
    risk: RiskCfg
    stress: StressCfg
    backtest: BacktestCfg
    run: Optional[RunCfg] = None
