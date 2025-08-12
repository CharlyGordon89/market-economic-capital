import numpy as np
import pandas as pd
from market_ec.models.gbm import fit_gbm_from_log_returns, TRADING_DAYS

def test_fit_gbm_from_log_returns_sanity():
    np.random.seed(0)
    mu_ann_true = 0.10
    sigma_ann_true = 0.20
    n = 10_000
    dt = 1.0 / TRADING_DAYS
    mu_d = mu_ann_true * dt
    sigma_d = sigma_ann_true * np.sqrt(dt)
    r = np.random.normal(mu_d, sigma_d, size=n)
    params = fit_gbm_from_log_returns(pd.Series(r))

    # 2 standard errors on annualized mean
    se_annual = (sigma_d / np.sqrt(n)) * TRADING_DAYS
    assert abs(params.mu_annual - mu_ann_true) < 2 * se_annual