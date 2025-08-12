from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import norm, t

@dataclass(frozen=True)
class GaussianCopulaParams:
    corr: np.ndarray

    def to_dict(self):
        return {"corr": self.corr.tolist()}

@dataclass(frozen=True)
class TCopulaParams:
    corr: np.ndarray
    df: int

    def to_dict(self):
        return {"corr": self.corr.tolist(), "df": self.df}

def _pseudo_obs(X: np.ndarray) -> np.ndarray:
    """rank transform to (0,1) using (rank / (n+1))"""
    n = X.shape[0]
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    U = ranks / (n + 1.0)
    return U

def fit_gaussian_copula(returns_df: pd.DataFrame) -> GaussianCopulaParams:
    """
    Fit Gaussian copula via normal-scores correlation:
      corr = Corr( Phi^{-1}(U_i), Phi^{-1}(U_j) )
    """
    X = returns_df.dropna().values
    U = _pseudo_obs(X)
    Z = norm.ppf(U)
    corr = np.corrcoef(Z, rowvar=False)
    return GaussianCopulaParams(corr=corr)

def fit_t_copula(returns_df: pd.DataFrame, df: int = 6) -> TCopulaParams:
    """
    Simple t-copula fit: use normal-score correlation and fixed df (configurable).
    (Full MLE for df can be added later.)
    """
    X = returns_df.dropna().values
    U = _pseudo_obs(X)
    Z = norm.ppf(U)
    corr = np.corrcoef(Z, rowvar=False)
    return TCopulaParams(corr=corr, df=int(df))
