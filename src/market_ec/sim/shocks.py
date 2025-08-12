from __future__ import annotations
import numpy as np
from numpy.linalg import cholesky, eigh
from scipy.stats import t as t_dist, norm


def _to_corr_pd(C: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive-definite *correlation* matrix
    via eigenvalue clipping, then renormalize to unit diagonal.
    """
    C = 0.5 * (C + C.T)
    w, V = eigh(C)
    w_clipped = np.maximum(w, eps)
    C_pd = (V @ (w_clipped[:, None] * V.T))
    d = np.sqrt(np.clip(np.diag(C_pd), eps, None))
    C_corr = C_pd / np.outer(d, d)
    np.fill_diagonal(C_corr, 1.0)
    return C_corr


def _chol_with_jitter(C: np.ndarray, base_jitter: float = 1e-12, max_jitter: float = 1e-3) -> np.ndarray:
    """
    Try Cholesky; if it fails, add increasing diagonal jitter until it succeeds.
    """
    jitter = base_jitter
    while True:
        try:
            return cholesky(C)
        except np.linalg.LinAlgError:
            if jitter > max_jitter:
                raise
            C = C.copy()
            C[np.diag_indices_from(C)] += jitter
            jitter *= 10.0


def sample_gaussian_copula_normals(
    corr: np.ndarray,
    n_sims: int,
    horizon_days: int,
    n_assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return Z ~ N(0,1) with Gaussian copula dependence, shape (n_sims, horizon_days, n_assets).
    """
    C = _to_corr_pd(np.asarray(corr, dtype=float))
    L = _chol_with_jitter(C)
    Z = rng.standard_normal(size=(n_sims, horizon_days, n_assets))
    Z_corr = Z @ L.T
    return Z_corr


def sample_t_copula_normals(
    corr: np.ndarray,
    df: int,
    n_sims: int,
    horizon_days: int,
    n_assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    t-copula: sample multivariate t → map to uniforms via CDF → map to standard normal.
    Output has standard-normal marginals but t-copula dependence.
    """
    C = _to_corr_pd(np.asarray(corr, dtype=float))
    L = _chol_with_jitter(C)
    G = rng.standard_normal(size=(n_sims, horizon_days, n_assets)) @ L.T
    W = rng.chisquare(df, size=(n_sims, horizon_days, 1)) / df
    T = G / np.sqrt(W)
    U = t_dist.cdf(T, df=df)
    Z = norm.ppf(U)
    return Z
