"""Univariate regression and sufficient-statistic helpers.

Ports susieR's ``univariate_regression.R`` (per-column simple linear
regression for z-scores) and ``compute_ss.R`` (sufficient statistics
``X'X``, ``X'y``, ``y'y`` for :func:`pysusie.susie_suff_stat`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["univariate_regression", "calc_z", "compute_suff_stat", "compute_ss"]


def _colsds(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return np.zeros(X.shape[1])
    return X.std(axis=0, ddof=1)


def univariate_regression(X, y, Z=None, center=True, scale=False,
                          return_residuals=False):
    """Per-column simple linear regression of ``y`` on each column of ``X``.

    Faithful port of ``univariate_regression``.  Returns a dict with
    ``betahat`` and ``sebetahat`` (and ``residuals`` when ``Z`` is given
    and ``return_residuals=True``).
    """
    X = np.array(X, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True).ravel()
    na = np.where(np.isnan(y))[0]
    if len(na):
        keep = ~np.isnan(y)
        X = X[keep]
        y = y[keep]
    if center:
        y = y - np.mean(y)
        X = X - X.mean(axis=0)
    if scale:
        sd = _colsds(X)
        sd[sd == 0] = 1.0
        X = X / sd
    X = np.where(np.isnan(X), 0.0, X)
    if Z is not None:
        Z = np.array(Z, dtype=float, copy=True)
        if center:
            Z = Z - Z.mean(axis=0)
            if scale:
                zsd = _colsds(Z)
                zsd[zsd == 0] = 1.0
                Z = Z / zsd
        beta_z, *_ = np.linalg.lstsq(Z, y, rcond=None)
        y = y - Z @ beta_z

    n, p = X.shape
    betahat = np.zeros(p)
    sebetahat = np.zeros(p)
    for i in range(p):
        D = np.column_stack([np.ones(n), X[:, i]])
        coef, *_ = np.linalg.lstsq(D, y, rcond=None)
        resid = y - D @ coef
        # calc_stderr: sqrt(diag(SSE/(n-2) * (X'X)^-1))
        if n > 2:
            sse = np.sum(resid ** 2)
            try:
                xtx_inv = np.linalg.inv(D.T @ D)
                se = np.sqrt(np.diag(sse / (n - 2) * xtx_inv))
            except np.linalg.LinAlgError:
                se = np.array([0.0, 0.0])
        else:
            se = np.array([0.0, 0.0])
        betahat[i] = coef[1]
        sebetahat[i] = se[1] if len(se) == 2 else 0.0
    out = {"betahat": betahat, "sebetahat": sebetahat}
    if return_residuals and Z is not None:
        out["residuals"] = y
    return out


def calc_z(X, Y, center=False, scale=False):
    """Univariate association z-scores (t-statistics) of ``Y`` on each X column.

    Faithful port of ``calc_z``.  Accepts a vector or a matrix ``Y``.
    """
    Y = np.asarray(Y, dtype=float)

    def _z(y):
        out = univariate_regression(X, y, center=center, scale=scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            return out["betahat"] / out["sebetahat"]

    if Y.ndim == 1:
        return _z(Y)
    return np.column_stack([_z(Y[:, i]) for i in range(Y.shape[1])])


def compute_suff_stat(X, y, standardize=False):
    """Sufficient statistics ``X'X``, ``X'y``, ``y'y`` for ``susie_suff_stat``.

    Faithful port of ``compute_suff_stat`` (a.k.a. ``compute_ss``).  X and
    y are centered (and optionally X standardized) before the cross
    products are computed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    y_mean = float(np.mean(y))
    yc = y - y_mean
    n = X.shape[0]
    mu = X.mean(axis=0)
    s = _colsds(X)
    Xty = yc @ X
    XtX = X.T @ X
    XtX = XtX - n * np.outer(mu, mu)
    if standardize:
        XtX = XtX / s
        XtX = XtX.T
        XtX = XtX / s
        Xty = Xty / s
    yty = float(np.sum(yc ** 2))
    return {"XtX": XtX, "Xty": Xty, "yty": yty, "n": n,
            "y_mean": y_mean, "X_colmeans": mu}


# Synonym kept for parity with R (deprecated alias).
compute_ss = compute_suff_stat
