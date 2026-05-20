"""Low-level linear-algebra helpers ported from susieR.

These mirror susieR's ``sparse_multiplication.R`` and ``susie_utils.R``
column-statistics utilities.  All matrices are dense ``numpy`` arrays; the
"standardized X" trick (centering / scaling without materialising a copy
of X) is reproduced exactly so that numerical results match R.

A susie matrix-context bundles the original (un-standardized) matrix ``X``
together with its column means (``cm``), column standard deviations
(``csd``) and the per-column sums of squares of the standardized matrix
(``d``).  These are the three attributes susieR attaches to ``X`` as
``scaled:center``, ``scaled:scale`` and ``d``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "compute_colstats",
    "compute_Xb",
    "compute_Xty",
    "compute_MXt",
]


def compute_colstats(X, center=True, scale=True):
    """Column means, column sds and standardized sum-of-squares.

    Reproduces ``compute_colstats`` from ``susie_utils.R``.  Returns a
    dict with keys ``cm`` (p-vector of column means or zeros), ``csd``
    (p-vector of column sds or ones; zero-variance columns get sd 1) and
    ``d`` (p-vector of ``colSums(X.standardized^2)``).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if center:
        cm = X.mean(axis=0)
    else:
        cm = np.zeros(p)
    if scale:
        # R uses the unbiased (n-1) sd via matrixStats::colSds.
        csd = _colsds(X)
        csd[csd == 0] = 1.0
    else:
        csd = np.ones(p)
    # d = colSums(X.standardized^2). R computes it as
    #   d = n*colMeans(X)^2 + (n-1)*colSds(X)^2
    #   d = (d - n*cm^2)/csd^2
    colmeans = X.mean(axis=0)
    d = n * colmeans ** 2 + (n - 1) * _colsds(X) ** 2
    d = (d - n * cm ** 2) / csd ** 2
    return {"cm": cm, "csd": csd, "d": d}


def _colsds(X):
    """Unbiased column standard deviations (matches matrixStats::colSds)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return np.zeros(X.shape[1])
    return X.std(axis=0, ddof=1)


def compute_Xb(ctx, b):
    """Compute ``standardized.X %*% b`` using the sparse-multiplication trick.

    ``ctx`` is a dict with keys ``X``, ``cm``, ``csd``.  Mirrors
    ``compute_Xb`` in ``sparse_multiplication.R``.
    """
    X = ctx["X"]
    cm = ctx["cm"]
    csd = ctx["csd"]
    b = np.asarray(b, dtype=float)
    scaled_Xb = X @ (b / csd)
    Xb = scaled_Xb - np.sum(cm * b / csd)
    return np.asarray(Xb, dtype=float)


def compute_Xty(ctx, y):
    """Compute ``t(standardized.X) %*% y`` using the sparse trick.

    Mirrors ``compute_Xty`` in ``sparse_multiplication.R``.
    """
    X = ctx["X"]
    cm = ctx["cm"]
    csd = ctx["csd"]
    y = np.asarray(y, dtype=float)
    ytX = y @ X  # length-p
    scaled_Xty = ytX / csd
    centered = scaled_Xty - cm / csd * np.sum(y)
    return np.asarray(centered, dtype=float)


def compute_MXt(ctx, M):
    """Compute ``M %*% t(standardized.X)`` for an L-by-p matrix M.

    Mirrors ``compute_MXt`` in ``sparse_multiplication.R``; returns an
    L-by-n matrix.
    """
    X = ctx["X"]
    cm = ctx["cm"]
    csd = ctx["csd"]
    M = np.atleast_2d(np.asarray(M, dtype=float))
    # t(X %*% (t(M)/csd)) - drop(M %*% (cm/csd))
    res = (X @ (M.T / csd[:, None])).T - (M @ (cm / csd))[:, None]
    return np.asarray(res, dtype=float)
