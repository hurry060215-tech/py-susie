"""Inference helpers — ``susie_get_*`` / credible-set functions.

Faithful port of susieR's ``susie_utils.R``: credible-set extraction
(:func:`susie_get_cs`), posterior inclusion probabilities
(:func:`susie_get_pip`), posterior summaries, lfsr, the ELBO accessor,
prior-pruning utilities, and CS-correlation diagnostics.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm as _norm

__all__ = [
    "susie_get_objective",
    "susie_get_posterior_mean",
    "susie_get_posterior_sd",
    "susie_get_niter",
    "susie_get_prior_variance",
    "susie_get_residual_variance",
    "susie_get_lfsr",
    "susie_get_posterior_samples",
    "susie_get_cs",
    "susie_get_pip",
    "get_cs_correlation",
    "n_in_CS",
    "in_CS",
]


# ----------------------------------------------------------------------
def _alpha_of(res):
    """Return the L-by-p alpha matrix from a fit or pass through a matrix."""
    if isinstance(res, dict) and "alpha" in res:
        return np.asarray(res["alpha"], dtype=float)
    return np.asarray(res, dtype=float)


def susie_get_objective(res, last_only=True, warning_tol=1e-6):
    """Return the ELBO of a fitted susie model (``susie_get_objective``)."""
    elbo = np.asarray(res["elbo"], dtype=float)
    if not np.all(np.diff(elbo) >= -warning_tol):
        warnings.warn("Objective is decreasing")
    if last_only:
        return float(elbo[-1])
    return elbo


def susie_get_posterior_mean(res, prior_tol=1e-9):
    """Posterior mean of the regression coefficients (``susie_get_posterior_mean``)."""
    V = res.get("V")
    if V is not None and np.asarray(V).dtype.kind in "fiu":
        include = np.where(np.asarray(V) > prior_tol)[0]
    else:
        include = np.arange(res["alpha"].shape[0])
    scale = res.get("X_column_scale_factors")
    if scale is None:
        scale = np.ones(res["mu"].shape[1])
    if len(include) > 0:
        am = (res["alpha"] * res["mu"])[include]
        return np.sum(am, axis=0) / scale
    return np.zeros(res["mu"].shape[1])


def susie_get_posterior_sd(res, prior_tol=1e-9):
    """Posterior standard deviation of coefficients (``susie_get_posterior_sd``)."""
    V = res.get("V")
    if V is not None and np.asarray(V).dtype.kind in "fiu":
        include = np.where(np.asarray(V) > prior_tol)[0]
    else:
        include = np.arange(res["alpha"].shape[0])
    scale = res.get("X_column_scale_factors")
    if scale is None:
        scale = np.ones(res["mu"].shape[1])
    if len(include) > 0:
        var = (res["alpha"] * res["mu2"] - (res["alpha"] * res["mu"]) ** 2)[include]
        return np.sqrt(np.sum(var, axis=0)) / scale
    return np.zeros(res["mu"].shape[1])


def susie_get_niter(res):
    """Number of IBSS iterations performed."""
    return res["niter"]


def susie_get_prior_variance(res):
    """The (estimated or fixed) prior variances."""
    return res["V"]


def susie_get_residual_variance(res):
    """The (estimated or fixed) residual variance."""
    return res["sigma2"]


def susie_get_lfsr(res):
    """Local false sign rate per single-effect (``susie_get_lfsr``)."""
    mu = np.asarray(res["mu"], dtype=float)
    mu2 = np.asarray(res["mu2"], dtype=float)
    alpha = np.asarray(res["alpha"], dtype=float)
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(mu2 - mu ** 2)
    pos_prob = _norm.cdf(0.0, loc=mu, scale=sd)
    neg_prob = 1.0 - pos_prob
    return 1.0 - np.sum(alpha * np.maximum(pos_prob, neg_prob), axis=1)


def susie_get_posterior_samples(susie_fit, num_samples, random_state=None):
    """Draw posterior samples of effect sizes (``susie_get_posterior_samples``)."""
    rng = np.random.default_rng(random_state)
    V = susie_fit.get("V")
    if V is not None and np.asarray(V).dtype.kind in "fiu":
        include = np.where(np.asarray(V) > 1e-9)[0]
    else:
        include = np.arange(susie_fit["alpha"].shape[0])
    scale = susie_fit.get("X_column_scale_factors")
    if scale is None:
        scale = np.ones(susie_fit["mu"].shape[1])
    post_mean = susie_fit["mu"] / scale
    with np.errstate(invalid="ignore"):
        post_sd = np.sqrt(susie_fit["mu2"] - susie_fit["mu"] ** 2) / scale
    pip = susie_fit["alpha"]
    num_snps = pip.shape[1]
    b_samples = np.full((num_snps, num_samples), np.nan)
    gamma_samples = np.full((num_snps, num_samples), np.nan)
    for j in range(num_samples):
        b = np.zeros(num_snps)
        for l in include:
            idx = rng.choice(num_snps, p=pip[l])
            eff = rng.normal(post_mean[l, idx], post_sd[l, idx])
            b[idx] += eff
        b_samples[:, j] = b
        gamma_samples[:, j] = (b != 0).astype(float)
    return {"b": b_samples, "gamma": gamma_samples}


# ----------------------------------------------------------------------
# Credible-set machinery
# ----------------------------------------------------------------------
def n_in_CS_x(x, coverage=0.9):
    """Number of variables needed to reach ``coverage`` (``n_in_CS_x``)."""
    x = np.asarray(x, dtype=float)
    return int(np.sum(np.cumsum(np.sort(x)[::-1]) < coverage) + 1)


def in_CS_x(x, coverage=0.9):
    """Binary membership vector for one effect (``in_CS_x``)."""
    x = np.asarray(x, dtype=float)
    n = n_in_CS_x(x, coverage)
    # R's order(decreasing=TRUE) is a stable sort; argsort with kind=stable
    # on the negated array reproduces R's tie-breaking.
    o = np.argsort(-x, kind="stable")
    result = np.zeros(len(x), dtype=int)
    result[o[:n]] = 1
    return result


def in_CS(res, coverage=0.9):
    """L-by-p binary membership matrix (``in_CS``)."""
    alpha = _alpha_of(res)
    return np.array([in_CS_x(alpha[i], coverage) for i in range(alpha.shape[0])])


def n_in_CS(res, coverage=0.9):
    """Per-effect CS sizes (``n_in_CS``)."""
    alpha = _alpha_of(res)
    return np.array([n_in_CS_x(alpha[i], coverage) for i in range(alpha.shape[0])])


def _muffled_corr(x):
    """Pearson correlation of columns of x (``muffled_corr``)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x, rowvar=False)
    return np.atleast_2d(c)


def _muffled_cov2cor(x):
    """Convert a covariance matrix to a correlation matrix (``cov2cor``)."""
    x = np.asarray(x, dtype=float)
    d = np.sqrt(np.diag(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = x / np.outer(d, d)
    c[~np.isfinite(c)] = 0.0
    np.fill_diagonal(c, 1.0)
    return c


def _is_symmetric(x, tol=1e-10):
    x = np.asarray(x, dtype=float)
    return x.shape[0] == x.shape[1] and np.allclose(x, x.T, atol=tol, rtol=0)


def _get_purity(pos, X, Xcorr, squared=False, n=100, random_state=None):
    """Min / mean / median absolute correlation among CS variables (``get_purity``)."""
    pos = list(pos)
    if len(pos) == 1:
        return (1.0, 1.0, 1.0)
    if len(pos) > n:
        rng = np.random.default_rng(random_state)
        pos = list(rng.choice(pos, size=n, replace=False))
    if Xcorr is None:
        X_sub = np.asarray(X, dtype=float)[:, pos]
        corr = _muffled_corr(X_sub)
    else:
        corr = np.asarray(Xcorr, dtype=float)[np.ix_(pos, pos)]
    iu = np.triu_indices_from(corr, k=1)
    value = np.abs(corr[iu])
    if squared:
        value = value ** 2
    return (float(np.min(value)), float(np.sum(value) / len(value)),
            float(np.median(value)))


def susie_get_cs(res, X=None, Xcorr=None, coverage=0.95, min_abs_corr=0.5,
                 dedup=True, squared=False, check_symmetric=True,
                 n_purity=100, random_state=None):
    """Extract credible sets from a susie fit (``susie_get_cs``).

    Returns a dict with ``cs`` (list of 0-based index arrays), ``coverage``
    (claimed coverage per CS), ``purity`` and ``cs_index`` (when ``X`` or
    ``Xcorr`` is supplied, allowing purity filtering), and
    ``requested_coverage``.
    """
    if X is not None and Xcorr is not None:
        raise ValueError("Only one of X or Xcorr should be specified")
    if check_symmetric and Xcorr is not None and not _is_symmetric(Xcorr):
        warnings.warn("Xcorr is not symmetric; forcing Xcorr to be symmetric")
        Xcorr = (np.asarray(Xcorr) + np.asarray(Xcorr).T) / 2.0

    alpha = _alpha_of(res)
    null_index = res.get("null_index", 0) or 0
    include_idx = np.ones(alpha.shape[0], dtype=bool)
    V = res.get("V") if isinstance(res, dict) else None
    if V is not None and np.asarray(V).dtype.kind in "fiu":
        include_idx = np.asarray(V) > 1e-9

    status = in_CS(alpha, coverage)
    cs = [np.where(status[i] != 0)[0] for i in range(status.shape[0])]
    claimed_coverage = np.array([np.sum(alpha[i][cs[i]]) for i in range(len(cs))])
    include_idx = include_idx * np.array([len(c) > 0 for c in cs])

    if dedup:
        seen = set()
        keep_dup = np.zeros(len(cs), dtype=bool)
        for i, c in enumerate(cs):
            key = tuple(c.tolist())
            if key not in seen:
                seen.add(key)
                keep_dup[i] = True
        include_idx = include_idx * keep_dup
    include_idx = include_idx.astype(bool)

    if np.sum(include_idx) == 0:
        return {"cs": None, "coverage": None, "requested_coverage": coverage}

    sel = np.where(include_idx)[0]
    cs = [cs[i] for i in sel]
    claimed_coverage = claimed_coverage[sel]

    if Xcorr is None and X is None:
        names = [f"L{i + 1}" for i in sel]
        return {"cs": cs, "cs_names": names, "coverage": claimed_coverage,
                "requested_coverage": coverage}

    purity = []
    for i, c in enumerate(cs):
        if null_index > 0 and (null_index - 1) in c:
            purity.append((-9.0, -9.0, -9.0))
        else:
            purity.append(_get_purity(c, X, Xcorr, squared, n_purity,
                                      random_state))
    purity = np.array(purity)
    threshold = min_abs_corr ** 2 if squared else min_abs_corr
    is_pure = np.where(purity[:, 0] >= threshold)[0]
    if len(is_pure) == 0:
        return {"cs": None, "coverage": None, "requested_coverage": coverage}

    cs = [cs[i] for i in is_pure]
    purity = purity[is_pure]
    cs_index = sel[is_pure]
    # re-order by purity (descending), stable.
    ordering = np.argsort(-purity[:, 0], kind="stable")
    cs = [cs[i] for i in ordering]
    purity = purity[ordering]
    cs_index = cs_index[ordering]
    pcols = (["min.sq.corr", "mean.sq.corr", "median.sq.corr"] if squared
             else ["min.abs.corr", "mean.abs.corr", "median.abs.corr"])
    return {
        "cs": cs,
        "cs_names": [f"L{i + 1}" for i in cs_index],
        "purity": {pcols[k]: purity[:, k] for k in range(3)},
        "cs_index": cs_index,
        "coverage": claimed_coverage[is_pure][ordering],
        "requested_coverage": coverage,
    }


def susie_get_pip(res, prune_by_cs=False, prior_tol=1e-9):
    """Posterior inclusion probabilities for all variables (``susie_get_pip``)."""
    if isinstance(res, dict) and "alpha" in res:
        alpha = np.asarray(res["alpha"], dtype=float)
        null_index = res.get("null_index", 0) or 0
        if null_index > 0:
            alpha = np.delete(alpha, null_index - 1, axis=1)
        V = res.get("V")
        if V is not None and np.asarray(V).dtype.kind in "fiu":
            include = np.where(np.asarray(V) > prior_tol)[0]
        else:
            include = np.arange(alpha.shape[0])
        sets = res.get("sets")
        cs_index = sets.get("cs_index") if isinstance(sets, dict) else None
        if cs_index is not None and prune_by_cs:
            include = np.intersect1d(include, np.asarray(cs_index))
        elif cs_index is None and prune_by_cs:
            include = np.array([], dtype=int)
        if len(include) > 0:
            mat = alpha[include]
        else:
            mat = np.zeros((1, alpha.shape[1]))
    else:
        mat = np.asarray(res, dtype=float)
    return 1.0 - np.prod(1.0 - mat, axis=0)


def get_cs_correlation(model, X=None, Xcorr=None, max=False):
    """Correlation between the lead variables of distinct CSs (``get_cs_correlation``)."""
    sets = model.get("sets")
    cs = sets.get("cs") if isinstance(sets, dict) else None
    if cs is None or len(cs) == 1:
        return np.nan
    if X is not None and Xcorr is not None:
        raise ValueError("Only one of X or Xcorr should be specified")
    if X is None and Xcorr is None:
        raise ValueError("One of X or Xcorr must be specified")
    if Xcorr is not None and not _is_symmetric(Xcorr):
        warnings.warn("Xcorr is not symmetric; forcing Xcorr to be symmetric")
        Xcorr = (np.asarray(Xcorr) + np.asarray(Xcorr).T) / 2.0
    pip = np.asarray(model["pip"], dtype=float)
    max_pip_idx = np.array([c[int(np.argmax(pip[c]))] for c in cs])
    if Xcorr is None:
        cs_corr = _muffled_corr(np.asarray(X, dtype=float)[:, max_pip_idx])
    else:
        cs_corr = np.asarray(Xcorr, dtype=float)[np.ix_(max_pip_idx, max_pip_idx)]
    if max:
        iu = np.triu_indices_from(cs_corr, k=1)
        return float(np.max(np.abs(cs_corr[iu])))
    return cs_corr
