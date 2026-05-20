"""``susie`` — IBSS fine-mapping with individual-level data.

Faithful port of susieR's ``susie.R`` (the ``susie`` function), the IBSS
coordinate-ascent loop in ``update_each_effect.R``, and the prediction /
coefficient helpers in ``predict.susie.R``.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._linalg import compute_colstats, compute_Xb
from ._model import (
    SusieFit, init_setup, init_finalize, get_objective,
    estimate_residual_variance as _estimate_residual_variance,
    SER_posterior_e_loglik,
)
from .ser import single_effect_regression
from .getters import susie_get_cs, susie_get_pip
from .univariate import calc_z

__all__ = ["susie", "susie_init_coef", "coef_susie", "predict_susie",
           "susie_prune_single_effects"]


# ----------------------------------------------------------------------
def susie_init_coef(coef_index, coef_value, p):
    """Initialise a susie object from regression coefficients (``susie_init_coef``)."""
    coef_index = np.asarray(coef_index)
    coef_value = np.asarray(coef_value, dtype=float)
    L = len(coef_index)
    if L <= 0:
        raise ValueError("Need at least one non-zero effect")
    if not np.all(coef_value != 0):
        raise ValueError("Input coef_value must be non-zero for all elements")
    if L != len(coef_value):
        raise ValueError("coef_index and coef_value must be of the same length")
    if np.max(coef_index) >= p:
        raise ValueError("Input coef_index exceeds the boundary of p")
    alpha = np.zeros((L, p))
    mu = np.zeros((L, p))
    for i in range(L):
        alpha[i, coef_index[i]] = 1.0
        mu[i, coef_index[i]] = coef_value[i]
    return SusieFit(alpha=alpha, mu=mu, mu2=mu * mu)


def susie_prune_single_effects(s, L=0, V=None):
    """Prune or expand single effects to L (``susie_prune_single_effects``)."""
    num_effects = s["alpha"].shape[0]
    if L == 0:
        if s.get("V") is not None:
            L = int(np.sum(np.asarray(s["V"]) > 0))
        else:
            L = num_effects
    if L == num_effects:
        s["sets"] = None
        return s
    sets = s.get("sets")
    cs_index = sets.get("cs_index") if isinstance(sets, dict) else None
    if cs_index is not None:
        rank = list(cs_index) + [i for i in range(num_effects)
                                 if i not in set(cs_index)]
    else:
        rank = list(range(num_effects))
    rank = np.asarray(rank, dtype=int)
    if L > num_effects:
        p = s["alpha"].shape[1]
        s["alpha"] = np.vstack([s["alpha"][rank],
                                np.full((L - num_effects, p), 1.0 / p)])
        for n in ("mu", "mu2", "lbf_variable"):
            if s.get(n) is not None:
                s[n] = np.vstack([np.asarray(s[n])[rank],
                                  np.zeros((L - num_effects, p))])
        for n in ("KL", "lbf"):
            if s.get(n) is not None:
                s[n] = np.append(np.asarray(s[n])[rank],
                                 np.full(L - num_effects, np.nan))
        if V is not None:
            V = np.atleast_1d(np.asarray(V, dtype=float))
            if len(V) > 1:
                V[:num_effects] = np.asarray(s["V"])[rank]
            else:
                V = np.full(L, V[0])
        s["V"] = V
    else:
        # L < num_effects: keep the first L by rank.
        keep = rank[:L]
        for n in ("alpha", "mu", "mu2", "lbf_variable"):
            if s.get(n) is not None:
                s[n] = np.asarray(s[n])[keep]
        for n in ("KL", "lbf"):
            if s.get(n) is not None:
                s[n] = np.asarray(s[n])[keep]
        if s.get("V") is not None:
            s["V"] = np.asarray(s["V"])[keep]
    s["sets"] = None
    return s


# ----------------------------------------------------------------------
def _update_each_effect(ctx, y, s, estimate_prior_variance,
                        estimate_prior_method, check_null_threshold):
    """One IBSS sweep over the L single effects (``update_each_effect``)."""
    optimize_V = estimate_prior_method if estimate_prior_variance else "none"
    L = s["alpha"].shape[0]
    for l in range(L):
        s["Xr"] = s["Xr"] - compute_Xb(ctx, s["alpha"][l] * s["mu"][l])
        R = np.asarray(y, dtype=float) - s["Xr"]
        res = single_effect_regression(R, ctx, s["V"][l], s["sigma2"],
                                       s["pi"], optimize_V,
                                       check_null_threshold)
        s["mu"][l] = res["mu"]
        s["alpha"][l] = res["alpha"]
        s["mu2"][l] = res["mu2"]
        s["V"][l] = res["V"]
        s["lbf"][l] = res["lbf_model"]
        s["lbf_variable"][l] = res["lbf"]
        s["KL"][l] = (-res["loglik"]
                      + SER_posterior_e_loglik(ctx, R, s["sigma2"],
                                               res["alpha"] * res["mu"],
                                               res["alpha"] * res["mu2"]))
        s["Xr"] = s["Xr"] + compute_Xb(ctx, s["alpha"][l] * s["mu"][l])
    return s


# ----------------------------------------------------------------------
def susie(X, y, L=10, scaled_prior_variance=0.2, residual_variance=None,
          prior_weights=None, null_weight=0.0, standardize=True,
          intercept=True, estimate_residual_variance=True,
          estimate_prior_variance=True, estimate_prior_method="optim",
          check_null_threshold=0.0, prior_tol=1e-9,
          residual_variance_upperbound=np.inf, s_init=None,
          coverage=0.95, min_abs_corr=0.5, compute_univariate_zscore=False,
          na_rm=False, max_iter=100, tol=1e-3, verbose=False,
          track_fit=False, residual_variance_lowerbound=None, refine=False,
          n_purity=100):
    """Sum of Single Effects (SuSiE) regression with individual-level data.

    Faithful Python port of susieR's ``susie``.  Fits ``y = mu + Xb + e``
    with the SuSiE prior (``b`` is a sum of L single effects, each having
    exactly one non-zero element) via the IBSS algorithm.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Covariate matrix.
    y : array-like, shape (n,)
        Response vector.
    L : int
        Maximum number of non-zero effects.
    scaled_prior_variance : float or array-like
        Prior variance divided by ``var(y)``.
    residual_variance : float, optional
        Initial residual variance (defaults to ``var(y)``).
    prior_weights : array-like, optional
        Per-variable prior inclusion probabilities.
    null_weight : float
        Prior probability of no effect (adds a null column).
    standardize, intercept : bool
        Whether to standardize columns of X and fit an intercept.
    estimate_residual_variance, estimate_prior_variance : bool
        Whether to estimate these hyperparameters during fitting.
    estimate_prior_method : {"optim", "EM", "simple"}
        Method for prior-variance estimation.
    coverage : float
        Coverage of the credible sets.
    min_abs_corr : float
        Purity threshold for credible sets.
    refine : bool
        Run the optional refinement procedure to escape local optima.

    Returns
    -------
    SusieFit
        Fitted SuSiE model (see the module docstring of :mod:`pysusie`).
    """
    if estimate_prior_method not in ("optim", "EM", "simple"):
        raise ValueError("estimate_prior_method must be optim, EM or simple")

    X = np.array(X, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True).ravel()
    L_specified = L

    if isinstance(null_weight, (int, float)) and null_weight == 0:
        null_weight = None
    if null_weight is not None:
        if not isinstance(null_weight, (int, float)):
            raise ValueError("Null weight must be numeric")
        if null_weight < 0 or null_weight >= 1:
            raise ValueError("Null weight must be between 0 and 1")
        if prior_weights is None:
            prior_weights = np.append(
                np.full(X.shape[1], 1.0 / X.shape[1] * (1 - null_weight)),
                null_weight)
        else:
            prior_weights = np.append(np.asarray(prior_weights, dtype=float)
                                      * (1 - null_weight), null_weight)
        X = np.hstack([X, np.zeros((X.shape[0], 1))])

    if np.any(np.isnan(X)):
        raise ValueError("Input X must not contain missing values")
    if np.any(np.isnan(y)):
        if na_rm:
            keep = ~np.isnan(y)
            y = y[keep]
            X = X[keep]
        else:
            raise ValueError("Input y must not contain missing values")

    p = X.shape[1]
    n = X.shape[0]
    mean_y = float(np.mean(y))
    if intercept:
        y = y - mean_y

    if L > p:
        L = p
    out = compute_colstats(X, center=intercept, scale=standardize)
    ctx = {"X": X, "cm": out["cm"], "csd": out["csd"], "d": out["d"]}
    varY = float(np.var(y, ddof=1))

    s = init_setup(n, p, L, scaled_prior_variance, residual_variance,
                   prior_weights, null_weight, varY, standardize)
    if s_init is not None:
        if np.max(s_init["alpha"]) > 1 or np.min(s_init["alpha"]) < 0:
            raise ValueError("s_init$alpha has invalid values outside [0,1]")
        s_init = susie_prune_single_effects(_copy_fit(s_init))
        num_effects = s_init["alpha"].shape[0]
        if L_specified is None:
            L = num_effects
        elif min(p, L) < num_effects:
            warnings.warn("Specified L is smaller than effects in s_init")
            L = num_effects
        s_init = susie_prune_single_effects(s_init, min(p, L), s["V"])
        for k in s_init:
            if s_init[k] is not None:
                s[k] = s_init[k]
        s = init_finalize(s, ctx=ctx)
    else:
        s = init_finalize(s)

    if residual_variance_lowerbound is None:
        residual_variance_lowerbound = float(np.var(y, ddof=1)) / 1e4

    elbo = np.full(max_iter + 1, np.nan)
    elbo[0] = -np.inf
    tracking = []
    niter = 0
    for i in range(max_iter):
        niter = i + 1
        if track_fit:
            tracking.append(_susie_slim(s))
        s = _update_each_effect(ctx, y, s, estimate_prior_variance,
                                estimate_prior_method, check_null_threshold)
        if verbose:
            print(f"objective:{get_objective(ctx, y, s)}")
        elbo[i + 1] = get_objective(ctx, y, s)
        if (elbo[i + 1] - elbo[i]) < tol:
            s["converged"] = True
            break
        if estimate_residual_variance:
            s["sigma2"] = max(residual_variance_lowerbound,
                              _estimate_residual_variance(ctx, y, s))
            if s["sigma2"] > residual_variance_upperbound:
                s["sigma2"] = residual_variance_upperbound

    s["elbo"] = elbo[1:niter + 1]
    s["niter"] = niter
    if "converged" not in s:
        warnings.warn(f"IBSS algorithm did not converge in {max_iter} "
                      "iterations!")
        s["converged"] = False

    if intercept:
        s["intercept"] = mean_y - np.sum(
            ctx["cm"] * (np.sum(s["alpha"] * s["mu"], axis=0) / ctx["csd"]))
        s["fitted"] = s["Xr"] + mean_y
    else:
        s["intercept"] = 0.0
        s["fitted"] = s["Xr"]
    s["fitted"] = np.asarray(s["fitted"], dtype=float).ravel()

    if track_fit:
        s["trace"] = tracking

    if coverage is not None and min_abs_corr is not None:
        s["sets"] = susie_get_cs(s, coverage=coverage, X=X,
                                 min_abs_corr=min_abs_corr,
                                 n_purity=n_purity)
        s["pip"] = susie_get_pip(s, prune_by_cs=False, prior_tol=prior_tol)

    if compute_univariate_zscore:
        Xz = X
        if null_weight is not None and null_weight != 0:
            Xz = X[:, :-1]
        s["z"] = calc_z(Xz, y, center=intercept, scale=standardize)

    s["X_column_scale_factors"] = ctx["csd"]

    if refine:
        s = _refine(s, X, y, L, scaled_prior_variance, residual_variance,
                    null_weight, standardize, intercept,
                    estimate_residual_variance, estimate_prior_variance,
                    estimate_prior_method, check_null_threshold, prior_tol,
                    coverage, residual_variance_upperbound, min_abs_corr,
                    compute_univariate_zscore, na_rm, max_iter, tol)
    return s


def _refine(s, X, y, L, scaled_prior_variance, residual_variance,
            null_weight, standardize, intercept, estimate_residual_variance,
            estimate_prior_variance, estimate_prior_method,
            check_null_threshold, prior_tol, coverage,
            residual_variance_upperbound, min_abs_corr,
            compute_univariate_zscore, na_rm, max_iter, tol):
    """Refinement loop to escape IBSS local optima (``refine`` block in ``susie``)."""
    from .getters import susie_get_objective
    if null_weight is not None and null_weight != 0:
        pw_s = np.asarray(s["pi"], dtype=float)
        pw_s = np.delete(pw_s, s["null_index"] - 1) / (1 - null_weight)
        if not compute_univariate_zscore:
            X = X[:, :-1]
    else:
        pw_s = np.asarray(s["pi"], dtype=float)
    conti = True
    while conti and isinstance(s.get("sets"), dict) and s["sets"].get("cs"):
        m = []
        for cs in s["sets"]["cs"]:
            pw_cs = pw_s.copy()
            pw_cs[np.asarray(cs)] = 0.0
            if np.all(pw_cs == 0):
                break
            s2 = susie(X, y, L=L, scaled_prior_variance=scaled_prior_variance,
                       residual_variance=residual_variance,
                       prior_weights=pw_cs, s_init=None,
                       null_weight=null_weight, standardize=standardize,
                       intercept=intercept,
                       estimate_residual_variance=estimate_residual_variance,
                       estimate_prior_variance=estimate_prior_variance,
                       estimate_prior_method=estimate_prior_method,
                       check_null_threshold=check_null_threshold,
                       prior_tol=prior_tol, coverage=coverage,
                       residual_variance_upperbound=residual_variance_upperbound,
                       min_abs_corr=min_abs_corr,
                       compute_univariate_zscore=compute_univariate_zscore,
                       na_rm=na_rm, max_iter=max_iter, tol=tol, refine=False)
            sinit2 = SusieFit(alpha=s2["alpha"], mu=s2["mu"], mu2=s2["mu2"])
            s3 = susie(X, y, L=L, scaled_prior_variance=scaled_prior_variance,
                       residual_variance=residual_variance,
                       prior_weights=pw_s, s_init=sinit2,
                       null_weight=null_weight, standardize=standardize,
                       intercept=intercept,
                       estimate_residual_variance=estimate_residual_variance,
                       estimate_prior_variance=estimate_prior_variance,
                       estimate_prior_method=estimate_prior_method,
                       check_null_threshold=check_null_threshold,
                       prior_tol=prior_tol, coverage=coverage,
                       residual_variance_upperbound=residual_variance_upperbound,
                       min_abs_corr=min_abs_corr,
                       compute_univariate_zscore=compute_univariate_zscore,
                       na_rm=na_rm, max_iter=max_iter, tol=tol, refine=False)
            m.append(s3)
        if not m:
            conti = False
        else:
            elbos = np.array([susie_get_objective(x) for x in m])
            if (np.max(elbos) - susie_get_objective(s)) <= 0:
                conti = False
            else:
                s = m[int(np.argmax(elbos))]
    return s


def _copy_fit(s):
    """Shallow copy of a SusieFit-like object."""
    out = SusieFit()
    for k, v in s.items():
        out[k] = (v.copy() if isinstance(v, np.ndarray) else v)
    return out


def _susie_slim(res):
    """Minimal snapshot of a fit for ``track_fit`` (``susie_slim``)."""
    return SusieFit(alpha=res["alpha"].copy(), niter=res.get("niter"),
                    V=np.asarray(res["V"]).copy(), sigma2=res["sigma2"])


# ----------------------------------------------------------------------
# Prediction / coefficients
# ----------------------------------------------------------------------
def coef_susie(obj):
    """Extract the (p+1)-vector of coefficients (``coef.susie``).

    The first element is the intercept, the rest the per-variable effect
    estimates on the original input scale of X.
    """
    scale = obj.get("X_column_scale_factors")
    if scale is None:
        scale = np.ones(obj["mu"].shape[1])
    eff = np.sum(obj["alpha"] * obj["mu"], axis=0) / scale
    return np.append(obj.get("intercept", 0.0), eff)


def predict_susie(obj, newx=None, type="response"):
    """Predict outcomes or extract coefficients (``predict.susie``)."""
    if type not in ("response", "coefficients"):
        raise ValueError("type must be 'response' or 'coefficients'")
    if type == "coefficients":
        if newx is not None:
            raise ValueError("Do not supply newx when predicting coefficients")
        return coef_susie(obj)
    if newx is None:
        return obj["fitted"]
    coef = coef_susie(obj)
    intercept = obj.get("intercept", 0.0)
    newx = np.asarray(newx, dtype=float)
    if intercept is None or (isinstance(intercept, float) and np.isnan(intercept)):
        warnings.warn("The prediction assumes intercept = 0")
        return newx @ coef[1:]
    return intercept + newx @ coef[1:]
