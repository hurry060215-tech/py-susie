"""``susie_suff_stat`` — IBSS fine-mapping with sufficient statistics.

Faithful port of susieR's ``susie_ss.R`` (the ``susie_suff_stat``
function) and the suff-stat IBSS sweep in ``update_each_effect_ss.R``.

The sufficient statistics are the sample size ``n`` and the matrices
``XtX`` (= X'X), ``Xty`` (= X'y) and scalar ``yty`` (= y'y), all
computed after centering the columns of X and y.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._model import (
    SusieFit, init_setup, init_finalize, get_objective_ss,
    estimate_residual_variance_ss, SER_posterior_e_loglik_ss,
)
from .ser import single_effect_regression_ss
from .getters import susie_get_cs, susie_get_pip
from .susie import susie_prune_single_effects, _copy_fit, _susie_slim

__all__ = ["susie_suff_stat"]


def _is_symmetric(x, tol=1e-10):
    x = np.asarray(x, dtype=float)
    return x.shape[0] == x.shape[1] and np.allclose(x, x.T, atol=tol, rtol=0)


def _muffled_cov2cor(x):
    x = np.asarray(x, dtype=float)
    d = np.sqrt(np.diag(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = x / np.outer(d, d)
    c[~np.isfinite(c)] = 0.0
    np.fill_diagonal(c, 1.0)
    return c


def _update_each_effect_ss(XtX, Xty, s, estimate_prior_variance,
                           estimate_prior_method, check_null_threshold):
    """One IBSS sweep over the L effects, suff-stat (``update_each_effect_ss``)."""
    optimize_V = estimate_prior_method if estimate_prior_variance else "none"
    d = s["_d"]
    L = s["alpha"].shape[0]
    for l in range(L):
        s["XtXr"] = s["XtXr"] - XtX @ (s["alpha"][l] * s["mu"][l])
        XtR = Xty - s["XtXr"]
        res = single_effect_regression_ss(XtR, d, s["V"][l], s["sigma2"],
                                          s["pi"], optimize_V,
                                          check_null_threshold)
        s["mu"][l] = res["mu"]
        s["alpha"][l] = res["alpha"]
        s["mu2"][l] = res["mu2"]
        s["V"][l] = res["V"]
        s["lbf"][l] = res["lbf_model"]
        s["lbf_variable"][l] = res["lbf"]
        s["KL"][l] = (-res["lbf_model"]
                      + SER_posterior_e_loglik_ss(d, XtR, s["sigma2"],
                                                  res["alpha"] * res["mu"],
                                                  res["alpha"] * res["mu2"]))
        s["XtXr"] = s["XtXr"] + XtX @ (s["alpha"][l] * s["mu"][l])
    return s


def susie_suff_stat(XtX, Xty, yty, n, X_colmeans=np.nan, y_mean=np.nan,
                    maf=None, maf_thresh=0.0, L=10, scaled_prior_variance=0.2,
                    residual_variance=None, estimate_residual_variance=True,
                    estimate_prior_variance=True,
                    estimate_prior_method="optim", check_null_threshold=0.0,
                    prior_tol=1e-9, r_tol=1e-8, prior_weights=None,
                    null_weight=0.0, standardize=True, max_iter=100,
                    s_init=None, coverage=0.95, min_abs_corr=0.5, tol=1e-3,
                    verbose=False, track_fit=False, check_input=False,
                    refine=False, check_prior=False, n_purity=100):
    """SuSiE regression with sufficient statistics.

    Faithful Python port of susieR's ``susie_suff_stat``.  See
    :func:`pysusie.susie` for the meaning of the shared hyperparameters.

    Parameters
    ----------
    XtX : array-like, shape (p, p)
        ``X'X`` with columns of X centered to mean zero.
    Xty : array-like, shape (p,)
        ``X'y`` with X columns and y centered to mean zero.
    yty : float
        ``y'y`` with y centered to mean zero.
    n : int
        Sample size.
    X_colmeans, y_mean : float or array-like, optional
        Column means of X and mean of y; required to estimate the
        intercept (otherwise the intercept is NaN).

    Returns
    -------
    SusieFit
        Fitted SuSiE model.
    """
    if estimate_prior_method not in ("optim", "EM", "simple"):
        raise ValueError("estimate_prior_method must be optim, EM or simple")
    if n is None:
        raise ValueError("n must be provided")
    if n <= 1:
        raise ValueError("n must be greater than 1")

    XtX = np.array(XtX, dtype=float, copy=True)
    Xty = np.array(Xty, dtype=float, copy=True).ravel()
    L_specified = L

    if XtX.shape[1] != len(Xty):
        raise ValueError("The dimension of XtX does not agree with Xty")
    if not _is_symmetric(XtX):
        warnings.warn("XtX is not symmetric; forcing XtX to be symmetric")
        XtX = (XtX + XtX.T) / 2.0

    if maf is not None:
        maf = np.asarray(maf, dtype=float)
        if len(maf) != len(Xty):
            raise ValueError("The length of maf does not agree with Xty")
        idx = np.where(maf > maf_thresh)[0]
        XtX = XtX[np.ix_(idx, idx)]
        Xty = Xty[idx]

    if np.any(np.isinf(Xty)):
        raise ValueError("Input Xty contains infinite values")
    if np.any(np.isnan(XtX)):
        raise ValueError("Input XtX matrix contains NAs")
    if np.any(np.isnan(Xty)):
        warnings.warn("NA values in Xty are replaced with 0")
        Xty = np.where(np.isnan(Xty), 0.0, Xty)

    if check_input:
        eigvals = np.linalg.eigvalsh(XtX)
        ev = eigvals.copy()
        ev[np.abs(ev) < r_tol] = 0.0
        if np.any(ev < 0):
            raise ValueError("XtX is not a positive semidefinite matrix")

    if isinstance(null_weight, (int, float)) and null_weight == 0:
        null_weight = None
    if null_weight is not None:
        if not isinstance(null_weight, (int, float)):
            raise ValueError("Null weight must be numeric")
        if null_weight < 0 or null_weight >= 1:
            raise ValueError("Null weight must be between 0 and 1")
        pcur = XtX.shape[1]
        if prior_weights is None:
            prior_weights = np.append(
                np.full(pcur, 1.0 / pcur * (1 - null_weight)), null_weight)
        else:
            prior_weights = np.append(np.asarray(prior_weights, dtype=float)
                                      * (1 - null_weight), null_weight)
        XtX = np.pad(XtX, ((0, 1), (0, 1)), mode="constant")
        Xty = np.append(Xty, 0.0)

    p = XtX.shape[1]

    if standardize:
        dXtX = np.diag(XtX).copy()
        csd = np.sqrt(dXtX / (n - 1))
        csd[csd == 0] = 1.0
        XtX = (XtX / csd).T / csd
        Xty = Xty / csd
    else:
        csd = np.ones(p)
    d = np.diag(XtX).copy()

    X_colmeans = np.atleast_1d(np.asarray(X_colmeans, dtype=float))
    if len(X_colmeans) == 1:
        X_colmeans = np.full(p, X_colmeans[0])
    if len(X_colmeans) != p:
        raise ValueError("The length of X_colmeans does not agree with p")

    s = init_setup(0, p, L, scaled_prior_variance, residual_variance,
                   prior_weights, null_weight, yty / (n - 1), standardize)
    s["Xr"] = None
    s["XtXr"] = np.zeros(p)
    s["_d"] = d

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
        V = np.atleast_1d(np.asarray(s["V"], dtype=float))
        if V.shape[0] == 1:
            V = np.full(s["alpha"].shape[0], V[0])
        s["V"] = V
        s["sigma2"] = float(s["sigma2"])
        b = np.sum(s["mu"] * s["alpha"], axis=0)
        s["XtXr"] = XtX @ b
        s["KL"] = np.full(s["alpha"].shape[0], np.nan)
        s["lbf"] = np.full(s["alpha"].shape[0], np.nan)
    else:
        s = init_finalize(s)
        s["XtXr"] = np.zeros(p)
    s["_d"] = d

    bhat = (1.0 / d) * Xty
    with np.errstate(invalid="ignore", divide="ignore"):
        shat = np.sqrt(s["sigma2"] / d)
        z = bhat / shat
    zm = float(np.max(np.abs(z[~np.isnan(z)]))) if np.any(~np.isnan(z)) else 0.0

    elbo = np.full(max_iter + 1, np.nan)
    elbo[0] = -np.inf
    tracking = []
    niter = 0
    for i in range(max_iter):
        niter = i + 1
        if track_fit:
            tracking.append(_susie_slim(s))
        s = _update_each_effect_ss(XtX, Xty, s, estimate_prior_variance,
                                   estimate_prior_method, check_null_threshold)
        if check_prior and np.any(np.asarray(s["V"]) > 100 * zm ** 2):
            raise ValueError("The estimated prior variance is unreasonably "
                             "large. This is usually caused by mismatch "
                             "between the summary statistics and LD matrix.")
        elbo[i + 1] = get_objective_ss(XtX, Xty, s, yty, n)
        if np.isinf(elbo[i + 1]):
            raise ValueError("The objective becomes infinite.")
        if (elbo[i + 1] - elbo[i]) < tol:
            s["converged"] = True
            break
        if estimate_residual_variance:
            est = estimate_residual_variance_ss(XtX, Xty, s, yty, n)
            if est < 0:
                raise ValueError("Estimating residual variance failed: the "
                                 "estimated value is negative")
            s["sigma2"] = est

    s["elbo"] = elbo[1:niter + 1]
    s["niter"] = niter
    if "converged" not in s:
        warnings.warn(f"IBSS algorithm did not converge in {max_iter} "
                      "iterations!")
        s["converged"] = False

    s["X_column_scale_factors"] = csd
    s["intercept"] = y_mean - np.sum(
        X_colmeans * (np.sum(s["alpha"] * s["mu"], axis=0) / csd))

    if track_fit:
        s["trace"] = tracking

    if coverage is not None and min_abs_corr is not None:
        if np.any(~np.isin(np.diag(XtX), (0.0, 1.0))):
            Xcorr = _muffled_cov2cor(XtX)
        else:
            Xcorr = XtX
        s["sets"] = susie_get_cs(s, coverage=coverage, Xcorr=Xcorr,
                                 min_abs_corr=min_abs_corr,
                                 check_symmetric=False, n_purity=n_purity)
        s["pip"] = susie_get_pip(s, prune_by_cs=False, prior_tol=prior_tol)

    if refine:
        s = _refine_ss(s, XtX, Xty, yty, n, L, X_colmeans, y_mean,
                       scaled_prior_variance, residual_variance,
                       estimate_residual_variance, estimate_prior_variance,
                       estimate_prior_method, check_null_threshold,
                       prior_tol, r_tol, null_weight, standardize, coverage,
                       min_abs_corr, tol, p)
    return s


def _refine_ss(s, XtX, Xty, yty, n, L, X_colmeans, y_mean,
               scaled_prior_variance, residual_variance,
               estimate_residual_variance, estimate_prior_variance,
               estimate_prior_method, check_null_threshold, prior_tol,
               r_tol, null_weight, standardize, coverage, min_abs_corr,
               tol, p):
    """Refinement loop for ``susie_suff_stat`` (``refine`` block in ``susie_ss``)."""
    from .getters import susie_get_objective
    # Note: the refine block in susie_ss re-supplies XtX/Xty as already
    # standardized matrices; for parity we feed standardize=False so they
    # are not re-standardized.
    if null_weight is not None and null_weight != 0:
        XtX = XtX[:p - 1, :p - 1]
        Xty = Xty[:p - 1]
        pw_s = np.delete(np.asarray(s["pi"], dtype=float),
                         s["null_index"] - 1) / (1 - null_weight)
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
            s2 = susie_suff_stat(XtX, Xty, yty, n, L=L,
                                 X_colmeans=X_colmeans, y_mean=y_mean,
                                 prior_weights=pw_cs, s_init=None,
                                 scaled_prior_variance=scaled_prior_variance,
                                 residual_variance=residual_variance,
                                 estimate_residual_variance=estimate_residual_variance,
                                 estimate_prior_variance=estimate_prior_variance,
                                 estimate_prior_method=estimate_prior_method,
                                 check_null_threshold=check_null_threshold,
                                 prior_tol=prior_tol, r_tol=r_tol,
                                 null_weight=null_weight, standardize=False,
                                 coverage=coverage, min_abs_corr=min_abs_corr,
                                 tol=tol, refine=False)
            sinit2 = SusieFit(alpha=s2["alpha"], mu=s2["mu"], mu2=s2["mu2"])
            s3 = susie_suff_stat(XtX, Xty, yty, n, L=L,
                                 X_colmeans=X_colmeans, y_mean=y_mean,
                                 prior_weights=pw_s, s_init=sinit2,
                                 scaled_prior_variance=scaled_prior_variance,
                                 residual_variance=residual_variance,
                                 estimate_residual_variance=estimate_residual_variance,
                                 estimate_prior_variance=estimate_prior_variance,
                                 estimate_prior_method=estimate_prior_method,
                                 check_null_threshold=check_null_threshold,
                                 prior_tol=prior_tol, r_tol=r_tol,
                                 null_weight=null_weight, standardize=False,
                                 coverage=coverage, min_abs_corr=min_abs_corr,
                                 tol=tol, refine=False)
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
