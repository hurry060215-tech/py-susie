"""``susie_rss`` — SuSiE fine-mapping from summary statistics.

Faithful port of susieR's ``susie_rss.R`` (the ``susie_rss`` dispatcher),
``susie_rss_lambda.R`` (the RSS IBSS used when ``n`` is unknown), and the
LD-mismatch diagnostics ``estimate_s_rss`` / ``kriging_rss`` from
``susie_utils.R``.

When the sample size ``n`` is provided, ``susie_rss`` builds the
sufficient statistics ``XtX = (n-1)R``, ``Xty = sqrt(n-1) z_tilde``,
``yty = n-1`` and calls :func:`pysusie.susie_suff_stat`.  When ``n`` is
unknown, it calls :func:`pysusie.susie_suff_stat` with ``XtX = R``,
``Xty = z`` and a fixed residual variance of 1.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm as _norm

from ._model import (
    SusieFit, init_setup_rss, init_finalize_rss, get_objective_rss,
    Eloglik_rss, get_ER2_rss, SER_posterior_e_loglik_rss,
)
from .ser import single_effect_regression_rss
from .getters import susie_get_cs, susie_get_pip
from .susie import susie_prune_single_effects, _copy_fit, _susie_slim
from .suff_stat import susie_suff_stat

__all__ = ["susie_rss", "susie_rss_lambda", "estimate_s_rss", "kriging_rss"]


def _is_symmetric(x, tol=1e-8):
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


# ----------------------------------------------------------------------
# susie_rss dispatcher
# ----------------------------------------------------------------------
def susie_rss(z=None, R=None, n=None, bhat=None, shat=None, var_y=None,
              z_ld_weight=0.0, estimate_residual_variance=False,
              prior_variance=50.0, check_prior=True, **kwargs):
    """SuSiE regression using summary statistics.

    Faithful Python port of susieR's ``susie_rss``.

    Parameters
    ----------
    z : array-like, shape (p,), optional
        Z-scores.  Provide either ``z`` or (``bhat``, ``shat``).
    R : array-like, shape (p, p)
        Correlation (LD) matrix.
    n : int, optional
        Sample size (strongly recommended).
    bhat, shat : array-like, optional
        Estimated effects and their standard errors (alternative to ``z``).
    var_y : float, optional
        Sample variance of y; when given with ``shat`` the effects are
        returned on the original scale.
    estimate_residual_variance : bool
        Whether to estimate the residual variance (recommended only with
        an in-sample LD matrix).
    prior_variance : float
        Prior variance used when ``n`` is unknown.
    **kwargs
        Passed through to :func:`pysusie.susie_suff_stat`.

    Returns
    -------
    SusieFit
        Fitted SuSiE model.
    """
    if R is None:
        raise ValueError("R must be provided")
    R = np.asarray(R, dtype=float)

    if z is None and (bhat is None or shat is None):
        raise ValueError("Please provide either z or (bhat, shat)")
    if z is not None and (bhat is not None or shat is not None):
        raise ValueError("Please provide either z or (bhat, shat), not both")

    if z is None:
        bhat = np.asarray(bhat, dtype=float).ravel()
        shat = np.atleast_1d(np.asarray(shat, dtype=float)).ravel()
        if len(shat) == 1:
            shat = np.full(len(bhat), shat[0])
        if len(bhat) != len(shat):
            raise ValueError("The lengths of bhat and shat do not agree")
        if np.any(np.isnan(bhat)) or np.any(np.isnan(shat)):
            raise ValueError("bhat, shat cannot have missing values")
        if np.any(shat <= 0):
            raise ValueError("shat cannot have zero or negative elements")
        z = bhat / shat
        p = len(z)
    else:
        z = np.asarray(z, dtype=float).ravel()
        p = len(z)

    if R.shape[0] != p:
        raise ValueError(f"The dimension of R ({R.shape[0]} x {R.shape[1]}) "
                         f"does not agree with expected ({p} x {p})")
    if n is not None and n <= 1:
        raise ValueError("n must be greater than 1")
    if len(z) < 1:
        raise ValueError("Input vector z should have at least one element")
    z = np.where(np.isnan(z), 0.0, z)

    adj = None
    if n is not None:
        adj = (n - 1) / (z ** 2 + n - 2)
        z = np.sqrt(adj) * z

    if z_ld_weight > 0:
        warnings.warn("As of version 0.11.0, use of non-zero z_ld_weight is "
                      "no longer recommended")
        R = _muffled_cov2cor((1 - z_ld_weight) * R
                             + z_ld_weight * np.outer(z, z))
        R = (R + R.T) / 2.0

    if n is None:
        warnings.warn("Providing the sample size (n) is highly recommended.")
        s = susie_suff_stat(XtX=R, Xty=z, n=2, yty=1.0,
                            scaled_prior_variance=prior_variance,
                            estimate_residual_variance=estimate_residual_variance,
                            standardize=False, check_prior=check_prior,
                            **kwargs)
    else:
        if shat is not None and var_y is not None:
            shat = np.atleast_1d(np.asarray(shat, dtype=float)).ravel()
            if len(shat) == 1:
                shat = np.full(p, shat[0])
            XtXdiag = var_y * adj / (shat ** 2)
            XtX = (R * np.sqrt(XtXdiag)).T * np.sqrt(XtXdiag)
            XtX = (XtX + XtX.T) / 2.0
            Xty = z * np.sqrt(adj) * var_y / shat
        else:
            XtX = (n - 1) * R
            Xty = np.sqrt(n - 1) * z
            var_y = 1.0
        s = susie_suff_stat(XtX=XtX, Xty=Xty, n=n, yty=(n - 1) * var_y,
                            estimate_residual_variance=estimate_residual_variance,
                            check_prior=check_prior, **kwargs)
    return s


# ----------------------------------------------------------------------
# RSS attributes / Sigma update
# ----------------------------------------------------------------------
def _set_R_attributes(R, r_tol):
    """Eigen-decompose R, drop small/negative eigenvalues (``set_R_attributes``)."""
    vals, vecs = np.linalg.eigh(R)
    # eigh returns ascending; R's eigen() returns descending. Reverse.
    vals = vals[::-1].copy()
    vecs = vecs[:, ::-1].copy()
    vals[np.abs(vals) < r_tol] = 0.0
    if np.any(vals < 0):
        min_lambda = float(np.min(vals))
        vals[vals < 0] = 0.0
        warnings.warn(f"The input correlation matrix has negative eigenvalues "
                      f"(smallest {min_lambda}); they are set to zero.")
    res = vecs @ (vecs.T * vals[:, None])
    return res, {"values": vals, "vectors": vecs}


def _update_Sigma(R_ctx, sigma2, z):
    """Build the Sigma context (``update_Sigma`` in ``susie_rss_lambda.R``)."""
    eig_vals = R_ctx["eigen_values"]
    eig_vecs = R_ctx["eigen_vectors"]
    lam = R_ctx["lambda"]
    svals = sigma2 * eig_vals + lam
    with np.errstate(divide="ignore"):
        Dinv = 1.0 / svals
    Dinv[~np.isfinite(Dinv)] = 0.0
    SinvRj = eig_vecs @ ((Dinv * eig_vals)[:, None] * eig_vecs.T)
    if lam == 0:
        RjSinvRj = R_ctx["d"] / sigma2
    else:
        tmp = eig_vecs.T
        RjSinvRj = np.sum(tmp * ((Dinv * (eig_vals ** 2))[:, None] * tmp),
                          axis=0)
    return {"eigenS_values": svals, "eigenS_vectors": eig_vecs,
            "SinvRj": SinvRj, "RjSinvRj": RjSinvRj}


def _update_each_effect_rss(R, z, s, sigma_attrs, estimate_prior_variance,
                            estimate_prior_method, check_null_threshold):
    """One IBSS sweep over the L effects, RSS (``update_each_effect_rss``)."""
    optimize_V = estimate_prior_method if estimate_prior_variance else "none"
    L = s["alpha"].shape[0]
    for l in range(L):
        s["Rz"] = s["Rz"] - R @ (s["alpha"][l] * s["mu"][l])
        r = np.asarray(z, dtype=float) - s["Rz"]
        res = single_effect_regression_rss(r, sigma_attrs, s["V"][l], s["pi"],
                                           optimize_V, check_null_threshold)
        s["mu"][l] = res["mu"]
        s["alpha"][l] = res["alpha"]
        s["mu2"][l] = res["mu2"]
        s["V"][l] = res["V"]
        s["lbf"][l] = res["lbf_model"]
        s["lbf_variable"][l] = res["lbf"]
        s["KL"][l] = (-res["lbf_model"]
                      + SER_posterior_e_loglik_rss(R, sigma_attrs, r,
                                                   res["alpha"] * res["mu"],
                                                   res["alpha"] * res["mu2"]))
        s["Rz"] = s["Rz"] + R @ (s["alpha"][l] * s["mu"][l])
    return s


def susie_rss_lambda(z, R, maf=None, maf_thresh=0.0, L=10, lam=0.0,
                     prior_variance=50.0, residual_variance=None,
                     r_tol=1e-8, prior_weights=None, null_weight=0.0,
                     estimate_residual_variance=True,
                     estimate_prior_variance=True,
                     estimate_prior_method="optim", check_null_threshold=0.0,
                     prior_tol=1e-9, max_iter=100, s_init=None,
                     intercept_value=0.0, coverage=0.95, min_abs_corr=0.5,
                     tol=1e-3, verbose=False, track_fit=False, check_R=True,
                     check_z=False):
    """RSS SuSiE with lambda regularisation (``susie_rss_lambda``).

    This is the IBSS algorithm that fits ``z = sum_l R b_l + e`` with
    ``e ~ N(0, sigma2 R + lambda I)``.  ``susie_rss`` no longer calls this
    by default, but it is exposed for completeness and faithful coverage.
    """
    z = np.asarray(z, dtype=float).ravel()
    R = np.array(R, dtype=float, copy=True)
    if estimate_prior_method not in ("optim", "EM", "simple"):
        raise ValueError("estimate_prior_method must be optim, EM or simple")
    if R.shape[0] != len(z):
        raise ValueError("The dimension of R does not agree with z")
    if not _is_symmetric(R):
        raise ValueError("R is not a symmetric matrix")

    if maf is not None:
        maf = np.asarray(maf, dtype=float)
        idx = np.where(maf > maf_thresh)[0]
        R = R[np.ix_(idx, idx)]
        z = z[idx]
    if np.any(np.isinf(z)):
        raise ValueError("z contains infinite values")
    if np.any(np.isnan(R)):
        raise ValueError("R matrix contains missing values")
    if np.any(np.isnan(z)):
        warnings.warn("NA values in z-scores are replaced with 0")
        z = np.where(np.isnan(z), 0.0, z)

    if isinstance(null_weight, (int, float)) and null_weight == 0:
        null_weight = None
    if null_weight is not None:
        pcur = R.shape[1]
        if prior_weights is None:
            prior_weights = np.append(
                np.full(pcur, 1.0 / pcur * (1 - null_weight)), null_weight)
        else:
            prior_weights = np.append(np.asarray(prior_weights, dtype=float)
                                      * (1 - null_weight), null_weight)
        R = np.pad(R, ((0, 1), (0, 1)), mode="constant")
        z = np.append(z, 0.0)

    p = R.shape[1]
    raw_vals, raw_vecs = np.linalg.eigh(R)
    raw_vals = raw_vals[::-1].copy()
    raw_vecs = raw_vecs[:, ::-1].copy()
    if check_R and np.any(raw_vals < -r_tol):
        raise ValueError("The correlation matrix is not positive "
                         "semidefinite. Use check_R=False to continue.")

    R_reg, eig = _set_R_attributes(R, r_tol)
    eig_vals = eig["values"]
    eig_vecs = eig["vectors"]
    d = np.diag(R_reg).copy()

    if isinstance(lam, str) and lam == "estimate":
        colspace = np.where(eig_vals > 0)[0]
        if len(colspace) == len(z):
            lam = 0.0
        else:
            null_cols = np.setdiff1d(np.arange(len(eig_vals)), colspace)
            znull = eig_vecs[:, null_cols].T @ z
            lam = float(np.sum(znull ** 2) / len(znull))

    s = init_setup_rss(p, L, prior_variance, residual_variance,
                       prior_weights, null_weight)
    if s_init is not None:
        if np.max(s_init["alpha"]) > 1 or np.min(s_init["alpha"]) < 0:
            raise ValueError("s_init$alpha has invalid values outside [0,1]")
        s_init = susie_prune_single_effects(_copy_fit(s_init))
        num_effects = s_init["alpha"].shape[0]
        if L < num_effects:
            warnings.warn("Specified L is smaller than effects in s_init")
            L = num_effects
        s_init = susie_prune_single_effects(s_init, L, s["V"])
        for k in s_init:
            if s_init[k] is not None:
                s[k] = s_init[k]
        s = init_finalize_rss(s, R=R_reg)
    else:
        s = init_finalize_rss(s)

    s["sigma2"] = s["sigma2"] - lam

    R_ctx = {"R": R_reg, "eigen_values": eig_vals, "eigen_vectors": eig_vecs,
             "d": d, "lambda": lam}
    sigma_attrs = _update_Sigma(R_ctx, s["sigma2"], z)

    elbo = np.full(max_iter + 1, np.nan)
    elbo[0] = -np.inf
    tracking = []
    niter = 0
    for i in range(max_iter):
        niter = i + 1
        if track_fit:
            tracking.append(_susie_slim(s))
        s = _update_each_effect_rss(R_reg, z, s, sigma_attrs,
                                    estimate_prior_variance,
                                    estimate_prior_method,
                                    check_null_threshold)
        elbo[i + 1] = get_objective_rss(R_ctx, z, s)
        if (elbo[i + 1] - elbo[i]) < tol:
            s["converged"] = True
            break
        if estimate_residual_variance:
            if lam == 0:
                nz = np.sum(eig_vals != 0)
                est = (1.0 / nz) * get_ER2_rss(1.0, R_ctx, z, s)
                if est < 0:
                    raise ValueError("Estimating residual variance failed: "
                                     "the estimated value is negative")
                est = min(est, 1.0)
            else:
                res = minimize_scalar(
                    lambda v: -Eloglik_rss(v, R_ctx, z, s),
                    bounds=(1e-4, 1 - lam), method="bounded")
                est = res.x
                if Eloglik_rss(est, R_ctx, z, s) < \
                        Eloglik_rss(1 - lam, R_ctx, z, s):
                    est = 1 - lam
            s["sigma2"] = est
            sigma_attrs = _update_Sigma(R_ctx, s["sigma2"], z)

    s["elbo"] = elbo[1:niter + 1]
    s["niter"] = niter
    s["lambda"] = lam
    if "converged" not in s:
        warnings.warn(f"IBSS algorithm did not converge in {max_iter} "
                      "iterations!")
        s["converged"] = False

    s["intercept"] = intercept_value
    s["fitted"] = s["Rz"]
    s["X_column_scale_factors"] = np.ones(p)

    if track_fit:
        s["trace"] = tracking

    if coverage is not None and min_abs_corr is not None:
        Xcorr = _muffled_cov2cor(R_reg)
        s["sets"] = susie_get_cs(s, coverage=coverage, Xcorr=Xcorr,
                                 min_abs_corr=min_abs_corr)
        s["pip"] = susie_get_pip(s, prune_by_cs=False, prior_tol=prior_tol)
    return s


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def estimate_s_rss(z, R, n=None, r_tol=1e-8, method="null-mle"):
    """Estimate ``s`` quantifying z-score / LD inconsistency (``estimate_s_rss``).

    A larger ``s`` indicates a stronger inconsistency between the z-scores
    and the LD matrix ``R``.  Returns a number, usually between 0 and 1.
    """
    z = np.asarray(z, dtype=float).ravel().copy()
    z[np.isnan(z)] = 0.0
    R = np.asarray(R, dtype=float)
    vals, vecs = np.linalg.eigh(R)
    vals = vals[::-1].copy()
    vecs = vecs[:, ::-1].copy()
    if np.any(vals < -r_tol):
        warnings.warn("The matrix R is not positive semidefinite. Negative "
                      "eigenvalues are set to zero")
    vals[vals < r_tol] = 0.0

    if n is None:
        warnings.warn("Providing the sample size (n) is highly recommended.")
    elif n <= 1:
        raise ValueError("n must be greater than 1")
    if n is not None:
        sigma2 = (n - 1) / (z ** 2 + n - 2)
        z = np.sqrt(sigma2) * z

    if method == "null-mle":
        ztv = z @ vecs

        def negll(s):
            denom = (1 - s) * vals + s
            return 0.5 * np.sum(np.log(denom)) + 0.5 * np.sum(ztv ** 2 / denom)

        res = minimize_scalar(negll, bounds=(0.0, 1.0), method="bounded")
        return float(res.x)
    elif method == "null-partialmle":
        colspace = np.where(vals > 0)[0]
        if len(colspace) == len(z):
            return 0.0
        null_cols = np.setdiff1d(np.arange(len(vals)), colspace)
        znull = vecs[:, null_cols].T @ z
        return float(np.sum(znull ** 2) / len(znull))
    elif method == "null-pseudomle":
        def pseudoll(s):
            denom = (1 - s) * vals + s
            precision = vecs @ (vecs.T * (1.0 / denom)[:, None])
            postmean = np.zeros(len(z))
            postvar = np.zeros(len(z))
            for i in range(len(z)):
                others = np.delete(np.arange(len(z)), i)
                postmean[i] = -(1.0 / precision[i, i]) * \
                    (precision[i, others] @ z[others])
                postvar[i] = 1.0 / precision[i, i]
            return -np.sum(_norm.logpdf(z, loc=postmean,
                                        scale=np.sqrt(postvar)))

        res = minimize_scalar(pseudoll, bounds=(0.0, 1.0), method="bounded")
        return float(res.x)
    else:
        raise ValueError("The method is not implemented")


def _mixsqp_weights(matrix_llik):
    """Estimate mixture weights from a log-likelihood matrix.

    A small EM substitute for the ``mixsqp`` solver used in R's
    ``kriging_rss``.  ``matrix_llik`` is n-by-k log-likelihoods (already
    scaled by per-row max factors).
    """
    L = np.exp(matrix_llik)
    n, k = L.shape
    w = np.full(k, 1.0 / k)
    for _ in range(2000):
        num = L * w[None, :]
        denom = num.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1e-300
        post = num / denom
        new_w = post.mean(axis=0)
        if np.max(np.abs(new_w - w)) < 1e-9:
            w = new_w
            break
        w = new_w
    return w


def kriging_rss(z, R, n=None, r_tol=1e-8, s=None):
    """Conditional z-score distribution & allele-switch diagnostic (``kriging_rss``).

    Returns a dict with ``conditional_dist`` (a pandas DataFrame with
    columns ``z``, ``condmean``, ``condvar``, ``z_std_diff``, ``logLR``)
    and ``plot_data`` for plotting observed vs expected z-scores.
    """
    import pandas as pd

    z = np.asarray(z, dtype=float).ravel().copy()
    z[np.isnan(z)] = 0.0
    R = np.asarray(R, dtype=float)
    vals, vecs = np.linalg.eigh(R)
    vals = vals[::-1].copy()
    vecs = vecs[:, ::-1].copy()
    if np.any(vals < -r_tol):
        warnings.warn("The matrix R is not positive semidefinite.")
    vals[vals < r_tol] = 0.0

    if s is None:
        s = estimate_s_rss(z, R, n, r_tol, method="null-mle")
    if s > 1:
        warnings.warn("The given s is greater than 1; replaced with 0.8.")
        s = 0.8
    elif s < 0:
        raise ValueError("The s must be non-negative")

    if n is not None and n <= 1:
        raise ValueError("n must be greater than 1")
    if n is None:
        warnings.warn("Providing the sample size (n) is highly recommended.")
    else:
        sigma2 = (n - 1) / (z ** 2 + n - 2)
        z = np.sqrt(sigma2) * z

    denom = (1 - s) * vals + s
    with np.errstate(divide="ignore"):
        dinv = 1.0 / denom
    dinv[~np.isfinite(dinv)] = 0.0
    precision = vecs @ (vecs.T * dinv[:, None])
    p = len(z)
    condmean = np.zeros(p)
    condvar = np.zeros(p)
    for i in range(p):
        others = np.delete(np.arange(p), i)
        condmean[i] = -(1.0 / precision[i, i]) * (precision[i, others] @ z[others])
        condvar[i] = 1.0 / precision[i, i]
    z_std_diff = (z - condmean) / np.sqrt(condvar)

    a_min = 0.8
    if np.max(z_std_diff ** 2) < 1:
        a_max = 2.0
    else:
        a_max = 2 * np.sqrt(np.max(z_std_diff ** 2))
    npoint = int(np.ceil(np.log2(a_max / a_min) / np.log2(1.05)))
    a_grid = 1.05 ** np.arange(-npoint, 1) * a_max

    sd_mtx = np.outer(np.sqrt(condvar), a_grid)
    matrix_llik = _norm.logpdf((z - condmean)[:, None], scale=sd_mtx)
    lfactors = matrix_llik.max(axis=1)
    matrix_llik = matrix_llik - lfactors[:, None]
    w = _mixsqp_weights(matrix_llik)
    logl0mix = np.log(np.exp(matrix_llik) @ (w + 1e-15)) + lfactors

    matrix_llik = _norm.logpdf((z + condmean)[:, None], scale=sd_mtx)
    lfactors = matrix_llik.max(axis=1)
    matrix_llik = matrix_llik - lfactors[:, None]
    logl1mix = np.log(np.exp(matrix_llik) @ (w + 1e-15)) + lfactors
    logLRmix = logl1mix - logl0mix

    df = pd.DataFrame({"z": z, "condmean": condmean, "condvar": condvar,
                       "z_std_diff": z_std_diff, "logLR": logLRmix})
    return {"conditional_dist": df, "plot_data": df}
