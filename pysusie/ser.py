"""Bayesian single-effect regression (SER) and prior-variance estimation.

Faithful port of susieR's ``single_effect_regression.R``,
``single_effect_regression_ss.R`` and ``single_effect_regression_rss.R``.

The SER model fits ``y = Xb + e``, ``e ~ N(0, s^2)``, where ``b`` has
exactly one non-zero element.  The prior on the non-zero element is
``N(0, V)``.  Given summary statistics ``betahat`` and ``shat2`` per
variable, the per-variable log Bayes factor is

    lbf_j = dnorm(betahat_j; 0, sqrt(V + shat2_j), log) -
            dnorm(betahat_j; 0, sqrt(shat2_j), log)

and the posterior inclusion probabilities are the softmax of
``lbf + log(prior_weights)``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm as _norm

from ._linalg import compute_Xty

__all__ = [
    "single_effect_regression",
    "single_effect_regression_ss",
    "single_effect_regression_rss",
    "loglik",
    "optimize_prior_variance",
]

# sqrt(.Machine$double.eps) used by R for numerical stability.
_SQRT_EPS = np.sqrt(np.finfo(float).eps)


def _dnorm_log(x, sd):
    """log dnorm(x, mean=0, sd), elementwise; matches R's dnorm(...,log=TRUE)."""
    x = np.asarray(x, dtype=float)
    sd = np.asarray(sd, dtype=float)
    return _norm.logpdf(x, loc=0.0, scale=sd)


# ----------------------------------------------------------------------
# Likelihood as a function of the prior variance V (summary-data scale)
# ----------------------------------------------------------------------
def loglik(V, betahat, shat2, prior_weights):
    """Log-likelihood of the SER model as a function of prior variance V.

    Mirrors ``loglik`` in ``single_effect_regression.R``.
    """
    betahat = np.asarray(betahat, dtype=float)
    shat2 = np.asarray(shat2, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)
    lbf = _dnorm_log(betahat, np.sqrt(V + shat2)) - _dnorm_log(betahat, np.sqrt(shat2))
    lpo = lbf + np.log(prior_weights + _SQRT_EPS)
    inf = np.isinf(shat2)
    lbf = np.where(inf, 0.0, lbf)
    lpo = np.where(inf, 0.0, lpo)
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    return float(np.log(np.sum(w_weighted)) + maxlpo)


def _neg_loglik_logscale(lV, betahat, shat2, prior_weights):
    return -loglik(np.exp(lV), betahat, shat2, prior_weights)


def _lbf_grad(V, shat2, T2):
    """Gradient of logBF_j wrt prior variance V (``lbf.grad`` in R)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        l = 0.5 * (1.0 / (V + shat2)) * ((shat2 / (V + shat2)) * T2 - 1.0)
    l = np.where(np.isnan(l), 0.0, l)
    return l


def _loglik_grad(V, betahat, shat2, prior_weights):
    """Gradient of the SER log-likelihood wrt V (``loglik.grad`` in R)."""
    betahat = np.asarray(betahat, dtype=float)
    shat2 = np.asarray(shat2, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)
    lbf = _dnorm_log(betahat, np.sqrt(V + shat2)) - _dnorm_log(betahat, np.sqrt(shat2))
    lpo = lbf + np.log(prior_weights + _SQRT_EPS)
    inf = np.isinf(shat2)
    lbf = np.where(inf, 0.0, lbf)
    lpo = np.where(inf, 0.0, lpo)
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    alpha = w_weighted / np.sum(w_weighted)
    with np.errstate(invalid="ignore", divide="ignore"):
        T2 = betahat ** 2 / shat2
    return float(np.sum(alpha * _lbf_grad(V, shat2, T2)))


def _negloglik_grad_logscale(lV, betahat, shat2, prior_weights):
    return -np.exp(lV) * _loglik_grad(np.exp(lV), betahat, shat2, prior_weights)


def _est_V_uniroot(betahat, shat2, prior_weights):
    """Root-finding estimate of V (``est_V_uniroot`` in R).

    R uses ``uniroot`` with ``extendInt = "upX"`` on the interval
    [-10, 10] (on the log scale).  We replicate by widening the bracket
    until a sign change is found, then call brentq.
    """
    f = lambda lV: _negloglik_grad_logscale(lV, betahat, shat2, prior_weights)
    lo, hi = -10.0, 10.0
    flo, fhi = f(lo), f(hi)
    # extendInt = "upX": function is increasing; extend until sign change.
    it = 0
    while flo > 0 and it < 60:
        lo -= 10.0
        flo = f(lo)
        it += 1
    it = 0
    while fhi < 0 and it < 60:
        hi += 10.0
        fhi = f(hi)
        it += 1
    if flo > 0 or fhi < 0:
        return 0.0
    root = brentq(f, lo, hi, xtol=1e-8, rtol=1e-10)
    return float(np.exp(root))


def optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                            alpha=None, post_mean2=None, V_init=None,
                            check_null_threshold=0.0):
    """Estimate the SER prior variance.

    Faithful port of ``optimize_prior_variance`` in
    ``single_effect_regression.R``.  ``optimize_V`` is one of
    ``"simple"``, ``"optim"``, ``"uniroot"`` or ``"EM"``.
    """
    betahat = np.asarray(betahat, dtype=float)
    shat2 = np.asarray(shat2, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)
    V = V_init
    if optimize_V != "simple":
        if optimize_V == "optim":
            init = np.log(np.nanmax(np.append(betahat ** 2 - shat2, 1.0)))
            res = minimize_scalar(
                _neg_loglik_logscale, bounds=(-30.0, 15.0),
                args=(betahat, shat2, prior_weights), method="bounded",
                options={"xatol": 1e-8},
            )
            lV = res.x
            # if the estimate is worse than current, keep current.
            with np.errstate(divide="ignore"):
                logV = np.log(V)
            if (_neg_loglik_logscale(lV, betahat, shat2, prior_weights)
                    > _neg_loglik_logscale(logV, betahat, shat2, prior_weights)):
                lV = logV
            V = float(np.exp(lV))
        elif optimize_V == "uniroot":
            V = _est_V_uniroot(betahat, shat2, prior_weights)
        elif optimize_V == "EM":
            V = float(np.sum(np.asarray(alpha) * np.asarray(post_mean2)))
        else:
            raise ValueError("Invalid option for optimize_V method")
    # set V exactly 0 if the null beats the estimate by check_null_threshold.
    if (loglik(0.0, betahat, shat2, prior_weights) + check_null_threshold
            >= loglik(V, betahat, shat2, prior_weights)):
        V = 0.0
    return float(V)


# ----------------------------------------------------------------------
# Single-effect regression — individual-level data
# ----------------------------------------------------------------------
def single_effect_regression(y, ctx, V, residual_variance=1.0,
                             prior_weights=None, optimize_V="none",
                             check_null_threshold=0.0):
    """Bayesian single-effect regression with individual-level data.

    Faithful port of ``single_effect_regression`` in
    ``single_effect_regression.R``.  ``ctx`` is the susie matrix context
    (dict with ``X``, ``cm``, ``csd``, ``d``).
    """
    d = ctx["d"]
    Xty = compute_Xty(ctx, y)
    betahat = (1.0 / d) * Xty
    shat2 = residual_variance / d
    p = ctx["X"].shape[1]
    if prior_weights is None:
        prior_weights = np.full(p, 1.0 / p)
    prior_weights = np.asarray(prior_weights, dtype=float)

    if optimize_V not in ("EM", "none"):
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                                    alpha=None, post_mean2=None, V_init=V,
                                    check_null_threshold=check_null_threshold)

    lbf = _dnorm_log(betahat, np.sqrt(V + shat2)) - _dnorm_log(betahat, np.sqrt(shat2))
    lpo = lbf + np.log(prior_weights + _SQRT_EPS)
    inf = np.isinf(shat2)
    lbf = np.where(inf, 0.0, lbf)
    lpo = np.where(inf, 0.0, lpo)
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    weighted_sum_w = np.sum(w_weighted)
    alpha = w_weighted / weighted_sum_w
    post_var = 1.0 / (1.0 / V + d / residual_variance) if V > 0 else np.zeros_like(d)
    post_mean = (1.0 / residual_variance) * post_var * Xty
    post_mean2 = post_var + post_mean ** 2
    lbf_model = maxlpo + np.log(weighted_sum_w)
    loglik_val = lbf_model + np.sum(_dnorm_log(np.asarray(y, dtype=float),
                                               np.sqrt(residual_variance)))
    if optimize_V == "EM":
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                                    alpha, post_mean2,
                                    check_null_threshold=check_null_threshold)
    return {"alpha": alpha, "mu": post_mean, "mu2": post_mean2, "lbf": lbf,
            "lbf_model": float(lbf_model), "V": float(V),
            "loglik": float(loglik_val)}


# ----------------------------------------------------------------------
# Single-effect regression — sufficient statistics
# ----------------------------------------------------------------------
def single_effect_regression_ss(Xty, dXtX, V=1.0, residual_variance=1.0,
                                prior_weights=None, optimize_V="none",
                                check_null_threshold=0.0):
    """Bayesian single-effect regression with sufficient statistics.

    Faithful port of ``single_effect_regression_ss`` in
    ``single_effect_regression_ss.R``.  ``Xty`` is a p-vector, ``dXtX``
    the diagonal of ``X'X``.
    """
    Xty = np.asarray(Xty, dtype=float).ravel()
    dXtX = np.asarray(dXtX, dtype=float).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        betahat = (1.0 / dXtX) * Xty
        shat2 = residual_variance / dXtX
    p = dXtX.shape[0]
    if prior_weights is None:
        prior_weights = np.full(p, 1.0 / p)
    prior_weights = np.asarray(prior_weights, dtype=float)

    if optimize_V not in ("EM", "none"):
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                                    alpha=None, post_mean2=None, V_init=V,
                                    check_null_threshold=check_null_threshold)

    lbf = _dnorm_log(betahat, np.sqrt(V + shat2)) - _dnorm_log(betahat, np.sqrt(shat2))
    lpo = lbf + np.log(prior_weights + _SQRT_EPS)
    inf = np.isinf(shat2)
    lbf = np.where(inf, 0.0, lbf)
    lpo = np.where(inf, 0.0, lpo)
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    weighted_sum_w = np.sum(w_weighted)
    alpha = w_weighted / weighted_sum_w
    with np.errstate(invalid="ignore", divide="ignore"):
        post_var = 1.0 / (1.0 / V + dXtX / residual_variance) if V > 0 \
            else np.zeros_like(dXtX)
    post_mean = (1.0 / residual_variance) * post_var * Xty
    post_mean2 = post_var + post_mean ** 2
    lbf_model = maxlpo + np.log(weighted_sum_w)
    if optimize_V == "EM":
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                                    alpha, post_mean2,
                                    check_null_threshold=check_null_threshold)
    return {"alpha": alpha, "mu": post_mean, "mu2": post_mean2, "lbf": lbf,
            "V": float(V), "lbf_model": float(lbf_model)}


# ----------------------------------------------------------------------
# Single-effect regression — RSS (z-scores with lambda regularisation)
# ----------------------------------------------------------------------
def _loglik_rss(V, z, sigma_attrs, prior_weights):
    """Log-likelihood of the RSS SER model (``loglik_rss`` in R)."""
    z = np.asarray(z, dtype=float)
    p = z.shape[0]
    RjSinvRj = sigma_attrs["RjSinvRj"]
    SinvRj = sigma_attrs["SinvRj"]
    with np.errstate(divide="ignore"):
        shat2 = 1.0 / RjSinvRj
    sz = SinvRj.T @ z  # column j: sum(SinvRj[:,j] * z)
    lbf = -0.5 * np.log(1.0 + (V / shat2)) + 0.5 * (V / (1.0 + (V / shat2))) * sz ** 2
    lpo = lbf + np.log(prior_weights + _SQRT_EPS)
    inf = np.isinf(shat2)
    lbf = np.where(inf, 0.0, lbf)
    lpo = np.where(inf, 0.0, lpo)
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    return float(np.log(np.sum(w_weighted)) + maxlpo)


def _neg_loglik_logscale_rss(lV, z, sigma_attrs, prior_weights):
    return -_loglik_rss(np.exp(lV), z, sigma_attrs, prior_weights)


def _optimize_prior_variance_rss(optimize_V, z, sigma_attrs, prior_weights,
                                 alpha=None, post_mean2=None, V_init=None,
                                 check_null_threshold=0.0):
    """Estimate the RSS SER prior variance (``optimize_prior_variance_rss``)."""
    V = V_init
    if optimize_V != "simple":
        if optimize_V == "optim":
            RjSinvRj = sigma_attrs["RjSinvRj"]
            SinvRj = sigma_attrs["SinvRj"]
            sz2 = (SinvRj * z[:, None]).sum(axis=0) ** 2
            with np.errstate(divide="ignore"):
                cand = sz2 - 1.0 / RjSinvRj
            init = np.log(np.nanmax(np.append(cand, 1e-6)))
            res = minimize_scalar(
                _neg_loglik_logscale_rss, bounds=(-30.0, 15.0),
                args=(z, sigma_attrs, prior_weights), method="bounded",
                options={"xatol": 1e-8},
            )
            lV = res.x
            with np.errstate(divide="ignore"):
                logV = np.log(V)
            if (_neg_loglik_logscale_rss(lV, z, sigma_attrs, prior_weights)
                    > _neg_loglik_logscale_rss(logV, z, sigma_attrs,
                                               prior_weights)):
                lV = logV
            V = float(np.exp(lV))
        elif optimize_V == "EM":
            V = float(np.sum(np.asarray(alpha) * np.asarray(post_mean2)))
        else:
            raise ValueError("Invalid option for optimize_V")
    if (_loglik_rss(0.0, z, sigma_attrs, prior_weights) + check_null_threshold
            >= _loglik_rss(V, z, sigma_attrs, prior_weights)):
        V = 0.0
    return float(V)


def single_effect_regression_rss(z, sigma_attrs, V=1.0, prior_weights=None,
                                 optimize_V="none", check_null_threshold=0.0):
    """Bayesian single-effect regression with z-scores (RSS model).

    Faithful port of ``single_effect_regression_rss`` in
    ``single_effect_regression_rss.R``.  ``sigma_attrs`` carries the
    pre-computed ``SinvRj`` and ``RjSinvRj`` matrices/vectors.
    """
    z = np.asarray(z, dtype=float).ravel()
    p = z.shape[0]
    RjSinvRj = sigma_attrs["RjSinvRj"]
    SinvRj = sigma_attrs["SinvRj"]
    with np.errstate(divide="ignore"):
        shat2 = 1.0 / RjSinvRj
    if prior_weights is None:
        prior_weights = np.full(p, 1.0 / p)
    prior_weights = np.asarray(prior_weights, dtype=float)

    if optimize_V not in ("EM", "none"):
        V = _optimize_prior_variance_rss(optimize_V, z, sigma_attrs,
                                         prior_weights, V_init=V,
                                         check_null_threshold=check_null_threshold)

    sz = SinvRj.T @ z
    lbf = -0.5 * np.log(1.0 + (V / shat2)) + 0.5 * (V / (1.0 + (V / shat2))) * sz ** 2
    lpo = lbf + np.log(prior_weights + _SQRT_EPS)
    inf = np.isinf(shat2)
    lbf = np.where(inf, 0.0, lbf)
    lpo = np.where(inf, 0.0, lpo)
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    weighted_sum_w = np.sum(w_weighted)
    alpha = w_weighted / weighted_sum_w
    post_var = 1.0 / (RjSinvRj + 1.0 / V) if V > 0 else np.zeros_like(RjSinvRj)
    post_mean = post_var * sz
    post_mean2 = post_var + post_mean ** 2
    lbf_model = maxlpo + np.log(weighted_sum_w)
    if optimize_V == "EM":
        V = _optimize_prior_variance_rss(optimize_V, z, sigma_attrs,
                                         prior_weights, alpha, post_mean2,
                                         check_null_threshold=check_null_threshold)
    return {"alpha": alpha, "mu": post_mean, "mu2": post_mean2, "lbf": lbf,
            "V": float(V), "lbf_model": float(lbf_model)}
