"""The ``SusieFit`` result object, initialisation and ELBO computations.

Ports susieR's ``initialize.R``, ``initialize_rss.R``, ``elbo.R``,
``elbo_ss.R``, ``elbo_rss.R``, ``estimate_residual_variance.R`` and
``remove_null_effects.R``.

A ``SusieFit`` is a lightweight attribute container mirroring the
``"susie"`` S3 list returned by R: ``alpha``, ``mu``, ``mu2``, ``lbf``,
``lbf_variable``, ``V``, ``sigma2``, ``pi``, ``KL``, ``elbo``, ``pip``,
``sets``, ``niter``, ``converged``, ``intercept``, ``fitted`` etc.
"""
from __future__ import annotations

import numpy as np

from ._linalg import compute_Xb, compute_MXt

__all__ = ["SusieFit"]


class SusieFit(dict):
    """A fitted SuSiE model.

    Subclasses ``dict`` so every R list element is accessible both as
    ``fit["alpha"]`` and ``fit.alpha``; this keeps the port close to the
    R object while being convenient in Python.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self):  # pragma: no cover - cosmetic
        L, p = (self["alpha"].shape if "alpha" in self else (0, 0))
        ncs = len(self.get("sets", {}).get("cs") or []) \
            if isinstance(self.get("sets"), dict) else 0
        return (f"SusieFit(L={L}, p={p}, niter={self.get('niter')}, "
                f"converged={self.get('converged')}, n_cs={ncs})")


# ----------------------------------------------------------------------
# Initialisation — individual-level data
# ----------------------------------------------------------------------
def init_setup(n, p, L, scaled_prior_variance, residual_variance,
               prior_weights, null_weight, varY, standardize):
    """Default susie initialisation (``init_setup`` in ``initialize.R``)."""
    spv = np.atleast_1d(np.asarray(scaled_prior_variance, dtype=float))
    if np.any(spv < 0):
        raise ValueError("Scaled prior variance should be positive number")
    if np.any(spv > 1) and standardize:
        raise ValueError("Scaled prior variance should be no greater than 1 "
                         "when standardize = TRUE")
    if residual_variance is None:
        residual_variance = varY
    if prior_weights is None:
        prior_weights = np.full(p, 1.0 / p)
    else:
        prior_weights = np.asarray(prior_weights, dtype=float)
        if np.all(prior_weights == 0):
            raise ValueError("Prior weight should be greater than 0 for at "
                             "least one variable.")
        prior_weights = prior_weights / np.sum(prior_weights)
    if len(prior_weights) != p:
        raise ValueError("Prior weights must have length p")
    if p < L:
        L = p
    s = SusieFit(
        alpha=np.full((L, p), 1.0 / p),
        mu=np.zeros((L, p)),
        mu2=np.zeros((L, p)),
        Xr=np.zeros(n),
        KL=np.full(L, np.nan),
        lbf=np.full(L, np.nan),
        lbf_variable=np.full((L, p), np.nan),
        sigma2=float(residual_variance),
        V=np.asarray(scaled_prior_variance, dtype=float) * varY,
        pi=prior_weights,
    )
    s["null_index"] = 0 if null_weight is None else p
    return s


def init_finalize(s, ctx=None):
    """Finalise a susie object before fitting (``init_finalize`` in R)."""
    V = np.atleast_1d(np.asarray(s["V"], dtype=float))
    L = s["alpha"].shape[0]
    if V.shape[0] == 1:
        V = np.full(L, V[0])
    s["V"] = V
    s["sigma2"] = float(s["sigma2"])
    if s["sigma2"] <= 0:
        raise ValueError("Residual variance sigma2 must be positive")
    if np.any(V < 0):
        raise ValueError("prior variance must be non-negative")
    if s["mu"].shape != s["mu2"].shape:
        raise ValueError("dimension of mu and mu2 do not match")
    if s["mu"].shape != s["alpha"].shape:
        raise ValueError("dimension of mu and alpha do not match")
    if s["alpha"].shape[0] != len(V):
        raise ValueError("Input prior variance V must have length nrow(alpha)")
    if ctx is not None:
        s["Xr"] = compute_Xb(ctx, np.sum(s["mu"] * s["alpha"], axis=0))
    s["KL"] = np.full(L, np.nan)
    s["lbf"] = np.full(L, np.nan)
    return s


# ----------------------------------------------------------------------
# Initialisation — RSS
# ----------------------------------------------------------------------
def init_setup_rss(p, L, prior_variance, residual_variance, prior_weights,
                   null_weight):
    """Default RSS susie initialisation (``init_setup_rss``)."""
    if not np.isscalar(prior_variance) or prior_variance < 0:
        raise ValueError("Prior variance should be positive number.")
    if residual_variance is not None and (residual_variance > 1
                                          or residual_variance < 0):
        raise ValueError("Residual variance should be a scalar between 0 and 1")
    if residual_variance is None:
        residual_variance = 1.0
    if prior_weights is None:
        prior_weights = np.full(p, 1.0 / p)
    else:
        prior_weights = np.asarray(prior_weights, dtype=float)
        if np.all(prior_weights == 0):
            raise ValueError("Prior weight should be greater than 0 for at "
                             "least one variable.")
        prior_weights = prior_weights / np.sum(prior_weights)
    if len(prior_weights) != p:
        raise ValueError("Prior weights must have length p")
    if p < L:
        L = p
    s = SusieFit(
        alpha=np.full((L, p), 1.0 / p),
        mu=np.zeros((L, p)),
        mu2=np.zeros((L, p)),
        Rz=np.zeros(p),
        KL=np.full(L, np.nan),
        lbf=np.full(L, np.nan),
        lbf_variable=np.full((L, p), np.nan),
        sigma2=float(residual_variance),
        V=float(prior_variance),
        pi=prior_weights,
    )
    s["null_index"] = 0 if null_weight is None else p
    return s


def init_finalize_rss(s, R=None):
    """Finalise an RSS susie object (``init_finalize_rss``)."""
    V = np.atleast_1d(np.asarray(s["V"], dtype=float))
    L = s["alpha"].shape[0]
    if V.shape[0] == 1:
        V = np.full(L, V[0])
    s["V"] = V
    s["sigma2"] = float(s["sigma2"])
    if s["sigma2"] <= 0:
        raise ValueError("residual variance sigma2 must be positive")
    if np.any(V < 0):
        raise ValueError("prior variance must be non-negative")
    if s["alpha"].shape[0] != len(V):
        raise ValueError("Input prior variance V must have length nrow(alpha)")
    if R is not None:
        # compute_Xb on R with cm=0, csd=1 -> just R %*% b
        b = np.sum(s["mu"] * s["alpha"], axis=0)
        s["Rz"] = R @ b
    s["KL"] = np.full(L, np.nan)
    s["lbf"] = np.full(L, np.nan)
    return s


# ----------------------------------------------------------------------
# ELBO — individual-level data
# ----------------------------------------------------------------------
def get_objective(ctx, y, s):
    """ELBO of a susie fit (``get_objective`` in ``elbo.R``)."""
    return Eloglik(ctx, y, s) - np.sum(s["KL"])


def Eloglik(ctx, y, s):
    """Expected log-likelihood for a susie fit (``Eloglik``)."""
    n = ctx["X"].shape[0]
    return (-(n / 2.0) * np.log(2 * np.pi * s["sigma2"])
            - (1.0 / (2.0 * s["sigma2"])) * get_ER2(ctx, y, s))


def get_ER2(ctx, y, s):
    """Expected squared residuals (``get_ER2`` in ``elbo.R``)."""
    Xr_L = compute_MXt(ctx, s["alpha"] * s["mu"])  # L x n
    postb2 = s["alpha"] * s["mu2"]
    y = np.asarray(y, dtype=float)
    # R: sum(attr(X,"d") * t(postb2)) recycles d down each column of t(postb2).
    return (np.sum((y - s["Xr"]) ** 2) - np.sum(Xr_L ** 2)
            + np.sum(ctx["d"][:, None] * postb2.T))


def SER_posterior_e_loglik(ctx, y, s2, Eb, Eb2):
    """Posterior expected log-likelihood for a SER (``SER_posterior_e_loglik``)."""
    n = ctx["X"].shape[0]
    y = np.asarray(y, dtype=float)
    return (-0.5 * n * np.log(2 * np.pi * s2)
            - 0.5 / s2 * (np.sum(y * y)
                          - 2 * np.sum(y * compute_Xb(ctx, Eb))
                          + np.sum(ctx["d"] * Eb2)))


def estimate_residual_variance(ctx, y, s):
    """Estimate residual variance (``estimate_residual_variance`` in R)."""
    n = ctx["X"].shape[0]
    return (1.0 / n) * get_ER2(ctx, y, s)


# ----------------------------------------------------------------------
# ELBO — sufficient statistics
# ----------------------------------------------------------------------
def get_objective_ss(XtX, Xty, s, yty, n):
    """ELBO of a susie suff-stat fit (``get_objective_ss``)."""
    return Eloglik_ss(XtX, Xty, s, yty, n) - np.sum(s["KL"])


def Eloglik_ss(XtX, Xty, s, yty, n):
    """Expected log-likelihood for a suff-stat susie fit (``Eloglik_ss``)."""
    return (-n / 2.0 * np.log(2 * np.pi * s["sigma2"])
            - 1.0 / (2.0 * s["sigma2"]) * get_ER2_ss(XtX, Xty, s, yty))


def get_ER2_ss(XtX, Xty, s, yty):
    """Expected squared residuals, suff-stat (``get_ER2_ss``)."""
    B = s["alpha"] * s["mu"]
    XB2 = np.sum((B @ XtX) * B)
    betabar = np.sum(B, axis=0)
    d = s["_d"]
    postb2 = s["alpha"] * s["mu2"]
    return (yty - 2 * np.sum(betabar * Xty)
            + np.sum(betabar * (XtX @ betabar)) - XB2
            + np.sum(d[:, None] * postb2.T))


def SER_posterior_e_loglik_ss(dXtX, Xty, s2, Eb, Eb2):
    """Posterior expected log-likelihood for a SER, suff-stat."""
    dXtX = np.asarray(dXtX, dtype=float)
    Xty = np.asarray(Xty, dtype=float).ravel()
    Eb = np.asarray(Eb, dtype=float).ravel()
    Eb2 = np.asarray(Eb2, dtype=float).ravel()
    return -0.5 / s2 * (-2 * np.sum(Eb * Xty) + np.sum(dXtX * Eb2))


def estimate_residual_variance_ss(XtX, Xty, s, yty, n):
    """Estimate residual variance, suff-stat (``estimate_residual_variance_ss``)."""
    return (1.0 / n) * get_ER2_ss(XtX, Xty, s, yty)


# ----------------------------------------------------------------------
# ELBO — RSS (z-scores + lambda)
# ----------------------------------------------------------------------
def get_objective_rss(R_ctx, z, s):
    """ELBO of an RSS susie fit (``get_objective_rss``)."""
    return Eloglik_rss(s["sigma2"], R_ctx, z, s) - np.sum(s["KL"])


def Eloglik_rss(sigma2, R_ctx, z, s):
    """Expected log-likelihood for an RSS susie fit (``Eloglik_rss``)."""
    eigvals = R_ctx["eigen_values"]
    lam = R_ctx["lambda"]
    d = sigma2 * eigvals + lam
    z = np.asarray(z, dtype=float)
    if lam == 0:
        result = (-(np.sum(d != 0) / 2.0) * np.log(2 * np.pi * sigma2)
                  - 0.5 * get_ER2_rss(sigma2, R_ctx, z, s))
    else:
        result = (-(len(z) / 2.0) * np.log(2 * np.pi)
                  - 0.5 * np.sum(np.log(d))
                  - 0.5 * get_ER2_rss(sigma2, R_ctx, z, s))
    return result


def get_ER2_rss(sigma2, R_ctx, z, s):
    """Expected squared residuals for an RSS fit (``get_ER2_rss``)."""
    eigvals = R_ctx["eigen_values"]
    eigvecs = R_ctx["eigen_vectors"]
    lam = R_ctx["lambda"]
    R = R_ctx["R"]
    z = np.asarray(z, dtype=float)
    d = sigma2 * eigvals + lam
    with np.errstate(divide="ignore"):
        Dinv = 1.0 / d
    Dinv[~np.isfinite(Dinv)] = 0.0
    SinvR = eigvecs @ ((Dinv * eigvals)[:, None] * eigvecs.T)
    Utz = eigvecs.T @ z
    zSinvz = np.sum(Utz * (Dinv * Utz))
    Z = s["alpha"] * s["mu"]
    if lam == 0:
        RSinvR = R / sigma2
    else:
        RSinvR = R @ SinvR
    RZ2 = np.sum((Z @ RSinvR) * Z)
    zbar = np.sum(Z, axis=0)
    postb2 = s["alpha"] * s["mu2"]
    return (zSinvz - 2 * np.sum((SinvR @ z) * zbar)
            + np.sum(zbar * (RSinvR @ zbar))
            - RZ2 + np.sum(np.diag(RSinvR)[:, None] * postb2.T))


def SER_posterior_e_loglik_rss(R, sigma_attrs, r, Ez, Ez2):
    """Posterior expected log-likelihood for an RSS SER."""
    eigS_vals = sigma_attrs["eigenS_values"]
    eigS_vecs = sigma_attrs["eigenS_vectors"]
    with np.errstate(divide="ignore"):
        Dinv = 1.0 / eigS_vals
    Dinv[~np.isfinite(Dinv)] = 0.0
    r = np.asarray(r, dtype=float)
    Ez = np.asarray(Ez, dtype=float).ravel()
    Ez2 = np.asarray(Ez2, dtype=float).ravel()
    rR = R @ r
    SinvEz = eigS_vecs @ (Dinv * (eigS_vecs.T @ Ez))
    return -0.5 * (-2 * np.sum(rR * SinvEz)
                   + np.sum(sigma_attrs["RjSinvRj"] * Ez2))


# ----------------------------------------------------------------------
# Null-effect handling (``remove_null_effects.R``)
# ----------------------------------------------------------------------
def remove_null_effects(s):
    """Drop single-effects with estimated prior variance zero."""
    keep = ~(np.asarray(s["V"]) == 0)
    s["alpha"] = s["alpha"][keep]
    s["mu"] = s["mu"][keep]
    s["mu2"] = s["mu2"][keep]
    if "lbf_variable" in s and s["lbf_variable"] is not None:
        s["lbf_variable"] = s["lbf_variable"][keep]
    s["V"] = np.asarray(s["V"])[keep]
    return s


def add_null_effect(s, V):
    """Append an empty single-effect with prior variance ``V``."""
    p = s["alpha"].shape[1]
    s["alpha"] = np.vstack([s["alpha"], np.full(p, 1.0 / p)])
    s["mu"] = np.vstack([s["mu"], np.zeros(p)])
    s["mu2"] = np.vstack([s["mu2"], np.zeros(p)])
    if "lbf_variable" in s and s["lbf_variable"] is not None:
        s["lbf_variable"] = np.vstack([s["lbf_variable"], np.zeros(p)])
    s["V"] = np.append(np.asarray(s["V"]), V)
    return s
