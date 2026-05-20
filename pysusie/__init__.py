"""pysusie: Pure-Python port of the R/CRAN package susieR.

A standalone, dependency-light implementation of the "Sum of Single
Effects" (SuSiE) regression model for Bayesian variable selection and
genetic fine-mapping (Wang, Sarkar, Carbonetto & Stephens, *J. R. Stat.
Soc. B* 2020, 82:1273-1300; Zou et al., *PLoS Genetics* 2022).  It does
not require R or rpy2 -- only numpy, scipy and pandas.

The SuSiE model fits ``y = mu + Xb + e`` where ``b`` is a sum of ``L``
single-effect vectors, each with exactly one non-zero element.  The
fitting algorithm is Iterative Bayesian Stepwise Selection (IBSS), a
coordinate-ascent variational scheme over ``L`` Bayesian single-effect
regressions.

Entry points
------------
* :func:`susie` -- fine-mapping with individual-level ``X`` and ``y`` via
  the IBSS algorithm.
* :func:`susie_suff_stat` -- fine-mapping from sufficient statistics
  ``X'X``, ``X'y``, ``y'y`` and ``n``.
* :func:`susie_rss` -- fine-mapping from summary statistics (z-scores or
  ``bhat``/``shat``) and an LD matrix ``R``; the most-used entry point in
  statistical genetics.
* :func:`susie_rss_lambda` -- the lambda-regularised RSS IBSS.

Sufficient statistics & univariate helpers
------------------------------------------
* :func:`compute_suff_stat` / :func:`compute_ss` -- compute ``X'X``,
  ``X'y``, ``y'y``.
* :func:`univariate_regression` -- per-column simple linear regression.

Inference helpers (the ``susie_get_*`` family)
----------------------------------------------
* :func:`susie_get_cs` -- credible sets with purity filtering.
* :func:`susie_get_pip` -- posterior inclusion probabilities.
* :func:`susie_get_objective`, :func:`susie_get_posterior_mean`,
  :func:`susie_get_posterior_sd`, :func:`susie_get_lfsr`,
  :func:`susie_get_niter`, :func:`susie_get_prior_variance`,
  :func:`susie_get_residual_variance`, :func:`susie_get_posterior_samples`,
  :func:`get_cs_correlation`.

Prediction & utilities
----------------------
* :func:`coef_susie`, :func:`predict_susie` -- coefficients / predictions.
* :func:`susie_init_coef`, :func:`susie_prune_single_effects`.

Diagnostics & plotting
----------------------
* :func:`estimate_s_rss`, :func:`kriging_rss` -- LD-mismatch diagnostics.
* :func:`susie_plot` -- PIP / z-score Manhattan plot with credible sets.

Single-effect core
------------------
* :func:`single_effect_regression`, :func:`single_effect_regression_ss`,
  :func:`single_effect_regression_rss`.

Quick start
-----------
>>> import numpy as np, pysusie as ps
>>> rng = np.random.default_rng(1)
>>> X = rng.standard_normal((400, 200))
>>> beta = np.zeros(200); beta[[10, 50, 120]] = 1.0
>>> y = X @ beta + rng.standard_normal(400)
>>> fit = ps.susie(X, y, L=10)
>>> fit.sets["cs"]            # credible sets
>>> fit.pip                   # posterior inclusion probabilities
"""
from __future__ import annotations

from ._model import SusieFit
from .ser import (
    single_effect_regression,
    single_effect_regression_ss,
    single_effect_regression_rss,
)
from .univariate import (
    univariate_regression,
    calc_z,
    compute_suff_stat,
    compute_ss,
)
from .susie import (
    susie,
    susie_init_coef,
    susie_prune_single_effects,
    coef_susie,
    predict_susie,
)
from .suff_stat import susie_suff_stat
from .rss import (
    susie_rss,
    susie_rss_lambda,
    estimate_s_rss,
    kriging_rss,
)
from .getters import (
    susie_get_objective,
    susie_get_posterior_mean,
    susie_get_posterior_sd,
    susie_get_niter,
    susie_get_prior_variance,
    susie_get_residual_variance,
    susie_get_lfsr,
    susie_get_posterior_samples,
    susie_get_cs,
    susie_get_pip,
    get_cs_correlation,
    n_in_CS,
    in_CS,
)
from .plots import susie_plot

__version__ = "0.1.0"

__all__ = [
    # core result object
    "SusieFit",
    # main entry points
    "susie",
    "susie_suff_stat",
    "susie_rss",
    "susie_rss_lambda",
    # sufficient statistics / univariate
    "compute_suff_stat",
    "compute_ss",
    "univariate_regression",
    "calc_z",
    # single-effect regression core
    "single_effect_regression",
    "single_effect_regression_ss",
    "single_effect_regression_rss",
    # inference helpers
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
    # prediction / utilities
    "coef_susie",
    "predict_susie",
    "susie_init_coef",
    "susie_prune_single_effects",
    # diagnostics
    "estimate_s_rss",
    "kriging_rss",
    # plotting
    "susie_plot",
]
