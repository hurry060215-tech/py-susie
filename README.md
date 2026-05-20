# py-susie

Pure-Python port of the R/CRAN package
**[susieR](https://github.com/stephenslab/susieR)** — the "Sum of Single
Effects" (SuSiE) regression model for Bayesian variable selection and
genetic fine-mapping (Wang, Sarkar, Carbonetto & Stephens,
*J. R. Stat. Soc. B* 2020, 82:1273-1300; Zou, Carbonetto, Wang &
Stephens, *PLoS Genetics* 2022, 18:e1010299).

`pysusie` is a standalone, dependency-light implementation of susieR's
**complete computational core**. It does not require R or rpy2 — only
`numpy`, `scipy`, `pandas` and `matplotlib`.

| | |
|---|---|
| PyPI / import name | `pysusie` |
| License | BSD 3-Clause (same as upstream susieR) |
| Upstream | susieR 0.14.2 (CRAN) |

## What is SuSiE?

SuSiE fits the sparse linear regression `y = mu + Xb + e` under the
"susie assumption" that `b` is a **sum of L single effects**, each a
vector with exactly one non-zero element. The fitting algorithm,
**Iterative Bayesian Stepwise Selection (IBSS)**, is a fast coordinate
ascent over `L` Bayesian single-effect regressions. SuSiE summarises
uncertainty with **credible sets** — small groups of variables, one of
which is (with high probability) the true effect — which is especially
useful when variables are highly correlated, as in genetic fine-mapping.

## Install

```bash
pip install pysusie            # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`.

## Quick start

### Individual-level data

```python
import numpy as np
import pysusie as ps

rng = np.random.default_rng(1)
X = rng.standard_normal((400, 200))
beta = np.zeros(200); beta[[10, 50, 120]] = 1.0
y = X @ beta + rng.standard_normal(400)

fit = ps.susie(X, y, L=10)
fit.sets["cs"]                 # credible sets (0-based indices)
fit.pip                        # posterior inclusion probabilities
ps.coef_susie(fit)             # intercept + per-variable effects
ps.susie_plot(fit, "PIP", add_legend=True)
```

### Summary-statistics fine-mapping (genetics)

```python
# z  : p-vector of GWAS z-scores
# R  : p x p LD (correlation) matrix
# n  : sample size
fit = ps.susie_rss(z=z, R=R, n=n, L=10)

# Or from effect estimates and standard errors:
fit = ps.susie_rss(bhat=bhat, shat=shat, R=R, n=n)
```

### Sufficient statistics

```python
ss  = ps.compute_suff_stat(X, y)
fit = ps.susie_suff_stat(ss["XtX"], ss["Xty"], ss["yty"], ss["n"],
                         X_colmeans=ss["X_colmeans"], y_mean=ss["y_mean"])
```

## Public API

* **Fitting** — `susie`, `susie_suff_stat`, `susie_rss`,
  `susie_rss_lambda`.
* **Sufficient statistics / univariate** — `compute_suff_stat`
  (`compute_ss`), `univariate_regression`, `calc_z`.
* **Single-effect core** — `single_effect_regression`,
  `single_effect_regression_ss`, `single_effect_regression_rss`.
* **Inference** — `susie_get_cs`, `susie_get_pip`,
  `susie_get_objective`, `susie_get_posterior_mean`,
  `susie_get_posterior_sd`, `susie_get_lfsr`, `susie_get_niter`,
  `susie_get_prior_variance`, `susie_get_residual_variance`,
  `susie_get_posterior_samples`, `get_cs_correlation`, `n_in_CS`,
  `in_CS`.
* **Prediction / utilities** — `coef_susie`, `predict_susie`,
  `susie_init_coef`, `susie_prune_single_effects`.
* **Diagnostics** — `estimate_s_rss`, `kriging_rss`.
* **Plotting** — `susie_plot`.

## Numerical parity with susieR

`pysusie` is a faithful, line-by-line port of the susieR R source. On
susieR's bundled `N3finemapping` fine-mapping benchmark the Python
results are **numerically identical** to susieR 0.14.2:

| quantity | `susie` | `susie_rss` |
|---|---|---|
| PIP Pearson r | 1.000 (max abs diff ~1e-11) | 1.000 (~1e-9) |
| per-effect `alpha` | max abs diff ~1e-11 | ~1e-9 |
| ELBO | diff ~1e-12 | diff ~1e-12 |
| `sigma2` | diff ~1e-11 | exact |
| credible sets | identical | identical |

See `tests/test_r_parity.py` and `examples/compare_R_vs_Python.ipynb`.

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_smoke.py` runs without R; `tests/test_r_parity.py` compares
against susieR (skipped automatically when R / susieR is unavailable).

## Relationship to upstream

This port mirrors susieR's R source faithfully: `susie.R`,
`susie_rss.R`, `susie_ss.R`, `single_effect_regression*.R`, `elbo*.R`,
`estimate_residual_variance.R`, `susie_utils.R`, `cs`/credible-set logic
and `susie_plots.R`. The IBSS algorithm, ELBO, prior-variance estimation
(EM / `optim` / `uniroot`), credible-set purity filtering and the
`susie_get_*` family are all reproduced.
