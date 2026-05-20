"""Algorithmic smoke tests for pysusie — no R required.

These check internal consistency of every ported routine on synthetic
data with known true effects.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

import pysusie as ps

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# synthetic fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    """A synthetic regression with three sparse, separable effects."""
    rng = np.random.default_rng(0)
    n, p = 400, 150
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    true_idx = np.array([5, 60, 110])
    beta[true_idx] = np.array([1.5, -2.0, 1.0])
    y = X @ beta + rng.standard_normal(n)
    return dict(X=X, y=y, beta=beta, true_idx=true_idx, n=n, p=p)


@pytest.fixture(scope="module")
def correlated():
    """A synthetic regression with two highly correlated true variables."""
    rng = np.random.default_rng(7)
    n, p = 300, 80
    X = rng.standard_normal((n, p))
    # make columns 20 and 21 nearly identical
    X[:, 21] = X[:, 20] + 0.05 * rng.standard_normal(n)
    beta = np.zeros(p)
    beta[20] = 2.0
    beta[50] = 1.5
    y = X @ beta + rng.standard_normal(n)
    return dict(X=X, y=y, beta=beta, n=n, p=p)


# ----------------------------------------------------------------------
# susie
# ----------------------------------------------------------------------
def test_susie_recovers_true_effects(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    assert fit.converged
    assert fit.alpha.shape == (10, synthetic["p"])
    # PIP of each true variable is high.
    for idx in synthetic["true_idx"]:
        assert fit.pip[idx] > 0.9, f"PIP[{idx}] = {fit.pip[idx]}"
    # exactly three credible sets, one per true variable.
    cs = fit.sets["cs"]
    assert cs is not None and len(cs) == 3
    found = sorted(int(c[np.argmax(fit.pip[c])]) for c in cs)
    assert found == sorted(synthetic["true_idx"].tolist())


def test_susie_elbo_monotone(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    elbo = np.asarray(fit.elbo)
    # ELBO is non-decreasing (allowing tiny numerical slack).
    assert np.all(np.diff(elbo) >= -1e-6)
    assert np.isfinite(elbo[-1])


def test_susie_fit_object_fields(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=8)
    for key in ("alpha", "mu", "mu2", "lbf", "lbf_variable", "V", "sigma2",
                "elbo", "pip", "sets", "niter", "converged", "intercept",
                "fitted"):
        assert key in fit, f"missing field {key}"
    assert fit.fitted.shape == (synthetic["n"],)
    assert np.asarray(fit.V).shape == (8,)


def test_susie_coef_and_predict(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    coef = ps.coef_susie(fit)
    assert coef.shape == (synthetic["p"] + 1,)
    pred = ps.predict_susie(fit)
    assert pred.shape == (synthetic["n"],)
    # fitted values track y.
    r = np.corrcoef(pred, synthetic["y"])[0, 1]
    assert r > 0.8
    pred_new = ps.predict_susie(fit, newx=synthetic["X"])
    assert np.allclose(pred_new, pred, atol=1e-8)
    coef2 = ps.predict_susie(fit, type="coefficients")
    assert np.allclose(coef2, coef)


def test_susie_no_intercept(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10, intercept=False)
    assert fit.intercept == 0.0


def test_susie_estimate_prior_methods(synthetic):
    for method in ("optim", "EM", "simple"):
        fit = ps.susie(synthetic["X"], synthetic["y"], L=10,
                       estimate_prior_method=method)
        for idx in synthetic["true_idx"]:
            assert fit.pip[idx] > 0.8


def test_susie_correlated_variables_share_cs(correlated):
    fit = ps.susie(correlated["X"], correlated["y"], L=10)
    cs = fit.sets["cs"]
    assert cs is not None
    # the two correlated true variables (20, 21) live in one CS.
    member = [set(c.tolist()) for c in cs]
    assert any({20, 21}.issubset(m) for m in member)


def test_susie_refine_runs(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10, refine=True)
    assert fit.converged
    for idx in synthetic["true_idx"]:
        assert fit.pip[idx] > 0.8


def test_susie_init_coef(synthetic):
    s = ps.susie_init_coef(synthetic["true_idx"],
                           synthetic["beta"][synthetic["true_idx"]],
                           synthetic["p"])
    assert s["alpha"].shape == (3, synthetic["p"])
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10, s_init=s)
    assert fit.converged


# ----------------------------------------------------------------------
# susie_suff_stat
# ----------------------------------------------------------------------
def test_compute_suff_stat(synthetic):
    ss = ps.compute_suff_stat(synthetic["X"], synthetic["y"])
    assert ss["XtX"].shape == (synthetic["p"], synthetic["p"])
    assert ss["Xty"].shape == (synthetic["p"],)
    assert ss["n"] == synthetic["n"]
    # compute_ss is a synonym.
    ss2 = ps.compute_ss(synthetic["X"], synthetic["y"])
    assert np.allclose(ss["XtX"], ss2["XtX"])


def test_susie_suff_stat_matches_susie(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    ss = ps.compute_suff_stat(synthetic["X"], synthetic["y"])
    fit_ss = ps.susie_suff_stat(ss["XtX"], ss["Xty"], ss["yty"], ss["n"],
                                X_colmeans=ss["X_colmeans"],
                                y_mean=ss["y_mean"], L=10)
    # susie and susie_suff_stat solve the same problem.
    assert np.max(np.abs(fit.pip - fit_ss.pip)) < 1e-6


# ----------------------------------------------------------------------
# susie_rss
# ----------------------------------------------------------------------
def test_susie_rss_with_n(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    z = ur["betahat"] / ur["sebetahat"]
    R = np.corrcoef(synthetic["X"], rowvar=False)
    fit = ps.susie_rss(z=z, R=R, n=synthetic["n"], L=10)
    for idx in synthetic["true_idx"]:
        assert fit.pip[idx] > 0.8
    assert fit.sets["cs"] is not None and len(fit.sets["cs"]) == 3


def test_susie_rss_bhat_shat(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    R = np.corrcoef(synthetic["X"], rowvar=False)
    fit = ps.susie_rss(bhat=ur["betahat"], shat=ur["sebetahat"], R=R,
                       n=synthetic["n"], L=10)
    for idx in synthetic["true_idx"]:
        assert fit.pip[idx] > 0.7


def test_susie_rss_no_n(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    z = ur["betahat"] / ur["sebetahat"]
    R = np.corrcoef(synthetic["X"], rowvar=False)
    fit = ps.susie_rss(z=z, R=R, L=10)
    assert fit.alpha.shape[1] == synthetic["p"]


def test_susie_rss_lambda(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    z = ur["betahat"] / ur["sebetahat"]
    R = np.corrcoef(synthetic["X"], rowvar=False)
    fit = ps.susie_rss_lambda(z, R, L=10)
    for idx in synthetic["true_idx"]:
        assert fit.pip[idx] > 0.7


def test_susie_rss_toy_example():
    # two identical variables with near-equal z-scores share a CS.
    z = np.array([6.0, 6.01])
    R = np.ones((2, 2))
    fit = ps.susie_rss(z=z, R=R)
    assert fit.pip.shape == (2,)
    # second toy example: very different z-scores.
    z2 = np.array([6.0, 7.0])
    fit2 = ps.susie_rss(z=z2, R=R)
    assert fit2.pip[1] >= fit2.pip[0]


# ----------------------------------------------------------------------
# single-effect regression core
# ----------------------------------------------------------------------
def test_single_effect_regression(synthetic):
    from pysusie._linalg import compute_colstats
    X = synthetic["X"]
    out = compute_colstats(X, center=True, scale=True)
    ctx = {"X": X, "cm": out["cm"], "csd": out["csd"], "d": out["d"]}
    y = synthetic["y"] - synthetic["y"].mean()
    res = ps.single_effect_regression(y, ctx, V=1.0, residual_variance=1.0,
                                      optimize_V="optim")
    assert np.isclose(res["alpha"].sum(), 1.0)
    assert res["alpha"].shape == (synthetic["p"],)
    # the SER puts most mass on a true variable.
    assert np.argmax(res["alpha"]) in synthetic["true_idx"]


def test_single_effect_regression_ss():
    rng = np.random.default_rng(3)
    n, p = 200, 40
    X = rng.standard_normal((n, p))
    X = X - X.mean(axis=0)
    beta = np.zeros(p); beta[5] = 2.0
    y = X @ beta + rng.standard_normal(n)
    y = y - y.mean()
    Xty = X.T @ y
    dXtX = np.sum(X ** 2, axis=0)
    res = ps.single_effect_regression_ss(Xty, dXtX, V=1.0,
                                         residual_variance=1.0,
                                         optimize_V="EM")
    assert np.isclose(res["alpha"].sum(), 1.0)
    assert np.argmax(res["alpha"]) == 5


# ----------------------------------------------------------------------
# inference helpers
# ----------------------------------------------------------------------
def test_susie_get_cs_pip_consistency(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    pip = ps.susie_get_pip(fit)
    assert np.allclose(pip, fit.pip)
    cs = ps.susie_get_cs(fit, X=synthetic["X"])
    assert cs["cs"] is not None
    # PIPs are bounded in [0, 1].
    assert np.all(pip >= -1e-9) and np.all(pip <= 1 + 1e-9)


def test_susie_get_methods(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    assert np.isfinite(ps.susie_get_objective(fit))
    assert ps.susie_get_objective(fit, last_only=False).shape == (fit.niter,)
    assert ps.susie_get_niter(fit) == fit.niter
    assert ps.susie_get_residual_variance(fit) == fit.sigma2
    assert np.asarray(ps.susie_get_prior_variance(fit)).shape == (10,)
    pm = ps.susie_get_posterior_mean(fit)
    psd = ps.susie_get_posterior_sd(fit)
    assert pm.shape == (synthetic["p"],)
    assert psd.shape == (synthetic["p"],)
    assert np.all(psd >= -1e-9)
    lfsr = ps.susie_get_lfsr(fit)
    assert lfsr.shape == (10,)


def test_susie_get_posterior_samples(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    samp = ps.susie_get_posterior_samples(fit, num_samples=20,
                                          random_state=0)
    assert samp["b"].shape == (synthetic["p"], 20)
    assert samp["gamma"].shape == (synthetic["p"], 20)


def test_get_cs_correlation(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    cc = ps.get_cs_correlation(fit, X=synthetic["X"])
    # three CSs -> a 3x3 correlation matrix.
    assert cc.shape == (3, 3)
    mx = ps.get_cs_correlation(fit, X=synthetic["X"], max=True)
    assert 0 <= mx <= 1


def test_n_in_cs_and_in_cs(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    sizes = ps.n_in_CS(fit, coverage=0.95)
    membership = ps.in_CS(fit, coverage=0.95)
    assert sizes.shape == (10,)
    assert membership.shape == (10, synthetic["p"])
    for i in range(10):
        assert membership[i].sum() == sizes[i]


# ----------------------------------------------------------------------
# univariate regression / diagnostics
# ----------------------------------------------------------------------
def test_univariate_regression(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    assert ur["betahat"].shape == (synthetic["p"],)
    assert ur["sebetahat"].shape == (synthetic["p"],)
    z = ur["betahat"] / ur["sebetahat"]
    # true variables have large |z|.
    for idx in synthetic["true_idx"]:
        assert abs(z[idx]) > 3


def test_calc_z(synthetic):
    z = ps.calc_z(synthetic["X"], synthetic["y"], center=True)
    assert z.shape == (synthetic["p"],)


def test_estimate_s_rss(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    z = ur["betahat"] / ur["sebetahat"]
    R = np.corrcoef(synthetic["X"], rowvar=False)
    s0 = ps.estimate_s_rss(z, R)
    s1 = ps.estimate_s_rss(z, R, n=synthetic["n"])
    assert 0 <= s0 <= 1.5
    assert 0 <= s1 <= 1.5
    s2 = ps.estimate_s_rss(z, R, n=synthetic["n"],
                           method="null-partialmle")
    assert np.isfinite(s2)


def test_kriging_rss(synthetic):
    ur = ps.univariate_regression(synthetic["X"], synthetic["y"])
    z = ur["betahat"] / ur["sebetahat"]
    R = np.corrcoef(synthetic["X"], rowvar=False)
    res = ps.kriging_rss(z, R, n=synthetic["n"])
    df = res["conditional_dist"]
    assert {"z", "condmean", "condvar", "z_std_diff", "logLR"}.issubset(
        df.columns)
    assert len(df) == synthetic["p"]


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def test_susie_plot_pip(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10)
    ax = ps.susie_plot(fit, "PIP", add_legend=True)
    assert ax is not None
    plt.close("all")


def test_susie_plot_variants(synthetic):
    fit = ps.susie(synthetic["X"], synthetic["y"], L=10,
                   compute_univariate_zscore=True)
    for y in ("PIP", "log10PIP", "z", "z_original"):
        ax = ps.susie_plot(fit, y, b=synthetic["beta"], add_bar=True)
        assert ax is not None
        plt.close("all")
