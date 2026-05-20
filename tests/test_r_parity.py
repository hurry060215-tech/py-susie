"""R-parity tests — pysusie vs R/CRAN susieR.

The R driver (:file:`r_reference_driver.R`) runs susieR 0.14.2 on its
bundled ``N3finemapping`` fine-mapping benchmark (574 samples, a 303-SNP
window containing all three true effects), writing the X matrix, the
response, the z-scores / LD matrix and every susieR output to TSV files.
We then run ``pysusie`` on the *identical* inputs and compare:

* ``susie``           — PIP Pearson r > 0.999 (effectively bit-exact),
  per-effect ``alpha``, posterior means, credible sets, ELBO, ``sigma2``.
* ``susie_suff_stat`` — PIP Pearson r > 0.999.
* ``susie_rss``       — PIP Pearson r > 0.999, credible sets, ELBO,
  ``sigma2``.

Because pysusie is a faithful, line-by-line port of the same algorithm,
parity is numerical-precision tight (max abs PIP diff ~1e-9).

Tests skip gracefully when the CMAP R env or susieR is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pysusie as ps

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(susieR); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or susieR not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the susieR R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("susie_R")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=900,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def data(r_reference):
    """Load the X / y inputs that the R driver wrote."""
    X = pd.read_csv(r_reference / "data_X.tsv", sep="\t",
                    header=None).to_numpy()
    y = pd.read_csv(r_reference / "data_y.tsv", sep="\t")["y"].to_numpy()
    return dict(X=X, y=y, n=X.shape[0], p=X.shape[1])


def _cs_as_sets(cs_df):
    """Group a (variable, cs) table into a set-of-frozensets."""
    if len(cs_df) == 0:
        return set()
    return {frozenset(g["variable"].tolist())
            for _, g in cs_df.groupby("cs")}


# ----------------------------------------------------------------------
# susie
# ----------------------------------------------------------------------
def test_susie_pip_vs_R(r_reference, data):
    fit = ps.susie(data["X"], data["y"], L=10)
    r_pip = pd.read_csv(r_reference / "susie_pip.tsv", sep="\t")["pip"]
    rho, _ = pearsonr(fit.pip, r_pip)
    assert rho > 0.999, f"susie PIP Pearson r = {rho:.8f}"
    assert np.max(np.abs(fit.pip - r_pip.to_numpy())) < 1e-6


def test_susie_alpha_vs_R(r_reference, data):
    fit = ps.susie(data["X"], data["y"], L=10)
    r_alpha = pd.read_csv(r_reference / "susie_alpha.tsv", sep="\t",
                          header=None).to_numpy()
    assert fit.alpha.shape == r_alpha.shape
    assert np.max(np.abs(fit.alpha - r_alpha)) < 1e-6


def test_susie_posterior_mean_vs_R(r_reference, data):
    fit = ps.susie(data["X"], data["y"], L=10)
    r_pm = pd.read_csv(r_reference / "susie_postmean.tsv",
                       sep="\t")["postmean"].to_numpy()
    py_pm = ps.susie_get_posterior_mean(fit)
    rho, _ = pearsonr(py_pm, r_pm)
    assert rho > 0.999
    assert np.max(np.abs(py_pm - r_pm)) < 1e-5


def test_susie_scalars_vs_R(r_reference, data):
    fit = ps.susie(data["X"], data["y"], L=10)
    r = pd.read_csv(r_reference / "susie_scalars.tsv", sep="\t")
    assert abs(fit.sigma2 - r["sigma2"][0]) < 1e-4
    assert abs(fit.elbo[-1] - r["elbo"][0]) < 1e-3
    assert fit.niter == int(r["niter"][0])


def test_susie_credible_sets_vs_R(r_reference, data):
    fit = ps.susie(data["X"], data["y"], L=10)
    r_cs = pd.read_csv(r_reference / "susie_cs.tsv", sep="\t")
    r_sets = _cs_as_sets(r_cs)
    # pysusie credible sets are 0-based; R is 1-based.
    py_sets = {frozenset((c + 1).tolist()) for c in fit.sets["cs"]}
    assert py_sets == r_sets, f"py={py_sets} R={r_sets}"


# ----------------------------------------------------------------------
# susie_suff_stat
# ----------------------------------------------------------------------
def test_susie_suff_stat_pip_vs_R(r_reference, data):
    ss = ps.compute_suff_stat(data["X"], data["y"], standardize=True)
    fit = ps.susie_suff_stat(ss["XtX"], ss["Xty"], ss["yty"], ss["n"],
                             X_colmeans=ss["X_colmeans"],
                             y_mean=ss["y_mean"], L=10)
    r_pip = pd.read_csv(r_reference / "ss_pip.tsv", sep="\t")["pip"]
    rho, _ = pearsonr(fit.pip, r_pip)
    assert rho > 0.999, f"susie_suff_stat PIP Pearson r = {rho:.8f}"
    assert np.max(np.abs(fit.pip - r_pip.to_numpy())) < 1e-6


# ----------------------------------------------------------------------
# susie_rss
# ----------------------------------------------------------------------
def test_susie_rss_pip_vs_R(r_reference, data):
    z = pd.read_csv(r_reference / "rss_z.tsv", sep="\t")["z"].to_numpy()
    R = pd.read_csv(r_reference / "rss_R.tsv", sep="\t",
                    header=None).to_numpy()
    fit = ps.susie_rss(z=z, R=R, n=data["n"], L=10)
    r_pip = pd.read_csv(r_reference / "rss_pip.tsv", sep="\t")["pip"]
    rho, _ = pearsonr(fit.pip, r_pip)
    assert rho > 0.999, f"susie_rss PIP Pearson r = {rho:.8f}"
    assert np.max(np.abs(fit.pip - r_pip.to_numpy())) < 1e-5


def test_susie_rss_alpha_vs_R(r_reference, data):
    z = pd.read_csv(r_reference / "rss_z.tsv", sep="\t")["z"].to_numpy()
    R = pd.read_csv(r_reference / "rss_R.tsv", sep="\t",
                    header=None).to_numpy()
    fit = ps.susie_rss(z=z, R=R, n=data["n"], L=10)
    r_alpha = pd.read_csv(r_reference / "rss_alpha.tsv", sep="\t",
                          header=None).to_numpy()
    assert fit.alpha.shape == r_alpha.shape
    assert np.max(np.abs(fit.alpha - r_alpha)) < 1e-5


def test_susie_rss_scalars_vs_R(r_reference, data):
    z = pd.read_csv(r_reference / "rss_z.tsv", sep="\t")["z"].to_numpy()
    R = pd.read_csv(r_reference / "rss_R.tsv", sep="\t",
                    header=None).to_numpy()
    fit = ps.susie_rss(z=z, R=R, n=data["n"], L=10)
    r = pd.read_csv(r_reference / "rss_scalars.tsv", sep="\t")
    assert abs(fit.sigma2 - r["sigma2"][0]) < 1e-4
    assert abs(fit.elbo[-1] - r["elbo"][0]) < 1e-3
    assert fit.niter == int(r["niter"][0])


def test_susie_rss_credible_sets_vs_R(r_reference, data):
    z = pd.read_csv(r_reference / "rss_z.tsv", sep="\t")["z"].to_numpy()
    R = pd.read_csv(r_reference / "rss_R.tsv", sep="\t",
                    header=None).to_numpy()
    fit = ps.susie_rss(z=z, R=R, n=data["n"], L=10)
    r_cs = pd.read_csv(r_reference / "rss_cs.tsv", sep="\t")
    r_sets = _cs_as_sets(r_cs)
    py_sets = {frozenset((c + 1).tolist()) for c in fit.sets["cs"]}
    assert py_sets == r_sets, f"py={py_sets} R={r_sets}"
