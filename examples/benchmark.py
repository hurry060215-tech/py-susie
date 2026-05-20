"""Head-to-head speed + accuracy benchmark: R susieR vs pysusie.

Runs SuSiE fine-mapping -- ``susie`` (individual-level data) and
``susie_rss`` (summary statistics) -- on susieR's own bundled
``N3finemapping`` benchmark (574 samples; a 303-SNP window containing all
three true effects), so both languages analyse identical input.

Reports, per entry point:

* wall-clock time (Python via ``time.perf_counter``; R via ``Rscript``).
* accuracy of the Python output vs R: PIP Pearson r, max abs PIP diff,
  per-effect ``alpha`` diff, ELBO / sigma2 diffs and credible-set
  agreement.

Usage::

    python examples/benchmark.py --runs 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pysusie as ps

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
R_DRIVER = HERE.parent / "tests" / "r_reference_driver.R"


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def run_R(out_dir: Path):
    """Run the susieR R reference driver once; return wall time."""
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    t0 = time.perf_counter()
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"R driver failed:\n{res.stderr[-2000:]}")
    return time.perf_counter() - t0


def run_python_susie(X, y, runs):
    """Time ``susie`` ``runs`` times; return mean time + the last fit."""
    elapsed = []
    fit = None
    for _ in range(runs):
        t0 = time.perf_counter()
        fit = ps.susie(X, y, L=10)
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed)), fit


def run_python_rss(z, R, n, runs):
    """Time ``susie_rss`` ``runs`` times; return mean time + the last fit."""
    elapsed = []
    fit = None
    for _ in range(runs):
        t0 = time.perf_counter()
        fit = ps.susie_rss(z=z, R=R, n=n, L=10)
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed)), fit


def _cs_sets(cs_df):
    if len(cs_df) == 0:
        return set()
    return {frozenset(g["variable"].tolist())
            for _, g in cs_df.groupby("cs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    r_out = WORK / "R_out"

    print("Dataset: susieR bundled N3finemapping (574 samples, 303-SNP window)")

    print("\n--- R pipeline (single run) ---")
    r_time = run_R(r_out)
    print(f"  R susieR total      {r_time * 1000:9.1f} ms (incl. Rscript startup)")

    X = pd.read_csv(r_out / "data_X.tsv", sep="\t", header=None).to_numpy()
    y = pd.read_csv(r_out / "data_y.tsv", sep="\t")["y"].to_numpy()
    n = X.shape[0]
    z = pd.read_csv(r_out / "rss_z.tsv", sep="\t")["z"].to_numpy()
    R = pd.read_csv(r_out / "rss_R.tsv", sep="\t", header=None).to_numpy()

    print(f"\n--- Python pipeline (mean of {args.runs} runs) ---")
    susie_time, fit = run_python_susie(X, y, args.runs)
    rss_time, fit_rss = run_python_rss(z, R, n, args.runs)
    print(f"  pysusie susie       {susie_time * 1000:9.1f} ms "
          f"({X.shape[0]} x {X.shape[1]})")
    print(f"  pysusie susie_rss   {rss_time * 1000:9.1f} ms")

    summary = {"shape": list(X.shape),
               "r_time_ms": r_time * 1000,
               "py_susie_time_ms": susie_time * 1000,
               "py_rss_time_ms": rss_time * 1000}

    print("\n--- Accuracy: susie (Python vs R) ---")
    r_pip = pd.read_csv(r_out / "susie_pip.tsv", sep="\t")["pip"].to_numpy()
    r_alpha = pd.read_csv(r_out / "susie_alpha.tsv", sep="\t",
                          header=None).to_numpy()
    r_sc = pd.read_csv(r_out / "susie_scalars.tsv", sep="\t")
    pip_r = _pearson(fit.pip, r_pip)
    print(f"  PIP Pearson r        = {pip_r:.10f}")
    print(f"  PIP max abs diff     = {np.max(np.abs(fit.pip - r_pip)):.3e}")
    print(f"  alpha max abs diff   = {np.max(np.abs(fit.alpha - r_alpha)):.3e}")
    print(f"  ELBO diff            = {abs(fit.elbo[-1] - r_sc['elbo'][0]):.3e}")
    print(f"  sigma2 diff          = {abs(fit.sigma2 - r_sc['sigma2'][0]):.3e}")
    r_cs = _cs_sets(pd.read_csv(r_out / "susie_cs.tsv", sep="\t"))
    py_cs = {frozenset((c + 1).tolist()) for c in fit.sets["cs"]}
    print(f"  credible sets match  = {py_cs == r_cs}")
    summary.update(susie_pip_pearson_r=pip_r,
                   susie_pip_max_diff=float(np.max(np.abs(fit.pip - r_pip))),
                   susie_cs_match=bool(py_cs == r_cs))

    print("\n--- Accuracy: susie_rss (Python vs R) ---")
    r_pip = pd.read_csv(r_out / "rss_pip.tsv", sep="\t")["pip"].to_numpy()
    r_alpha = pd.read_csv(r_out / "rss_alpha.tsv", sep="\t",
                          header=None).to_numpy()
    r_sc = pd.read_csv(r_out / "rss_scalars.tsv", sep="\t")
    pip_r = _pearson(fit_rss.pip, r_pip)
    print(f"  PIP Pearson r        = {pip_r:.10f}")
    print(f"  PIP max abs diff     = {np.max(np.abs(fit_rss.pip - r_pip)):.3e}")
    print(f"  alpha max abs diff   = {np.max(np.abs(fit_rss.alpha - r_alpha)):.3e}")
    print(f"  ELBO diff            = {abs(fit_rss.elbo[-1] - r_sc['elbo'][0]):.3e}")
    r_cs = _cs_sets(pd.read_csv(r_out / "rss_cs.tsv", sep="\t"))
    py_cs = {frozenset((c + 1).tolist()) for c in fit_rss.sets["cs"]}
    print(f"  credible sets match  = {py_cs == r_cs}")
    summary.update(rss_pip_pearson_r=pip_r,
                   rss_pip_max_diff=float(np.max(np.abs(fit_rss.pip - r_pip))),
                   rss_cs_match=bool(py_cs == r_cs))

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report -> {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
