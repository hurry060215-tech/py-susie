"""SuSiE plotting — ``susie_plot`` (PIP / z-score Manhattan plots).

Faithful port of susieR's ``susie_plot`` from ``susie_plots.R``, rendered
with matplotlib instead of base R graphics.  Produces a per-variable
scatter (PIP, log10 PIP, z-score or -log10 p) with the variables in each
credible set highlighted, and an optional legend describing CS size and
purity.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm

from .getters import in_CS, n_in_CS

__all__ = ["susie_plot"]

# The 23-colour palette used by susieR's susie_plot.
_COLORS = [
    "#1C86EE",  # dodgerblue2
    "#008B00",  # green4
    "#6A3D9A",  # purple
    "#FF7F00",  # orange
    "#FFD700",  # gold1
    "#7EC0EE",  # skyblue2
    "#FB9A99",  # lt pink
    "#90EE90",  # palegreen2
    "#CAB2D6",  # lt purple
    "#FDBF6F",  # lt orange
    "#B3B3B3",  # gray70
    "#EEE685",  # khaki2
    "#B03060",  # maroon
    "#FF83FA",  # orchid1
    "#FF1493",  # deeppink1
    "#0000FF",  # blue1
    "#36648B",  # steelblue4
    "#00CED1",  # darkturquoise
    "#00FF00",  # green1
    "#8B8B00",  # yellow4
    "#CDCD00",  # yellow3
    "#8B4500",  # darkorange4
    "#A52A2A",  # brown
]


def susie_plot(model, y="PIP", add_bar=False, pos=None, b=None, max_cs=400,
               add_legend=None, ax=None, **kwargs):
    """Per-variable summary plot of a SuSiE fit.

    Parameters
    ----------
    model : SusieFit or array-like
        A fitted SuSiE model, or a raw vector of PIPs / z-scores.
    y : {"PIP", "log10PIP", "z", "z_original"}
        What to plot.  ``"z"`` plots -log10 p-values derived from
        z-scores.
    add_bar : bool
        Draw a stem from zero up to each credible-set variable.
    pos : array-like, optional
        Indices (0-based) of a subset of variables to plot.
    b : array-like, optional
        True effect vector; non-zero entries are highlighted in red.
    max_cs : float
        Maximum CS to display (a purity threshold if < 1, a size limit
        if > 1).
    add_legend : bool or str, optional
        Add a legend describing each CS; a string sets the location.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure/axes is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    import matplotlib.pyplot as plt

    is_susie = isinstance(model, dict) and "alpha" in model
    ylab = y

    if y == "z":
        if is_susie:
            if model.get("z") is None:
                raise ValueError("z-scores not available; set "
                                 "compute_univariate_zscore=True")
            zneg = -np.abs(np.asarray(model["z"], dtype=float))
        else:
            zneg = -np.abs(np.asarray(model, dtype=float))
        p = -np.log10(2 * _norm.cdf(zneg))
        ylab = "-log10(p)"
    elif y == "z_original":
        if is_susie:
            if model.get("z") is None:
                raise ValueError("z-scores not available; set "
                                 "compute_univariate_zscore=True")
            p = np.asarray(model["z"], dtype=float)
        else:
            p = np.asarray(model, dtype=float)
        ylab = "z score"
    elif y == "PIP":
        p = np.asarray(model["pip"] if is_susie else model, dtype=float)
    elif y == "log10PIP":
        p = np.log10(np.asarray(model["pip"] if is_susie else model,
                                dtype=float))
        ylab = "log10(PIP)"
    else:
        if is_susie:
            raise ValueError("Need z_original, z, PIP or log10PIP for "
                             "SuSiE fits")
        p = np.asarray(model, dtype=float)

    p = np.asarray(p, dtype=float)
    if b is None:
        b = np.zeros(len(p))
    else:
        b = np.asarray(b, dtype=float)
    if pos is None:
        pos = np.arange(len(p))
    else:
        pos = np.asarray(pos, dtype=int)

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 4)))

    ax.scatter(pos, p[pos], s=kwargs.pop("s", 16),
               c=kwargs.pop("color", "black"),
               marker=kwargs.pop("marker", "o"), zorder=2)
    ax.set_xlabel(kwargs.pop("xlab", "variable"))
    ax.set_ylabel(kwargs.pop("ylab", ylab))

    colors = list(_COLORS)
    legend_entries = []
    sets = model.get("sets") if is_susie else None
    if is_susie and isinstance(sets, dict) and sets.get("cs"):
        cs_list = sets["cs"]
        cs_index = sets.get("cs_index")
        purity = sets.get("purity")
        req_cov = sets.get("requested_coverage", 0.95)
        L = model["alpha"].shape[0]
        n_cs = n_in_CS(model, req_cov)
        membership = in_CS(model, req_cov)
        # iterate effects in reverse, like R.
        for i in reversed(range(L)):
            if cs_index is not None and i not in list(cs_index):
                continue
            if cs_index is not None:
                which = list(cs_index).index(i)
            else:
                which = None
            pur = (purity["min.abs.corr"][which]
                   if (purity is not None and which is not None
                       and "min.abs.corr" in purity) else np.nan)
            if (purity is not None and which is not None and max_cs < 1
                    and pur >= max_cs):
                x0 = np.intersect1d(pos, cs_list[which])
            elif n_cs[i] < max_cs:
                x0 = np.intersect1d(pos, np.where(membership[i] > 0)[0])
            else:
                x0 = None
            if x0 is None or len(x0) == 0:
                continue
            y1 = p[x0]
            col = colors[0]
            if add_bar:
                ax.vlines(x0, np.zeros(len(x0)), y1, color="gray",
                          linewidth=1.5, zorder=1)
            ax.scatter(x0, y1, s=80, facecolors="none", edgecolors=col,
                       linewidths=2.0, zorder=3)
            colors = colors[1:] + colors[:1]
            legend_entries.insert(0, (col, len(x0),
                                      round(pur, 4) if np.isfinite(pur)
                                      else None))
        if legend_entries and add_legend is not None and add_legend is not False:
            labels = []
            handles = []
            from matplotlib.lines import Line2D
            for k, (col, size, pur) in enumerate(legend_entries):
                if size == 1:
                    txt = f"L{k + 1}: C=1"
                else:
                    txt = f"L{k + 1}: C={size}/R={pur}"
                labels.append(txt)
                handles.append(Line2D([0], [0], marker="s", color="none",
                                      markerfacecolor=col, markersize=8))
            loc = add_legend if isinstance(add_legend, str) else "upper right"
            loc = loc.replace("topright", "upper right") \
                     .replace("topleft", "upper left") \
                     .replace("bottomright", "lower right") \
                     .replace("bottomleft", "lower left")
            ax.legend(handles, labels, loc=loc, fontsize=7, frameon=False)

    nz = np.where(b[pos] != 0)[0]
    if len(nz):
        ax.scatter(pos[nz], p[pos][nz], s=18, c="red", marker="o", zorder=4)
    return ax
