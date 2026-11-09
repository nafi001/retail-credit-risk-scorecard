"""EDA helpers (Step 2.2). Development sample only."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib as mpl
from statsmodels.stats.proportion import proportion_confint

TARGET = "SeriousDlqin2yrs"
PALETTE = {"main": "#1F4E79", "accent": "#C0504D", "grey": "#7F7F7F", "light": "#9DC3E6", "mid": "#5B9BD5"}

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
        "figure.facecolor": "white", "axes.axisbelow": True})

def band(s, bins, labels, missing_label=None):
    c = pd.cut(s, bins=bins, labels=labels).astype(object)
    cats = list(labels)
    if missing_label is not None:
        c = c.where(c.notna(), missing_label); cats.append(missing_label)
    return pd.Categorical(c, categories=cats, ordered=True)

def rate_table(df, group):
    g = df.groupby(group, observed=False)[TARGET].agg(n="size", defaults="sum")
    g["default_rate"] = g["defaults"] / g["n"].replace(0, np.nan)
    ok = g["n"] > 0
    g["ci_low"], g["ci_high"] = np.nan, np.nan
    lo, hi = proportion_confint(g.loc[ok, "defaults"], g.loc[ok, "n"], method="wilson")
    g.loc[ok, "ci_low"], g.loc[ok, "ci_high"] = lo, hi
    return g.reset_index()

def rate_bar(ax, tab, label_col, title, xlabel, base_rate, rotate=None, highlight=None):
    """`rotate` is kept for compatibility; rotation is chosen automatically from the number of bars."""
    x = np.arange(len(tab)); y = tab["default_rate"].values * 100
    err = np.vstack([(tab["default_rate"] - tab["ci_low"]).values * 100,
                     (tab["ci_high"] - tab["default_rate"]).values * 100])
    cols = [PALETTE["accent"] if (highlight and str(l) in highlight) else PALETTE["main"]
            for l in tab[label_col]]
    ax.bar(x, y, yerr=err, color=cols, capsize=2, width=0.75,
           error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.axhline(base_rate * 100, ls="--", lw=0.9, color=PALETTE["grey"])
    rot = 60 if len(tab) > 6 else 0
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={int(n):,}" for l, n in zip(tab[label_col], tab["n"])],
                       rotation=rot, ha="right" if rot else "center", fontsize=6.5)
    ax.set_ylabel("Default rate (%)"); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.set_ylim(0, np.nanmax(tab["ci_high"].values) * 100 * 1.12)
