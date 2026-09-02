"""The comparison figures, in the palette used throughout the thesis.

C_BASE = baseline arms (length, surface stats), C_V1 = V1 categorical
features, C_V2 = V2 granular features, C_REF = reference ceiling (TF-IDF).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
C_BASE, C_V1, C_V2, C_REF = "#9a9892", "#eb6834", "#2a78d6", "#1baf7a"


def _colour_of(name: str) -> str:
    if name.startswith("V2"):
        return C_V2
    if name.startswith("V1"):
        return C_V1
    if name.startswith("TF-IDF"):
        return C_REF
    return C_BASE


def plot_auc_comparison(table, baseline_model_name: str, title: str, out_path: Path):
    """`table` needs columns: model, AUC, CI_low, CI_high."""
    t = table.sort_values("AUC")
    fig, ax = plt.subplots(figsize=(9, 0.52 * len(t) + 1.4))
    ypos = np.arange(len(t))
    cols = [_colour_of(n) for n in t.model]

    ax.barh(ypos, t.AUC, height=0.62, color=cols, zorder=3)
    err = np.vstack([t.AUC - t.CI_low.fillna(t.AUC), t.CI_high.fillna(t.AUC) - t.AUC])
    ax.errorbar(t.AUC, ypos, xerr=err, fmt="none", ecolor=INK, elinewidth=1.6,
                capsize=3, alpha=0.65, zorder=4)

    base_rows = table.loc[table.model == baseline_model_name, "AUC"]
    if len(base_rows):
        base_auc = float(base_rows.iloc[0])
        ax.axvline(base_auc, color=INK, ls="--", lw=1.6, alpha=0.8, zorder=2)
        ax.text(base_auc + 0.006, len(t) - 0.45, "baseline", color=INK, fontsize=9,
                va="center", ha="left")

    for i, (v, hi) in enumerate(zip(t.AUC, t.CI_high.fillna(t.AUC))):
        ax.text(hi + 0.014, i, f"{v:.3f}", va="center", fontsize=9, color=INK, zorder=6)

    ax.set_yticks(ypos)
    ax.set_yticklabels(t.model, fontsize=10, color=INK)
    ax.set_xlim(0.45, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Cross-validated ROC-AUC  (10x5-fold, bars = 95% bootstrap CI)",
                  fontsize=10, color=MUTED)
    ax.set_title(title, fontsize=12.5, color=INK, pad=12, loc="left")
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_BASE, C_V1, C_V2, C_REF)]
    ax.legend(handles, ["baseline", "V1 features", "V2 features", "reference ceiling"],
              loc="lower right", frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def plot_top_feature_distributions(lab, uni, top_n: int, out_path: Path):
    """`lab` needs a `label` column and one column per feature; `uni` needs
    columns `feature` and `AUC` (per-feature univariate AUC)."""
    top = (uni.assign(d=(uni.AUC - 0.5).abs())
              .sort_values("d", ascending=False).feature.head(top_n).tolist())
    ncols = 3
    nrows = -(-top_n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3 * nrows))
    for axi, c in zip(np.array(axes).ravel(), top):
        a, b = lab.loc[lab.label == 0, c], lab.loc[lab.label == 1, c]
        bp = axi.boxplot([a, b], patch_artist=True, widths=0.55, showfliers=False,
                         medianprops=dict(color=INK, lw=2))
        for patch, col in zip(bp["boxes"], (C_V2, C_V1)):
            patch.set_facecolor(col)
            patch.set_edgecolor("white")
            patch.set_linewidth(2)
        axi.set_xticklabels(["negative", "positive"], fontsize=9, color=MUTED)
        axi.set_title(f"{c}\nAUC {uni.loc[uni.feature == c, 'AUC'].iloc[0]:.3f}",
                      fontsize=9.5, color=INK)
        axi.grid(axis="y", color=GRID, lw=1)
        axi.set_axisbelow(True)
        for s in ("top", "right"):
            axi.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            axi.spines[s].set_color(GRID)
        axi.tick_params(colors=MUTED, length=0)
    fig.suptitle("Most discriminative features", fontsize=12.5, color=INK, x=0.02, ha="left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig
