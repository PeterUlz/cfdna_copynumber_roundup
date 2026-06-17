"""
Genome-wide CNA and focal amplification frequency plots for cBioPortal data.

Note: focal amplification frequency plots for cBioPortal data use the same
function as ichorCNA data — plot_focal_amp_frequencies from ichorcna/plots.py
— so the colour scheme and bar-style rendering are identical across both
data sources.  The old lollipop-style plot previously defined here has been
removed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pandas as pd

from ..common.constants import (
    CHROM_ORDER, CHROM_LENGTHS, GENOME_SIZE, _OFFSETS, ENTITY_PALETTE,
)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure as both PDF and PNG."""
    for fmt in ("pdf", "png"):
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=180, facecolor="white",
                    bbox_inches="tight")
        print(f"      Plot saved: {out}")
    plt.close(fig)


def _chrom_ticks() -> tuple[list[float], list[str]]:
    tick_x = [_OFFSETS[c] + CHROM_LENGTHS[c] / 2 for c in CHROM_ORDER]
    labels  = [c.replace("chr", "") for c in CHROM_ORDER]
    return tick_x, labels


def _style_ax(ax: plt.Axes) -> None:
    """Apply light-theme styling to a genome-wide axes."""
    for ci, chrom in enumerate(CHROM_ORDER):
        x0 = _OFFSETS[chrom]
        x1 = x0 + CHROM_LENGTHS[chrom]
        ax.axvspan(x0, x1,
                   color="#f5f5f5" if ci % 2 == 0 else "white",
                   alpha=1.0, zorder=0)
        ax.axvline(x0, color="#cccccc", linewidth=0.4,
                   linestyle="-", zorder=1)
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.set_xlim(0, GENOME_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.set_facecolor("white")
    tick_x, tick_labels = _chrom_ticks()
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=7, color="#444444")
    ax.tick_params(axis="x", length=3, color="#cccccc")
    ax.tick_params(axis="y", length=2, color="#cccccc", colors="#444444")


# ---------------------------------------------------------------------------
# Broad CNA frequency plot (broad gains/losses — retained)
# ---------------------------------------------------------------------------

def plot_cna_frequencies(
        freq_by_entity: dict[str, pd.DataFrame],
        output_path:    Path,
) -> None:
    """
    Multi-panel genome-wide frequency plot.
    One panel per entity; gains upward, losses downward.
    """
    from matplotlib.gridspec import GridSpec

    entities = list(freq_by_entity.keys())
    n        = len(entities)
    fig = plt.figure(figsize=(22, 4.5 * n + 1.5), facecolor="white")
    gs  = GridSpec(n, 1, figure=fig,
                   hspace=0.55, top=0.93, bottom=0.07, left=0.07, right=0.97)

    for ax_idx, entity in enumerate(entities):
        ax   = fig.add_subplot(gs[ax_idx])
        freq = freq_by_entity[entity]
        pal  = ENTITY_PALETTE.get(entity, {
            "gain": "#e63946", "loss": "#457b9d", "label": entity.title()
        })
        _style_ax(ax)

        for _, row in freq.iterrows():
            chrom = row["chrom"]
            if chrom not in _OFFSETS:
                continue
            x0    = _OFFSETS[chrom] + int(row["bin_start"])
            width = int(row["bin_end"]) - int(row["bin_start"])
            if row["freq_gain"] > 0:
                ax.bar(x0, row["freq_gain"],  width=width, align="edge",
                       color=pal["gain"], alpha=0.85, linewidth=0, zorder=2)
            if row["freq_loss"] > 0:
                ax.bar(x0, -row["freq_loss"], width=width, align="edge",
                       color=pal["loss"], alpha=0.85, linewidth=0, zorder=2)

        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["100","75","50","25","0","25","50","75","100"],
                           fontsize=7, color="#444444")
        ax.set_ylabel("% patients", fontsize=8, color="#444444", labelpad=6)

        n_pts = int(freq["n_total"].iloc[0])
        ax.set_title(f"{pal['label']}   (n = {n_pts})",
                     fontsize=11, fontweight="bold",
                     color="#222222", pad=6, loc="left")
        ax.text(GENOME_SIZE * 1.002, 0.55, "Gain",
                fontsize=8, color=pal["gain"], va="center", ha="left")
        ax.text(GENOME_SIZE * 1.002, -0.55, "Loss",
                fontsize=8, color=pal["loss"], va="center", ha="left")

    fig.suptitle("Genome-Wide Copy-Number Frequencies",
                 fontsize=14, fontweight="bold", color="#222222", y=0.98)
    _save(fig, output_path)