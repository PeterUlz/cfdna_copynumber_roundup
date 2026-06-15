"""
Pairwise longitudinal log2-ratio comparison plots for ichorCNA data.

Uses bin-level (unsegmented) log2 ratios from the 50 kbp .cna.seg files,
corrected for tumor fraction, so both samples are on the same scale.

TF correction — simple multiplicative scaling
----------------------------------------------
The mixture model compresses all logR values toward 0 proportionally to TF.
Rather than inverting the full model (which produces -inf for deletions at
low TF), we use a simple multiplier: logR_rescaled = logR * (TF_high / TF_low).
This preserves neutral bins at 0, handles deletions gracefully, and has no
edge cases. Error vs exact inversion: ~0.03-0.07 log2 units for typical TF pairs.


Focal amplification shading
---------------------------
Regions where a focal amplification was detected in ANY sample of the patient
are highlighted as shaded boxes. If a COSMIC CGC gene overlaps the region it
is annotated above the box.
"""
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd

from ..common.constants import CHROM_ORDER, CHROM_LENGTHS, GENOME_SIZE, _OFFSETS
from ..common.io import parse_seg_file
from .plots import (
    ICHORCNA_ENTITY_PALETTE, entity_from_sample,
    _style_ax, _genome_x, _save,
)

# Fixed colour scheme for the "clean" publication-style longitudinal plots.
# Used consistently across all entities (no per-entity palette).
CLEAN_PALETTE = {"gain": "#d7191c", "loss": "#2c7bb6"}


# ---------------------------------------------------------------------------
# TF correction — simple multiplicative scaling
# ---------------------------------------------------------------------------

def correct_logr_relative(logr: float, tf_from: float, tf_to: float) -> float:
    """
    Rescale a logR value from tf_from to tf_to using a simple multiplier:

        logR_rescaled = logR_observed * (tf_to / tf_from)

    Approximates the exact mixture-model inversion with ~0.03-0.07 log2 unit
    error for typical TF pairs, but is robust (no -inf, no edge cases),
    preserves neutral bins at exactly 0, and handles deletions correctly.

    Returns NaN if either TF is None or <= 0.
    Returns logr unchanged if tf_from == tf_to (within 0.1%).
    """
    if tf_from is None or tf_to is None or tf_from <= 0 or tf_to <= 0:
        return float("nan")
    if abs(tf_from - tf_to) < 0.001:
        return float(logr)
    return float(logr) * (tf_to / tf_from)

# ---------------------------------------------------------------------------
# Optimal scaling factor search
# ---------------------------------------------------------------------------

def _optimise_scaling(
        logr_fixed:  np.ndarray,
        logr_adjust: np.ndarray,
        min_abs:     float = 0.05,
) -> tuple[float, float]:
    """
    Find the optimal scale factor m >= 1 for logr_adjust using weighted
    least squares, with weights = |logR_fixed|.

    High-signal bins (large absolute logR) dominate the fit; near-neutral
    bins are down-weighted proportionally so they cannot pull the estimate
    toward 1.0. This is more accurate and faster than a grid search.

    Closed-form solution for m minimising Σ w_i*(lf_i - m*la_i)²:

        m = Σ(w_i * lf_i * la_i) / Σ(w_i * la_i²)
        w_i = |lf_i|

    m is clamped to >= 1.0 (lower-TF sample is always scaled upward).

    Parameters
    ----------
    logr_fixed   : logR array for the fixed (higher-TF) sample.
    logr_adjust  : logR array for the sample to be scaled upward.
    min_abs      : bins where |logR_fixed| < min_abs are excluded as
                   near-neutral noise (default 0.05).

    Returns
    -------
    (best_multiplier, neighbour_noise_rmse)
    """
    mask = (
        np.isfinite(logr_fixed) &
        np.isfinite(logr_adjust) &
        (np.abs(logr_fixed) >= min_abs)
    )
    if mask.sum() < 10:
        # Fall back to all finite bins if too few informative bins
        mask = np.isfinite(logr_fixed) & np.isfinite(logr_adjust)
    if mask.sum() < 10:
        return 1.0, float("nan")

    lf = logr_fixed[mask]
    la = logr_adjust[mask]
    w  = np.abs(lf)   # weights proportional to signal strength

    numerator   = float(np.sum(w * lf * la))
    denominator = float(np.sum(w * la ** 2))

    if denominator <= 0:
        return 1.0, float("nan")

    m    = max(1.0, numerator / denominator)
    rmse = float(np.sqrt(np.mean((lf - la * m) ** 2)))
    return m, rmse


def _neighbour_noise(bins_df: pd.DataFrame) -> float:
    """
    Median absolute difference between consecutive bins on the same chromosome.
    Low for clean samples with flat segments; high for noisy samples.
    """
    diffs = []
    for chrom, grp in bins_df.groupby("chrom"):
        vals  = grp["logr_corrected"].values
        valid = vals[np.isfinite(vals)]
        if len(valid) >= 2:
            diffs.append(np.abs(np.diff(valid)))
    if not diffs:
        return float("nan")
    return float(np.median(np.concatenate(diffs)))


def _signal_metrics(
        bins_df:       pd.DataFrame,
        quantile:      float = 0.95,
        noise_mult:    float = 2.0,
) -> tuple[float, int, float, float]:
    """
    Compute signal quality metrics for a bin-level logR DataFrame.

    Returns
    -------
    (q_val, n_above_noise, neighbour_noise, snr)

    q_val          : *quantile* of absolute logR values (default 95th).
    n_above_noise  : number of bins where |logR| > noise_mult * neighbour_noise.
    neighbour_noise: median absolute consecutive bin difference.
    snr            : q_val / neighbour_noise — signal-to-noise ratio.
                     >= 1.5 indicates meaningful copy-number signal above noise.
    """
    vals = bins_df["logr_corrected"].dropna().values
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), 0, float("nan"), float("nan")

    noise = _neighbour_noise(bins_df)
    if np.isnan(noise) or noise <= 0:
        return float(np.quantile(np.abs(vals), quantile)), 0, float("nan"), float("nan")

    q_val   = float(np.quantile(np.abs(vals), quantile))
    n_above = int(np.sum(np.abs(vals) > noise_mult * noise))
    snr     = q_val / noise
    return q_val, n_above, noise, snr
    """
    Measure technical noise as the median absolute difference between
    consecutive bins on the same chromosome.

    Clean samples with well-defined CNA segments have low neighbour noise
    (bins within a segment are nearly identical). Noisy samples have high
    bin-to-bin variation everywhere, giving a large median difference even
    without true copy-number changes.

    Only within-chromosome differences are computed to avoid chromosome-
    boundary artefacts.
    """
    diffs = []
    for chrom, grp in bins_df.groupby("chrom"):
        vals  = grp["logr_corrected"].values
        valid = vals[np.isfinite(vals)]
        if len(valid) >= 2:
            diffs.append(np.abs(np.diff(valid)))
    if not diffs:
        return float("nan")
    return float(np.median(np.concatenate(diffs)))




# Bin-level data loading from .cna.seg
# ---------------------------------------------------------------------------

def _load_bins(
        seg_path:    Path,
        sample_name: str,
        tf_from:     float,
        tf_to:       float | None = None,
) -> pd.DataFrame | None:
    """
    Load 50 kbp bin-level log2 ratios from a .cna.seg file and apply
    relative TF correction.

    If tf_to is provided (and differs from tf_from), the logR values are
    re-expressed as if observed at tf_to.  Pass tf_to = max(tf_a, tf_b) for
    both samples in a pair so they end up on the same scale.
    If tf_to is None or equal to tf_from no correction is applied.

    Returns DataFrame with columns:
        sample, chrom, start, end, logr_raw, logr_corrected
    or None if the file cannot be parsed or has no logr column.
    """
    df = parse_seg_file(seg_path, sample_name)
    if df is None or df.empty:
        return None

    if "logr" not in df.columns:
        print(f"      [WARN] No log2-ratio column in {seg_path.name} — "
              f"cannot plot {sample_name}.")
        return None

    df = df[df["chrom"].isin(_OFFSETS)].copy()
    df["logr_raw"] = pd.to_numeric(df["logr"], errors="coerce")

    target_tf = tf_to if tf_to is not None else tf_from
    df["logr_corrected"] = df["logr_raw"].apply(
        lambda x: correct_logr_relative(x, tf_from, target_tf)
        if pd.notna(x) else float("nan")
    )
    df = df.dropna(subset=["logr_corrected"])
    df = df[np.isfinite(df["logr_corrected"])]

    return df[["sample", "chrom", "start", "end",
               "logr_raw", "logr_corrected"]].copy()


def _load_bins_with_seg_means(
        seg_path_bins: Path,
        seg_path_segs: Path,
        sample_name:   str,
        tf_from:       float,
) -> pd.DataFrame | None:
    """
    Load 50 kbp bins from a .cna.seg file and replace each bin's raw logR
    with the segment mean logR from the matching .seg.txt segment.

    This reduces noise dramatically: instead of per-bin variance, every bin
    within a segment gets the segment's median logR — a much smoother signal
    that makes the scaling optimisation more stable, especially when one
    sample has lower tumor fraction.

    The segment mean is assigned by overlap: for each bin, the .seg.txt
    segment with the largest overlap is used. Bins with no overlapping
    segment are dropped.

    Returns DataFrame with columns:
        sample, chrom, start, end, logr_raw (bin-level), logr_seg (segment mean),
        logr_corrected (= logr_seg, used downstream for scaling and plotting)
    or None on failure.
    """
    # Load raw bins
    df_bins = parse_seg_file(seg_path_bins, sample_name)
    if df_bins is None or df_bins.empty:
        return None

    logr_col_bins = _get_log2_col_from_df(df_bins)
    if logr_col_bins is None:
        print(f"      [WARN] No logR column in bins file {seg_path_bins.name}")
        return None

    df_bins = df_bins[df_bins["chrom"].isin(_OFFSETS)].copy()
    df_bins["logr_raw"] = pd.to_numeric(df_bins[logr_col_bins], errors="coerce")

    # Load segments
    df_segs = parse_seg_file(seg_path_segs, sample_name)
    if df_segs is None or df_segs.empty:
        # Fall back to raw bin logR if no seg file
        df_bins["logr_seg"]       = df_bins["logr_raw"]
        df_bins["logr_corrected"] = df_bins["logr_raw"].apply(
            lambda x: correct_logr_relative(x, tf_from, tf_from)
            if pd.notna(x) else float("nan")
        )
        return df_bins[["sample", "chrom", "start", "end",
                         "logr_raw", "logr_seg", "logr_corrected"]].copy()

    logr_col_segs = _get_log2_col_from_df(df_segs)
    if logr_col_segs is None:
        df_bins["logr_seg"]       = df_bins["logr_raw"]
        df_bins["logr_corrected"] = df_bins["logr_raw"]
        return df_bins[["sample", "chrom", "start", "end",
                         "logr_raw", "logr_seg", "logr_corrected"]].copy()

    df_segs = df_segs[df_segs["chrom"].isin(_OFFSETS)].copy()
    df_segs["logr_seg_val"] = pd.to_numeric(df_segs[logr_col_segs], errors="coerce")

    # Build per-chrom segment index: list of (start, end, logr)
    seg_index: dict[str, list] = {}
    for _, r in df_segs.iterrows():
        if pd.isna(r["logr_seg_val"]):
            continue
        seg_index.setdefault(r["chrom"], []).append(
            (int(r["start"]), int(r["end"]), float(r["logr_seg_val"]))
        )
    for k in seg_index:
        seg_index[k].sort()

    def _seg_mean_for_bin(chrom: str, bstart: int, bend: int) -> float:
        """Return logR of the segment with maximum overlap with this bin."""
        best_overlap = 0
        best_logr    = float("nan")
        for ss, se, lv in seg_index.get(chrom, []):
            if ss >= bend:
                break  # segments are sorted; no further overlap possible
            overlap = min(se, bend) - max(ss, bstart)
            if overlap > best_overlap:
                best_overlap = overlap
                best_logr    = lv
        return best_logr

    logr_seg_vals = []
    for _, row in df_bins.iterrows():
        logr_seg_vals.append(
            _seg_mean_for_bin(row["chrom"], int(row["start"]), int(row["end"]))
        )

    df_bins["logr_seg"] = logr_seg_vals
    df_bins = df_bins.dropna(subset=["logr_seg"]).copy()

    # logr_corrected = logr_seg (scaling is applied to these later)
    # No TF correction at load time — the optimiser handles the scaling
    df_bins["logr_corrected"] = df_bins["logr_seg"]

    return df_bins[["sample", "chrom", "start", "end",
                     "logr_raw", "logr_seg", "logr_corrected"]].copy()


def _get_log2_col_from_df(df: pd.DataFrame) -> str | None:
    """Return the first recognised log2-ratio column in df."""
    candidates = ["logr", "log2_ratio", "log2ratio", "seg.mean",
                  "median", "logr_copy_ratio"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Focal amp annotation helpers
# ---------------------------------------------------------------------------

def _build_focal_annotations(
        focal_tsv: Path,
        patient:   str,
) -> list[dict]:
    """
    Read focal_amplifications.tsv and return all focal amp regions for this
    patient across any of their samples.

    Returns list of dicts with keys:
        chrom, start, end, cosmic_genes
    """
    if not focal_tsv.exists() or focal_tsv.stat().st_size == 0:
        return []

    try:
        df = pd.read_csv(focal_tsv, sep="\t", low_memory=False)
    except Exception:
        return []

    if "sample" not in df.columns:
        return []

    # Filter to this patient's samples (patient prefix match)
    # sample names follow <patient>_<timepoint> after normalization
    pat_mask = df["sample"].str.startswith(patient + "_") | (df["sample"] == patient)
    pat_df   = df[pat_mask]
    if pat_df.empty:
        return []

    regions = []
    for _, row in pat_df.iterrows():
        chrom = str(row["chrom"])
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        genes = str(row.get("cosmic_genes", "")).strip()
        if genes == "nan":
            genes = ""
        regions.append({
            "chrom":        chrom,
            "start":        int(row["start"]),
            "end":          int(row["end"]),
            "cosmic_genes": genes,
        })
    return regions


def _merge_overlapping(regions: list[dict]) -> list[dict]:
    """
    Merge overlapping focal amp regions on the same chromosome into single
    intervals so the shading is clean.
    """
    if not regions:
        return []
    by_chrom: dict[str, list] = {}
    for r in regions:
        by_chrom.setdefault(r["chrom"], []).append(r)

    merged = []
    for chrom, segs in by_chrom.items():
        segs = sorted(segs, key=lambda x: x["start"])
        cur = segs[0].copy()
        for s in segs[1:]:
            if s["start"] <= cur["end"]:
                cur["end"]  = max(cur["end"], s["end"])
                # Combine gene names
                genes = {g for v in [cur["cosmic_genes"], s["cosmic_genes"]]
                         for g in v.split(",") if g and g != "nan"}
                cur["cosmic_genes"] = ",".join(sorted(genes))
            else:
                merged.append(cur)
                cur = s.copy()
        merged.append(cur)
    return merged


# ---------------------------------------------------------------------------
# Single pairwise plot
# ---------------------------------------------------------------------------

def plot_pairwise_longitudinal(
        sample_a:    str,
        seg_path_a:      Path,
        tf_a:            float,
        sample_b:        str,
        seg_path_b:      Path,
        tf_b:            float,
        output_path:     Path,
        patient:         str = "",
        entity:          str = "",
        timepoint_a:     int | None = None,
        timepoint_b:     int | None = None,
        logr_ylim:       tuple[float, float] = (-2.5, 2.5),
        focal_tsv:       Path | None = None,
        max_noise:       float | None = 0.15,
        min_signal:      float | None = None,
        seg_path_a_segs: Path | None = None,
        seg_path_b_segs: Path | None = None,
) -> None:
    """
    Generate a genome-wide pairwise log2-ratio comparison plot using
    unsegmented 50 kbp bin-level data for display, and segment means
    from the .seg.txt file for the scaling optimisation.

    Parameters
    ----------
    sample_a/b         : sample names
    seg_path_a/b       : paths to 50 kbp .cna.seg bin-level files (for display)
    seg_path_a/b_segs  : paths to 50 kbp .seg.txt segmented files (for scaling)
    tf_a/b             : tumor fractions
    output_path        : stem path (.pdf and .png added)
    patient            : patient ID for title and focal amp lookup
    entity             : entity name for colour selection
    timepoint_a/b      : timepoint labels
    logr_ylim          : y-axis range
    focal_tsv          : path to focal_amplifications.tsv for shading
    """
    pal     = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
    color_a = "#888888"
    color_b = pal["gain"]

    # Load bin-level data for display (raw scatter dots)
    bins_a = _load_bins(seg_path_a, sample_a, tf_from=tf_a)
    bins_b = _load_bins(seg_path_b, sample_b, tf_from=tf_b)

    if bins_a is None or bins_b is None:
        print(f"      [WARN] Could not load bin-level data for "
              f"{sample_a} or {sample_b} — skipping.")
        return

    # Load segment-mean data for scaling optimisation (much less noisy)
    # Falls back to raw bins if .seg.txt files not provided
    if seg_path_a_segs is not None and seg_path_b_segs is not None:
        seg_a = _load_bins_with_seg_means(seg_path_a, seg_path_a_segs,
                                          sample_a, tf_a)
        seg_b = _load_bins_with_seg_means(seg_path_b, seg_path_b_segs,
                                          sample_b, tf_b)
    else:
        seg_a = bins_a
        seg_b = bins_b

    # Align segment-mean bins for scaling optimisation
    aligned = _align_bins(
        seg_a if seg_a is not None else bins_a,
        seg_b if seg_b is not None else bins_b,
    )
    if aligned.empty:
        print(f"      [WARN] No overlapping bins — skipping.")
        return

    # Higher-TF sample is fixed; lower-TF sample is scaled upward.
    # Use segment means (logr_corrected from seg_a/seg_b) for the fit
    # since they are much less noisy than raw bin logR.
    if tf_a >= tf_b:
        # a is fixed (higher TF), b is adjusted (lower TF)
        best_m, best_rmse = _optimise_scaling(
            aligned["logr_a"].values, aligned["logr_b"].values
        )
        bins_b = bins_b.copy()
        bins_b["logr_corrected"] = bins_b["logr_corrected"] * best_m
        adj_label = f"{sample_b} scaled ×{best_m:.2f}"
    else:
        # b is fixed (higher TF), a is adjusted (lower TF)
        best_m, best_rmse = _optimise_scaling(
            aligned["logr_b"].values, aligned["logr_a"].values
        )
        bins_a = bins_a.copy()
        bins_a["logr_corrected"] = bins_a["logr_corrected"] * best_m
        adj_label = f"{sample_a} scaled ×{best_m:.2f}"

    print(f"        Optimal multiplier: {best_m:.3f},  RMSE: {best_rmse:.4f}")

    # Per-sample signal metrics
    q_a, n_above_a, noise_a, snr_a = _signal_metrics(bins_a)
    q_b, n_above_b, noise_b, snr_b = _signal_metrics(bins_b)
    pair_noise = max(
        noise_a if not np.isnan(noise_a) else 0.0,
        noise_b if not np.isnan(noise_b) else 0.0,
    )
    print(f"        {sample_a}: noise={noise_a:.4f}  p95={q_a:.3f}  "
          f"bins_above_2x={n_above_a}")
    print(f"        {sample_b}: noise={noise_b:.4f}  p95={q_b:.3f}  "
          f"bins_above_2x={n_above_b}")
    if max_noise is not None and pair_noise > max_noise:
        print(f"        Pair noise {pair_noise:.4f} > max_noise {max_noise:.4f} "
              f"— skipping plot.")
        return
    if min_signal is not None:
        if snr_a < min_signal or snr_b < min_signal:
            print(f"        SNR ({snr_a:.2f}, {snr_b:.2f}) below "
                  f"min_signal {min_signal:.2f} — skipping plot.")
            return

    # Focal amp regions for shading
    focal_regions: list[dict] = []
    if focal_tsv is not None:
        raw_regions  = _build_focal_annotations(focal_tsv, patient)
        focal_regions = _merge_overlapping(raw_regions)

    fig, ax = plt.subplots(figsize=(22, 5.0), facecolor="white")
    _style_ax(ax)

    # Reference lines
    ax.axhline( 0,    color="#999999", linewidth=0.8, linestyle="-",  zorder=1)
    ax.axhline( 0.58, color="#cccccc", linewidth=0.4, linestyle=":",  zorder=1)  # ~gain
    ax.axhline(-0.58, color="#cccccc", linewidth=0.4, linestyle=":",  zorder=1)  # ~loss

    # Focal amp shading (drawn before data so it sits behind the points)
    for region in focal_regions:
        chrom = region["chrom"]
        if chrom not in _OFFSETS:
            continue
        x0    = _genome_x(chrom, region["start"])
        x1    = _genome_x(chrom, region["end"])
        # Shaded box spanning the full y range
        ax.axvspan(x0, x1, alpha=0.18, color="#b8860b", zorder=2)
        ax.vlines([x0, x1], logr_ylim[0], logr_ylim[1],
                  colors="#ffd166", linewidth=0.6, alpha=0.5, zorder=2)
        # Gene annotation above the plot area
        genes = region["cosmic_genes"]
        if genes:
            gene_label = genes.split(",")[0]
            x_mid = (x0 + x1) / 2
            ax.text(x_mid, logr_ylim[1] * 0.93, gene_label,
                    fontsize=6, color="#b8860b", ha="center", va="bottom",
                    rotation=90, fontweight="bold", zorder=5,
                    clip_on=True)

    # Plot bin-level logR as scatter.
    # Dot size 3 pt is the minimum visible at genome scale; rasterized=True
    # keeps PDF file sizes manageable with ~50k points per sample.
    dot_size = 3.0
    for bins, color, alpha, zorder in [
        (bins_a, color_a, 0.50, 3),
        (bins_b, color_b, 0.75, 4),
    ]:
        # Compute x at bin midpoint
        x_vals = (
            bins.apply(
                lambda r: _genome_x(r["chrom"], (r["start"] + r["end"]) / 2),
                axis=1,
            ).values
        )
        y_vals = bins["logr_corrected"].values
        # Clip for display — extreme values occur at very low TF
        y_vals = np.clip(y_vals, logr_ylim[0], logr_ylim[1])
        ax.scatter(x_vals, y_vals,
                   s=dot_size, c=color, alpha=alpha,
                   linewidths=0, zorder=zorder, rasterized=True)

    # Axes
    ax.set_ylim(logr_ylim)
    ax.set_ylabel("TF-corrected log2 ratio",
                  fontsize=9, color="#444444", labelpad=5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Right-edge labels
    ax.text(GENOME_SIZE * 1.001,  0,    "0",      fontsize=7, color="#888888", va="center")
    ax.text(GENOME_SIZE * 1.001,  0.58, "+gain",  fontsize=6, color="#999999", va="center")
    ax.text(GENOME_SIZE * 1.001, -0.58, "-loss",  fontsize=6, color="#999999", va="center")

    # Title
    tp_a_str   = f"TP{timepoint_a}" if timepoint_a is not None else sample_a
    tp_b_str   = f"TP{timepoint_b}" if timepoint_b is not None else sample_b
    noise_str  = f"{pair_noise:.3f}" if not np.isnan(pair_noise) else "n/a"
    ax.set_title(
        f"{patient}  ({entity})   "
        f"{tp_a_str} (TF={tf_a:.1%})  vs  {tp_b_str} (TF={tf_b:.1%})"
        f"   scale={best_m:.2f}  noise={noise_str}\n"
        f"{tp_a_str}: p95={q_a:.3f}  above2×={n_above_a}    "
        f"{tp_b_str}: p95={q_b:.3f}  above2×={n_above_b}",
        fontsize=10, fontweight="bold", color="#222222", pad=6, loc="left",
    )

    # Legend
    handles = [
        mpatches.Patch(color=color_a,
                       label=f"{tp_a_str}  {sample_a}  (TF={tf_a:.1%})"),
        mpatches.Patch(color=color_b,
                       label=f"{tp_b_str}  {sample_b}  (TF={tf_b:.1%})"),
        plt.Line2D([0], [0], color="#999999", linewidth=1,
                   linestyle="-", label="log2R = 0  (neutral)"),
    ]
    if focal_regions:
        handles.append(
            mpatches.Patch(color="#b8860b", alpha=0.4,
                           label="Focal amp in any sample of this patient")
        )
    ax.legend(handles=handles, fontsize=8, loc="lower left",
              facecolor="white", edgecolor="#cccccc",
              labelcolor="#222222", framealpha=1.0)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Emergent CNA detection from differential track
# ---------------------------------------------------------------------------

def _detect_emergent_cnas(
        diff_df:     pd.DataFrame,
        min_bins:    int   = 3,
        min_delta:   float = 0.2,
        patient:     str   = "",
        sample_a:    str   = "",
        sample_b:    str   = "",
        timepoint_a: "int | None" = None,
        timepoint_b: "int | None" = None,
) -> list[dict]:
    """
    Detect emergent CNA regions in the smoothed differential track.

    A region is reported when >= min_bins *consecutive* bins on the same
    chromosome have |delta_logr| > min_delta.  Consecutive runs with the
    same direction are merged; direction changes start a new region.

    Returns a list of dicts with keys:
        chrom, start, end, n_bins, mean_delta, max_abs_delta, direction,
        patient, sample_a, sample_b, timepoint_a, timepoint_b
    """
    if diff_df.empty or "delta_logr" not in diff_df.columns:
        return []

    regions = []
    for chrom, grp in diff_df.groupby("chrom"):
        grp  = grp.sort_values("start").reset_index(drop=True)
        vals = grp["delta_logr"].values

        run_start  = None
        run_end    = None
        run_vals: list[float] = []
        run_dir    = None

        def _flush_run():
            if run_start is None or len(run_vals) < min_bins:
                return
            regions.append({
                "chrom":          chrom,
                "start":          run_start,
                "end":            run_end,
                "n_bins":         len(run_vals),
                "mean_delta":     round(float(np.mean(run_vals)), 4),
                "max_abs_delta":  round(float(np.max(np.abs(run_vals))), 4),
                "direction":      run_dir,
                "patient":        patient,
                "sample_a":       sample_a,
                "sample_b":       sample_b,
                "timepoint_a":    timepoint_a,
                "timepoint_b":    timepoint_b,
            })

        for i, row in grp.iterrows():
            delta = vals[i]
            if pd.isna(delta) or abs(delta) <= min_delta:
                _flush_run()
                run_start = run_end = run_vals = None
                run_dir   = None
                run_vals  = []
                continue
            direction = "gain" if delta > 0 else "loss"
            if run_start is None:
                run_start = int(row["start"])
                run_end   = int(row["end"])
                run_vals  = [float(delta)]
                run_dir   = direction
            elif direction == run_dir:
                run_end = int(row["end"])
                run_vals.append(float(delta))
            else:
                _flush_run()
                run_start = int(row["start"])
                run_end   = int(row["end"])
                run_vals  = [float(delta)]
                run_dir   = direction
        _flush_run()

    return regions


def _append_emergent_tsv(regions: list[dict], tsv_path: Path) -> None:
    """Append emergent CNA regions to a TSV (creates header on first write)."""
    if not regions:
        return
    df      = pd.DataFrame(regions)
    write_h = not tsv_path.exists() or tsv_path.stat().st_size == 0
    df.to_csv(tsv_path, sep="\t", index=False,
              mode="a" if not write_h else "w",
              header=write_h)


# ---------------------------------------------------------------------------
# Combined two-panel plot: profiles on top, differential below
# ---------------------------------------------------------------------------

def plot_longitudinal_combined(
        sample_a:             str,
        seg_path_a:           Path,
        tf_a:                 float,
        sample_b:             str,
        seg_path_b:           Path,
        tf_b:                 float,
        output_path:          Path,
        patient:              str = "",
        entity:               str = "",
        timepoint_a:          int | None = None,
        timepoint_b:          int | None = None,
        logr_ylim:            tuple[float, float] = (-2.5, 2.5),
        delta_ylim:           tuple[float, float] = (-3.0, 3.0),
        focal_tsv:            Path | None = None,
        max_noise:            float | None = 0.15,
        min_signal:           float | None = None,
        seg_path_a_segs:      Path | None = None,
        seg_path_b_segs:      Path | None = None,
        min_altered_fraction: float = 0.05,
        plot_raw:             bool  = False,
        emergent_min_bins:    int   = 3,
        emergent_min_delta:   float = 0.5,
        emergent_tsv:         Path | None = None,
        clean_plots:          bool  = True,
) -> None:
    """
    Two-panel longitudinal comparison. Scaling is computed on raw bins
    and applied before smoothing, so the smoothed profiles reflect the
    correctly scaled signal.

    Pipeline
    --------
    1. Load raw 50 kbp bin-level logR.
    2. Filter: fraction of raw bins with |logR| > 0.2 must exceed
       min_altered_fraction for both samples (default 5 %).
    3. Compute neighbour-noise filter on raw bins.
    4. Compute TF scaling on raw profiles (using seg means for the fit).
    5. Apply scale to raw bins.
    6. Smooth scaled raw bins → 5-bin and 10-bin versions.
    7. Compute SNR filter on 10-bin-smoothed profiles.
    8. Build 5-bin and 10-bin differentials.
    9. Detect emergent CNA regions (>= emergent_min_bins consecutive bins
       with |delta| > emergent_min_delta).
    10. Output:
          - 5-bin smoothed plot  (always)
          - 10-bin smoothed plot (always)
          - raw plot             (only if plot_raw=True)
          - "clean" variants of the smoothed plots (if clean_plots=True),
            written to <output_dir>/clean/, with no focal-amp shading and
            a fixed gain/loss colour scheme (CLEAN_PALETTE) shared across
            all entities — suitable for compiling into figures.
    """
    pal     = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
    color_a = "#888888"
    color_b = pal["gain"]

    # ── Load raw bins ────────────────────────────────────────────────────────
    bins_a_raw = _load_bins(seg_path_a, sample_a, tf_from=tf_a)
    bins_b_raw = _load_bins(seg_path_b, sample_b, tf_from=tf_b)
    if bins_a_raw is None or bins_b_raw is None:
        print(f"      [WARN] Could not load bin-level data for "
              f"{sample_a} or {sample_b} — skipping.")
        return

    # ── Smoothing helpers ────────────────────────────────────────────────────
    def _smooth(df: pd.DataFrame, window: int,
                col: str = "logr_corrected") -> pd.DataFrame:
        if window <= 1:
            return df
        out      = df.reset_index(drop=True).copy()
        smoothed = np.full(len(out), np.nan)
        half     = window // 2
        for _, grp in out.groupby("chrom"):
            pos  = grp.index.tolist()
            vals = grp[col].values.astype(float)
            n    = len(vals)
            for i in range(n):
                lo     = max(0, i - half)
                hi     = min(n, i + window - half)
                fin    = vals[lo:hi][np.isfinite(vals[lo:hi])]
                smoothed[pos[i]] = float(np.mean(fin)) if len(fin) else np.nan
        out[col] = smoothed
        return out

    # ── Filter 1: altered bin fraction (on raw bins) ────────────────────────
    def _altered_frac(df: pd.DataFrame) -> float:
        v = df["logr_corrected"].dropna().values
        return float(np.mean(np.abs(v) > 0.2)) if len(v) else 0.0

    frac_a = _altered_frac(bins_a_raw)
    frac_b = _altered_frac(bins_b_raw)

    if min_altered_fraction > 0 and (frac_a < min_altered_fraction or
                                      frac_b < min_altered_fraction):
        print(f"        Altered fraction ({frac_a:.3f}, {frac_b:.3f}) "
              f"< {min_altered_fraction:.3f} — skipping.")
        return

    # ── Filter 2: neighbour noise on raw bins ────────────────────────────────
    noise_a    = _neighbour_noise(bins_a_raw)
    noise_b    = _neighbour_noise(bins_b_raw)
    pair_noise = max(noise_a if not np.isnan(noise_a) else 0.0,
                     noise_b if not np.isnan(noise_b) else 0.0)
    if max_noise is not None and pair_noise > max_noise:
        print(f"        Pair noise {pair_noise:.4f} > {max_noise:.4f} — skipping.")
        return

    # ── TF scaling on raw profiles (using seg means for the fit) ─────────────
    if seg_path_a_segs is not None and seg_path_b_segs is not None:
        _seg_a = _load_bins_with_seg_means(seg_path_a, seg_path_a_segs,
                                           sample_a, tf_a)
        _seg_b = _load_bins_with_seg_means(seg_path_b, seg_path_b_segs,
                                           sample_b, tf_b)
        seg_a = _seg_a if _seg_a is not None else bins_a_raw
        seg_b = _seg_b if _seg_b is not None else bins_b_raw
    else:
        seg_a, seg_b = bins_a_raw, bins_b_raw

    aligned = _align_bins(seg_a, seg_b)
    if aligned.empty:
        print("      [WARN] No overlapping bins — skipping.")
        return

    if tf_a >= tf_b:
        best_m, _ = _optimise_scaling(
            aligned["logr_a"].values, aligned["logr_b"].values)
        def _scale(df): out=df.copy(); out["logr_corrected"]*=best_m; return out
        bins_b_raw = _scale(bins_b_raw)
    else:
        best_m, _ = _optimise_scaling(
            aligned["logr_b"].values, aligned["logr_a"].values)
        def _scale(df): out=df.copy(); out["logr_corrected"]*=best_m; return out
        bins_a_raw = _scale(bins_a_raw)

    # ── Smooth scaled raw bins ───────────────────────────────────────────────
    bins_a_s5  = _smooth(bins_a_raw,  5)
    bins_b_s5  = _smooth(bins_b_raw,  5)
    bins_a_s10 = _smooth(bins_a_raw, 10)
    bins_b_s10 = _smooth(bins_b_raw, 10)

    # ── Signal metrics on smoothed bins (for title / SNR filter) ────────────
    q_a, n_above_a, _, snr_a = _signal_metrics(bins_a_s10)
    q_b, n_above_b, _, snr_b = _signal_metrics(bins_b_s10)
    print(f"        {sample_a}: noise={noise_a:.4f}  p95={q_a:.3f}  "
          f"snr={snr_a:.2f}  altered={frac_a:.3f}")
    print(f"        {sample_b}: noise={noise_b:.4f}  p95={q_b:.3f}  "
          f"snr={snr_b:.2f}  altered={frac_b:.3f}")

    if min_signal is not None and (snr_a < min_signal or snr_b < min_signal):
        print(f"        SNR ({snr_a:.2f}, {snr_b:.2f}) < "
              f"{min_signal:.2f} — skipping.")
        return

    # ── Build differentials ──────────────────────────────────────────────────
    def _diff(ba, bb):
        d = _align_bins(ba, bb)
        return d[d["chrom"].isin(_OFFSETS)].reset_index(drop=True)

    diff_raw = _diff(bins_a_raw, bins_b_raw) if plot_raw else pd.DataFrame()
    diff_s5  = _smooth(_diff(bins_a_s5,  bins_b_s5),  5,  col="delta_logr")
    diff_s10 = _smooth(_diff(bins_a_s10, bins_b_s10), 10, col="delta_logr")

    # ── Emergent CNA detection ───────────────────────────────────────────────
    emergent_regions = _detect_emergent_cnas(
        diff_s10,
        min_bins    = emergent_min_bins,
        min_delta   = emergent_min_delta,
        patient     = patient,
        sample_a    = sample_a,
        sample_b    = sample_b,
        timepoint_a = timepoint_a,
        timepoint_b = timepoint_b,
    )
    if emergent_regions and emergent_tsv is not None:
        _append_emergent_tsv(emergent_regions, emergent_tsv)

    # ── Shared strings ───────────────────────────────────────────────────────
    tp_a_str  = f"TP{timepoint_a}" if timepoint_a is not None else sample_a
    tp_b_str  = f"TP{timepoint_b}" if timepoint_b is not None else sample_b
    noise_str = f"{pair_noise:.3f}" if not np.isnan(pair_noise) else "n/a"
    title_l1  = (f"{patient}  ({entity})   "
                 f"{tp_a_str} (TF={tf_a:.1%})  vs  {tp_b_str} (TF={tf_b:.1%})"
                 f"   scale={best_m:.2f}  noise={noise_str}")
    title_l2  = (f"{tp_a_str}: p95={q_a:.3f}  snr={snr_a:.2f}  "
                 f"altered={frac_a:.3f}    "
                 f"{tp_b_str}: p95={q_b:.3f}  snr={snr_b:.2f}  "
                 f"altered={frac_b:.3f}")

    # ── Focal amp regions ────────────────────────────────────────────────────
    focal_regions: list[dict] = []
    if focal_tsv is not None:
        focal_regions = _merge_overlapping(
            _build_focal_annotations(focal_tsv, patient)
        )

    # ── Render helper ────────────────────────────────────────────────────────
    def _render(ba, bb, df_diff, out_path, label, clean=False):
        # Clean plots use a fixed gain/loss colour scheme (no per-entity
        # palette) and omit focal amplification shading entirely.
        cur_pal     = CLEAN_PALETTE if clean else pal
        cur_color_b = CLEAN_PALETTE["gain"] if clean else color_b
        cur_focal_regions = [] if clean else focal_regions

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(22, 9.0), facecolor="white",
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35},
        )

        def _shade(ax, ylim):
            for region in cur_focal_regions:
                ch = region["chrom"]
                if ch not in _OFFSETS:
                    continue
                x0 = _genome_x(ch, region["start"])
                x1 = _genome_x(ch, region["end"])
                ax.axvspan(x0, x1, alpha=0.18, color="#b8860b", zorder=2)
                ax.vlines([x0, x1], ylim[0], ylim[1], colors="#b8860b",
                          linewidth=0.6, alpha=0.5, zorder=2)
                genes = str(region.get("cosmic_genes", ""))
                if genes and genes != "nan":
                    ax.text((x0+x1)/2, ylim[1]*0.91, genes.split(",")[0],
                            fontsize=6, color="#b8860b", ha="center",
                            va="bottom", rotation=90, fontweight="bold",
                            zorder=5, clip_on=True)

        _style_ax(ax_top)
        for ref in [0, 0.58, -0.58]:
            lw = 0.8 if ref == 0 else 0.4
            ls = "-" if ref == 0 else ":"
            cl = "#999999" if ref == 0 else "#cccccc"
            ax_top.axhline(ref, color=cl, linewidth=lw, linestyle=ls, zorder=1)
        if cur_focal_regions:
            _shade(ax_top, logr_ylim)
        for bins, col, al, zo in [(ba, color_a, 0.50, 3),
                                   (bb, cur_color_b, 0.75, 4)]:
            xv = bins.apply(lambda r: _genome_x(
                r["chrom"], (r["start"]+r["end"])/2), axis=1).values
            yv = np.clip(bins["logr_corrected"].values, *logr_ylim)
            ax_top.scatter(xv, yv, s=3.0, c=col, alpha=al,
                           linewidths=0, zorder=zo, rasterized=True)
        ax_top.set_ylim(logr_ylim)
        ax_top.set_ylabel("TF-corrected log2 ratio",
                          fontsize=9, color="#444444", labelpad=5)
        ax_top.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax_top.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax_top.text(GENOME_SIZE*1.001, 0, "0", fontsize=7,
                    color="#888888", va="center")
        ax_top.set_title(f"{title_l1}   [{label}]\n{title_l2}",
                         fontsize=10, fontweight="bold", color="#222222",
                         pad=6, loc="left")
        top_h = [
            mpatches.Patch(color=color_a,
                           label=f"{tp_a_str}  {sample_a}  (TF={tf_a:.1%})"),
            mpatches.Patch(color=cur_color_b,
                           label=f"{tp_b_str}  {sample_b}  (TF={tf_b:.1%})"),
            plt.Line2D([0],[0], color="#999999", linewidth=1, label="log2R=0"),
        ]
        if cur_focal_regions:
            top_h.append(mpatches.Patch(color="#b8860b", alpha=0.4,
                                         label="Focal amp"))
        ax_top.legend(handles=top_h, fontsize=8, loc="lower left",
                      facecolor="white", edgecolor="#cccccc",
                      labelcolor="#222222", framealpha=1.0)

        _style_ax(ax_bot)
        for ref in [0, 0.5, -0.5]:
            lw = 0.9 if ref == 0 else 0.4
            ls = "-" if ref == 0 else ":"
            cl = "#999999" if ref == 0 else "#cccccc"
            ax_bot.axhline(ref, color=cl, linewidth=lw, linestyle=ls, zorder=1)
        if cur_focal_regions:
            _shade(ax_bot, delta_ylim)
        if not df_diff.empty:
            xv = df_diff.apply(lambda r: _genome_x(
                r["chrom"], (r["start"]+r["end"])/2), axis=1).values
            yv = np.clip(df_diff["delta_logr"].values, *delta_ylim)
            ca = np.where(yv >= 0.5, cur_pal["gain"],
                 np.where(yv <= -0.5, cur_pal["loss"], "#999999"))
            ax_bot.scatter(xv, yv, s=3.0, c=ca, alpha=0.70,
                           linewidths=0, zorder=3, rasterized=True)
        for reg in emergent_regions:
            ch = reg["chrom"]
            if ch not in _OFFSETS:
                continue
            xr0 = _genome_x(ch, reg["start"])
            xr1 = _genome_x(ch, reg["end"])
            shade_c = cur_pal["gain"] if reg["direction"] == "gain" else cur_pal["loss"]
            ax_bot.axvspan(xr0, xr1, alpha=0.22, color=shade_c, zorder=2)
        ax_bot.set_ylim(delta_ylim)
        ax_bot.set_ylabel(f"\u0394 log2 ratio\n({tp_b_str} \u2212 {tp_a_str})",
                          fontsize=9, color="#444444", labelpad=5)
        ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        for txt, yp in [("0", 0), ("+0.5", 0.5), ("\u22120.5", -0.5)]:
            ax_bot.text(GENOME_SIZE*1.001, yp, txt, fontsize=6 if yp!=0 else 7,
                        color="#999999" if yp!=0 else "#888888", va="center")
        bot_h = [
            mpatches.Patch(color=cur_pal["gain"],
                           label=f"Gain in {tp_b_str}  (\u0394 > +0.5)"),
            mpatches.Patch(color=cur_pal["loss"],
                           label=f"Loss in {tp_b_str}  (\u0394 < \u22120.5)"),
            mpatches.Patch(color="#999999", label="Unchanged"),
        ]
        if emergent_regions:
            bot_h.append(mpatches.Patch(color=cur_pal["gain"], alpha=0.3,
                                         label="Emergent CNA"))
        ax_bot.legend(handles=bot_h, fontsize=8, loc="lower left",
                      facecolor="white", edgecolor="#cccccc",
                      labelcolor="#222222", framealpha=1.0)
        fig.tight_layout()
        _save(fig, out_path)

    # ── Output ───────────────────────────────────────────────────────────────
    if plot_raw:
        _render(bins_a_raw, bins_b_raw, diff_raw, output_path, "raw")
    _render(bins_a_s5,  bins_b_s5,  diff_s5,
            output_path.with_name(output_path.name + "_smooth5"),
            "5-bin smooth")
    _render(bins_a_s10, bins_b_s10, diff_s10,
            output_path.with_name(output_path.name + "_smooth10"),
            "10-bin smooth")

    # ── Clean (publication-figure) variants ─────────────────────────────────
    # Same overlaid + differential panels, no focal amp shading, fixed
    # gain/loss colour scheme (CLEAN_PALETTE) consistent across all entities.
    if clean_plots:
        clean_dir = output_path.parent / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_out = clean_dir / output_path.name
        _render(bins_a_s5,  bins_b_s5,  diff_s5,
                clean_out.with_name(clean_out.name + "_clean_smooth5"),
                "5-bin smooth", clean=True)
        _render(bins_a_s10, bins_b_s10, diff_s10,
                clean_out.with_name(clean_out.name + "_clean_smooth10"),
                "10-bin smooth", clean=True)

# ---------------------------------------------------------------------------
# Batch: one combined plot per consecutive pair per patient
# ---------------------------------------------------------------------------

def plot_all_pairwise_longitudinal(
        tf_df:                pd.DataFrame,
        seg_files:            dict[str, Path],
        output_dir:           Path,
        min_tf:               float = 0.10,
        logr_ylim:            tuple[float, float] = (-2.5, 2.5),
        focal_tsv:            Path | None = None,
        max_noise:            float | None = 0.15,
        min_signal:           float | None = None,
        seg_files_segs:       dict[str, Path] | None = None,
        min_altered_fraction: float = 0.05,
        plot_raw:             bool  = False,
        emergent_min_bins:    int   = 3,
        emergent_min_delta:   float = 0.5,
        emergent_tsv:         Path | None = None,
        clean_plots:          bool  = True,
) -> int:
    """
    Generate one genome-wide log2-ratio comparison plot per consecutive
    timepoint pair per patient.

    Parameters
    ----------
    tf_df          : tumor fraction table.
    seg_files      : {sample: Path} to 50 kbp .cna.seg bin-level files.
    seg_files_segs : {sample: Path} to 50 kbp .seg.txt segmented files,
                     used for segment-mean scaling. Falls back to seg_files
                     if not provided.
    seg_files  : {sample_name: Path} pointing to 50 kbp .cna.seg files.
    output_dir : plots written as <patient>_TP<a>_vs_TP<b>.pdf/png.
    min_tf     : skip samples with TF below this value.
    logr_ylim  : y-axis range for all plots.
    focal_tsv  : path to focal_amplifications.tsv for shading (optional).

    Returns
    -------
    Number of plots generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = tf_df.dropna(subset=["timepoint", "tumor_fraction"]).copy()
    df = df[df["tumor_fraction"] >= min_tf]
    df["timepoint"] = df["timepoint"].astype(int)

    n_plots = 0

    # Diagnostics — print once to help debug empty output
    n_multi = sum(1 for _, g in df.groupby("patient")
                  if g["timepoint"].nunique() >= 2)
    n_with_seg = sum(
        1 for _, g in df.groupby("patient")
        if g["timepoint"].nunique() >= 2
        and len(g[g["sample"].isin(seg_files)]) >= 2
    )
    print(f"        Patients with >= 2 timepoints in tf_df : {n_multi}")
    print(f"        ...of which have >= 2 seg files found  : {n_with_seg}")
    if n_multi > 0 and n_with_seg == 0:
        # Show a sample of the mismatch
        _tf_samples  = set(df["sample"].unique())
        _seg_samples = set(seg_files.keys())
        _overlap     = _tf_samples & _seg_samples
        print(f"        tf_df samples (first 5)  : {sorted(_tf_samples)[:5]}")
        print(f"        seg_files keys (first 5) : {sorted(_seg_samples)[:5]}")
        print(f"        Overlapping samples      : {len(_overlap)}")

    for patient, pat_df in df.groupby("patient"):
        if pat_df["timepoint"].nunique() < 2:
            continue
        entity = entity_from_sample(str(patient))
        pat_df = pat_df[pat_df["sample"].isin(seg_files)].sort_values("timepoint")
        if len(pat_df) < 2:
            continue

        samples    = pat_df["sample"].tolist()
        timepoints = pat_df["timepoint"].tolist()
        tfs        = pat_df["tumor_fraction"].tolist()

        for idx in range(len(samples) - 1):
            sa, sb   = samples[idx],    samples[idx + 1]
            tpa, tpb = timepoints[idx], timepoints[idx + 1]
            tfa, tfb = tfs[idx],        tfs[idx + 1]

            plot_name   = f"{patient}_TP{tpa}_vs_TP{tpb}"
            output_path = output_dir / plot_name
            print(f"        {plot_name}  (TF: {tfa:.1%} -> {tfb:.1%})")

            if min(tfa, tfb) < min_tf:
                print(f"        Skipping {patient} TP{tpa}->TP{tpb}: "
                      f"TF {tfa:.1%}/{tfb:.1%} both must be >= {min_tf:.0%}")
                continue

            plot_longitudinal_combined(
                sample_a             = sa,
                seg_path_a           = seg_files[sa],
                tf_a                 = tfa,
                sample_b             = sb,
                seg_path_b           = seg_files[sb],
                tf_b                 = tfb,
                output_path          = output_path,
                patient              = patient,
                entity               = entity,
                timepoint_a          = tpa,
                timepoint_b          = tpb,
                logr_ylim            = logr_ylim,
                focal_tsv            = focal_tsv,
                max_noise            = max_noise,
                min_signal           = min_signal,
                seg_path_a_segs      = seg_files_segs.get(sa) if seg_files_segs else None,
                seg_path_b_segs      = seg_files_segs.get(sb) if seg_files_segs else None,
                min_altered_fraction = min_altered_fraction,
                plot_raw             = plot_raw,
                emergent_min_bins    = emergent_min_bins,
                emergent_min_delta   = emergent_min_delta,
                emergent_tsv         = emergent_tsv,
                clean_plots          = clean_plots,
            )
            n_plots += 1
            gc.collect()

    return n_plots


# ---------------------------------------------------------------------------
# Differential longitudinal plot  (later − earlier, both TF-corrected)
# ---------------------------------------------------------------------------

def _align_bins(
        bins_a: pd.DataFrame,
        bins_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join two bin DataFrames on (chrom, start, end) and return a
    DataFrame with columns:
        chrom, start, end, logr_a, logr_b, delta_logr

    delta_logr = logr_b (later) − logr_a (earlier)
    Positive delta = gained signal in later sample.
    Negative delta = lost signal in later sample.

    Bins present in only one sample are dropped (inner join), which handles
    the rare case where a chromosome is missing from one file.
    """
    a = bins_a[["chrom", "start", "end", "logr_corrected"]].rename(
        columns={"logr_corrected": "logr_a"}
    )
    b = bins_b[["chrom", "start", "end", "logr_corrected"]].rename(
        columns={"logr_corrected": "logr_b"}
    )
    merged = a.merge(b, on=["chrom", "start", "end"], how="inner")
    merged["delta_logr"] = merged["logr_b"] - merged["logr_a"]
    return merged


def plot_differential_longitudinal(
        sample_a:        str,
        seg_path_a:      Path,
        tf_a:            float,
        sample_b:        str,
        seg_path_b:      Path,
        tf_b:            float,
        output_path:     Path,
        patient:         str = "",
        entity:          str = "",
        timepoint_a:     int | None = None,
        timepoint_b:     int | None = None,
        delta_ylim:      tuple[float, float] = (-3.0, 3.0),
        focal_tsv:       Path | None = None,
        max_noise:       float | None = 0.15,
        seg_path_a_segs: Path | None = None,
        min_signal:      float | None = None,
        seg_path_b_segs: Path | None = None,
) -> None:
    """
    Genome-wide differential log2-ratio plot: (later − earlier), both samples
    TF-corrected before subtraction.

    Uses segment means from .seg.txt for the scaling optimisation and
    raw bin logR from .cna.seg for the differential scatter display.
    """
    pal = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])

    bins_a = _load_bins(seg_path_a, sample_a, tf_from=tf_a)
    bins_b = _load_bins(seg_path_b, sample_b, tf_from=tf_b)

    if bins_a is None or bins_b is None:
        print(f"      [WARN] Could not load bin-level data for "
              f"{sample_a} or {sample_b} — skipping differential plot.")
        return

    # Segment means for scaling optimisation
    if seg_path_a_segs is not None and seg_path_b_segs is not None:
        seg_a = _load_bins_with_seg_means(seg_path_a, seg_path_a_segs,
                                          sample_a, tf_a)
        seg_b = _load_bins_with_seg_means(seg_path_b, seg_path_b_segs,
                                          sample_b, tf_b)
    else:
        seg_a, seg_b = bins_a, bins_b

    aligned = _align_bins(
        seg_a if seg_a is not None else bins_a,
        seg_b if seg_b is not None else bins_b,
    )
    if aligned.empty:
        print(f"      [WARN] No overlapping bins between {sample_a} and "
              f"{sample_b} — skipping differential plot.")
        return

    # Higher-TF sample is fixed; lower-TF sample scaled upward
    if tf_a >= tf_b:
        best_m, best_rmse = _optimise_scaling(
            aligned["logr_a"].values, aligned["logr_b"].values
        )
        bins_b = bins_b.copy()
        bins_b["logr_corrected"] = bins_b["logr_corrected"] * best_m
    else:
        best_m, best_rmse = _optimise_scaling(
            aligned["logr_b"].values, aligned["logr_a"].values
        )
        bins_a = bins_a.copy()
        bins_a["logr_corrected"] = bins_a["logr_corrected"] * best_m

    print(f"        Differential — optimal multiplier: {best_m:.3f},  "
          f"RMSE: {best_rmse:.4f}")

    # Per-sample signal metrics
    q_a, n_above_a, noise_a, snr_a = _signal_metrics(bins_a)
    q_b, n_above_b, noise_b, snr_b = _signal_metrics(bins_b)
    pair_noise = max(
        noise_a if not np.isnan(noise_a) else 0.0,
        noise_b if not np.isnan(noise_b) else 0.0,
    )
    print(f"        {sample_a}: noise={noise_a:.4f}  p95={q_a:.3f}  "
          f"bins_above_2x={n_above_a}")
    print(f"        {sample_b}: noise={noise_b:.4f}  p95={q_b:.3f}  "
          f"bins_above_2x={n_above_b}")
    if max_noise is not None and pair_noise > max_noise:
        print(f"        Pair noise {pair_noise:.4f} > max_noise {max_noise:.4f} "
              f"— skipping differential plot.")
        return
    if min_signal is not None:
        if snr_a < min_signal or snr_b < min_signal:
            print(f"        SNR ({snr_a:.2f}, {snr_b:.2f}) below "
                  f"min_signal {min_signal:.2f} — skipping differential plot.")
            return

    # Recompute aligned diff with the scaled bins
    diff = _align_bins(bins_a, bins_b)

    # Focal amp shading
    focal_regions: list[dict] = []
    if focal_tsv is not None:
        focal_regions = _merge_overlapping(
            _build_focal_annotations(focal_tsv, patient)
        )

    fig, ax = plt.subplots(figsize=(22, 4.5), facecolor="white")
    _style_ax(ax)

    # Reference lines
    ax.axhline( 0,    color="#999999", linewidth=0.9, linestyle="-",  zorder=1)
    ax.axhline( 0.5,  color="#cccccc", linewidth=0.4, linestyle=":",  zorder=1)
    ax.axhline(-0.5,  color="#cccccc", linewidth=0.4, linestyle=":",  zorder=1)

    # Focal amp shading
    for region in focal_regions:
        chrom = region["chrom"]
        if chrom not in _OFFSETS:
            continue
        x0 = _genome_x(chrom, region["start"])
        x1 = _genome_x(chrom, region["end"])
        ax.axvspan(x0, x1, alpha=0.18, color="#b8860b", zorder=2)
        ax.vlines([x0, x1], delta_ylim[0], delta_ylim[1],
                  colors="#ffd166", linewidth=0.6, alpha=0.5, zorder=2)
        genes = region["cosmic_genes"]
        if genes:
            x_mid = (x0 + x1) / 2
            ax.text(x_mid, delta_ylim[1] * 0.92, genes.split(",")[0],
                    fontsize=6, color="#b8860b", ha="center", va="bottom",
                    rotation=90, fontweight="bold", zorder=5, clip_on=True)

    # Colour each point by direction and magnitude
    diff = diff[diff["chrom"].isin(_OFFSETS)].copy()
    x_vals = diff.apply(
        lambda r: _genome_x(r["chrom"], (r["start"] + r["end"]) / 2),
        axis=1,
    ).values
    y_vals = np.clip(diff["delta_logr"].values, delta_ylim[0], delta_ylim[1])

    # Assign colours: gain / neutral / loss
    gain_thresh =  0.5
    loss_thresh = -0.5
    colors = np.where(
        y_vals >= gain_thresh, pal["gain"],
        np.where(y_vals <= loss_thresh, pal["loss"], "#4a5568")
    )

    ax.scatter(x_vals, y_vals,
               s=3.0, c=colors, alpha=0.70,
               linewidths=0, zorder=3, rasterized=True)

    # Axes
    ax.set_ylim(delta_ylim)
    ax.set_ylabel("Δ log2 ratio\n(later − earlier)",
                  fontsize=9, color="#444444", labelpad=5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.text(GENOME_SIZE * 1.001,  0,    "0",      fontsize=7, color="#888888", va="center")
    ax.text(GENOME_SIZE * 1.001,  0.5,  "+0.5",   fontsize=6, color="#999999", va="center")
    ax.text(GENOME_SIZE * 1.001, -0.5,  "−0.5",   fontsize=6, color="#999999", va="center")

    # Title
    tp_a_str  = f"TP{timepoint_a}" if timepoint_a is not None else sample_a
    tp_b_str  = f"TP{timepoint_b}" if timepoint_b is not None else sample_b
    n_bins    = len(diff)
    noise_str = f"{pair_noise:.3f}" if not np.isnan(pair_noise) else "n/a"
    ax.set_title(
        f"{patient}  ({entity})   Δ {tp_b_str} − {tp_a_str}   "
        f"(TF: {tf_a:.1%} → {tf_b:.1%},  scale={best_m:.2f},  "
        f"noise={noise_str},  {n_bins:,} bins)\n"
        f"{tp_a_str}: p95={q_a:.3f}  above2×={n_above_a}    "
        f"{tp_b_str}: p95={q_b:.3f}  above2×={n_above_b}",
        fontsize=10, fontweight="bold", color="#222222", pad=6, loc="left",
    )

    handles = [
        mpatches.Patch(color=pal["gain"],  label=f"Gain in {tp_b_str}  (Δ > +0.5)"),
        mpatches.Patch(color=pal["loss"],  label=f"Loss in {tp_b_str}  (Δ < −0.5)"),
        mpatches.Patch(color="#999999",    label="Unchanged"),
        plt.Line2D([0], [0], color="#999999", linewidth=1,
                   linestyle="-", label="Δ = 0  (no change)"),
    ]
    if focal_regions:
        handles.append(
            mpatches.Patch(color="#b8860b", alpha=0.4,
                           label="Focal amp in any sample of this patient")
        )
    ax.legend(handles=handles, fontsize=8, loc="lower left",
              facecolor="white", edgecolor="#cccccc",
              labelcolor="#222222", framealpha=1.0)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Batch differential plots
# ---------------------------------------------------------------------------

def plot_all_differential_longitudinal(
        tf_df:          pd.DataFrame,
        seg_files:      dict[str, Path],
        output_dir:     Path,
        min_tf:         float = 0.10,
        delta_ylim:     tuple[float, float] = (-3.0, 3.0),
        focal_tsv:      Path | None = None,
        max_noise:      float | None = 0.15,
        min_signal:     float | None = None,
        seg_files_segs: dict[str, Path] | None = None,
) -> int:
    """
    Generate one differential (later − earlier) log2-ratio plot per
    consecutive timepoint pair per patient.

    Parameters and return value mirror plot_all_pairwise_longitudinal.
    Output files are named <patient>_TP<a>_vs_TP<b>_diff.pdf/png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = tf_df.dropna(subset=["timepoint", "tumor_fraction"]).copy()
    df = df[df["tumor_fraction"] >= min_tf]
    df["timepoint"] = df["timepoint"].astype(int)

    n_plots = 0

    for patient, pat_df in df.groupby("patient"):
        if pat_df["timepoint"].nunique() < 2:
            continue
        entity = entity_from_sample(str(patient))
        pat_df = pat_df[pat_df["sample"].isin(seg_files)].sort_values("timepoint")
        if len(pat_df) < 2:
            continue

        samples    = pat_df["sample"].tolist()
        timepoints = pat_df["timepoint"].tolist()
        tfs        = pat_df["tumor_fraction"].tolist()

        for idx in range(len(samples) - 1):
            sa, sb   = samples[idx],    samples[idx + 1]
            tpa, tpb = timepoints[idx], timepoints[idx + 1]
            tfa, tfb = tfs[idx],        tfs[idx + 1]

            plot_name   = f"{patient}_TP{tpa}_vs_TP{tpb}_diff"
            output_path = output_dir / plot_name
            if min(tfa, tfb) < min_tf:
                print(f"        Skipping {patient} TP{tpa}->TP{tpb} diff: "
                      f"TF {tfa:.1%}/{tfb:.1%} both must be >= {min_tf:.0%}")
                continue

            print(f"        {plot_name}  (TF: {tfa:.1%} -> {tfb:.1%})")

            plot_differential_longitudinal(
                sample_a         = sa,
                seg_path_a       = seg_files[sa],
                tf_a             = tfa,
                sample_b         = sb,
                seg_path_b       = seg_files[sb],
                tf_b             = tfb,
                output_path      = output_path,
                patient          = patient,
                entity           = entity,
                timepoint_a      = tpa,
                timepoint_b      = tpb,
                delta_ylim       = delta_ylim,
                focal_tsv        = focal_tsv,
                max_noise        = max_noise,
                min_signal       = min_signal,
                seg_path_a_segs  = seg_files_segs.get(sa) if seg_files_segs else None,
                seg_path_b_segs  = seg_files_segs.get(sb) if seg_files_segs else None,
            )
            n_plots += 1
            gc.collect()

    return n_plots


# ---------------------------------------------------------------------------
# Focal amplification trajectories — relative CN across timepoints
# ---------------------------------------------------------------------------

def _region_label(region: dict) -> str:
    """
    Human-readable label for a focal amp region: first COSMIC gene name if
    available, otherwise the genomic coordinates in Mb.
    """
    genes = str(region.get("cosmic_genes", "")).strip()
    if genes and genes.lower() != "nan":
        return genes.split(",")[0].strip()
    chrom    = region["chrom"]
    start_mb = region["start"] / 1_000_000
    end_mb   = region["end"] / 1_000_000
    return f"{chrom}:{start_mb:.2f}-{end_mb:.2f}Mb"


def _region_logr_stats(bins_df: pd.DataFrame, region: dict) -> tuple[float, float, float]:
    """
    Mean, Q1, and Q3 of the TF-corrected log2 ratio of all 50 kbp bins
    overlapping a focal amp region. Returns (nan, nan, nan) if no bins
    overlap.

    The (Q1, Q3) interquartile range serves as a measure of variation
    across the bins comprising the region — wide segments with many bins
    give a tighter range, very small segments (few bins) naturally have
    a wider or degenerate (Q1==Q3) range.
    """
    chrom = region["chrom"]
    s, e  = region["start"], region["end"]
    sub = bins_df[
        (bins_df["chrom"] == chrom) &
        (bins_df["end"]   > s) &
        (bins_df["start"] < e)
    ]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    vals = sub["logr_corrected"].values
    mean = float(np.mean(vals))
    q1   = float(np.percentile(vals, 25))
    q3   = float(np.percentile(vals, 75))
    return mean, q1, q3


def _trajectory_palette(n: int) -> list:
    """Return n visually distinct colours for line plots."""
    if n <= 10:
        cmap = plt.colormaps.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.colormaps.get_cmap("tab20")
        return [cmap(i) for i in range(n)]
    cmap = plt.colormaps.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


def plot_focal_amp_trajectories(
        patient:        str,
        entity:         str,
        samples:        list[str],
        timepoints:     list[int],
        tfs:            list[float],
        seg_files_bins: dict[str, Path],
        focal_regions:  list[dict],
        output_dir:     Path,
        seg_files_segs: dict[str, Path] | None = None,
) -> bool:
    """
    For one patient, plot the relative copy-number trajectory of each focal
    amplification region across timepoints as line plots.

    Two figures are written:
      <patient>_focal_amp_trajectory_log2.pdf/.png
          y-axis: log2 ratio scaled to a common reference (see below),
          one line per focal amp region.
      <patient>_focal_amp_trajectory_cn.pdf/.png
          y-axis: relative copy number = 2 * 2^(log2 ratio), same scaling.

    Scaling
    -------
    All timepoints are placed on a single common scale defined by the
    patient's HIGHEST-TF timepoint (the reference; scale factor = 1.0).
    For every other timepoint, an empirical scale factor m >= 1 is fit
    via weighted least squares (_optimise_scaling) between that
    timepoint's segment-mean log2 ratios and the reference's, then
    applied to that timepoint's bin-level log2 ratios. This is the same
    empirical-scaling approach used for the pairwise longitudinal plots,
    extended to a single shared reference rather than pairwise references
    — so all timepoints for a patient are directly comparable on one
    plot. If seg_files_segs is not provided, falls back to a naive
    TF-ratio rescaling (logR * TF_ref / TF_sample).

    Each point is the mean over the 50 kbp bins overlapping the region;
    error bars show the interquartile range (Q1-Q3) of those bins, giving
    a sense of within-segment variation/noise at each timepoint. Regions
    spanning very few bins will have narrow or degenerate (zero-length)
    error bars.

    Each focal amp region (merged across all of the patient's timepoints)
    gets its own colour, consistent between the two plots.

    Returns True if plots were written, False if there was nothing to plot
    (fewer than 2 timepoints, no focal regions, or no bin data available).
    """
    if len(samples) < 2 or not focal_regions:
        return False

    ref_tf = max(tfs)
    if ref_tf <= 0:
        return False
    ref_idx = tfs.index(ref_tf)

    # ── Region labels (unique) ───────────────────────────────────────────
    labels: list[str] = []
    seen: dict[str, int] = {}
    for region in focal_regions:
        lbl = _region_label(region)
        if lbl in seen:
            seen[lbl] += 1
            lbl = f"{lbl} ({seen[lbl]})"
        else:
            seen[lbl] = 0
        labels.append(lbl)

    # ── Determine a common scale factor for each timepoint ───────────────
    # Reference (highest-TF) timepoint: scale factor = 1.0.
    # All other timepoints: empirical scale factor (>= 1) fit against the
    # reference via segment-mean log2 ratios, applied to bin-level logR.
    scale_factors: list[float] = [1.0] * len(samples)

    ref_sample = samples[ref_idx]
    ref_seg_path = seg_files_bins.get(ref_sample)
    ref_segmeans = None
    if seg_files_segs is not None and ref_seg_path is not None:
        ref_segs_path = seg_files_segs.get(ref_sample)
        if ref_segs_path is not None:
            ref_segmeans = _load_bins_with_seg_means(
                ref_seg_path, ref_segs_path, ref_sample, tf_from=tfs[ref_idx]
            )

    for i, (sample, tf) in enumerate(zip(samples, tfs)):
        if i == ref_idx:
            continue
        m = None
        if ref_segmeans is not None:
            seg_path  = seg_files_bins.get(sample)
            segs_path = seg_files_segs.get(sample) if seg_files_segs else None
            if seg_path is not None and segs_path is not None:
                samp_segmeans = _load_bins_with_seg_means(
                    seg_path, segs_path, sample, tf_from=tf
                )
                if samp_segmeans is not None and not samp_segmeans.empty:
                    aligned = _align_bins(ref_segmeans, samp_segmeans)
                    if not aligned.empty:
                        m, _ = _optimise_scaling(
                            aligned["logr_a"].values, aligned["logr_b"].values
                        )
        if m is None:
            # Fallback: naive TF-ratio scaling
            m = (ref_tf / tf) if tf > 0 else 1.0
        scale_factors[i] = m

    # ── Per-sample, per-region mean logR + IQR (common scale applied) ───
    log2_matrix = np.full((len(samples), len(focal_regions)), np.nan)
    log2_q1     = np.full((len(samples), len(focal_regions)), np.nan)
    log2_q3     = np.full((len(samples), len(focal_regions)), np.nan)
    for i, (sample, tf) in enumerate(zip(samples, tfs)):
        seg_path = seg_files_bins.get(sample)
        if seg_path is None:
            continue
        # tf_to=tf_from -> no TF-ratio correction; the empirical scale
        # factor (or fallback TF-ratio factor) is applied explicitly below.
        bins_df = _load_bins(seg_path, sample, tf_from=tf)
        if bins_df is None or bins_df.empty:
            continue
        m = scale_factors[i]
        if m != 1.0:
            bins_df = bins_df.copy()
            bins_df["logr_corrected"] = bins_df["logr_corrected"] * m
        for j, region in enumerate(focal_regions):
            mean, q1, q3 = _region_logr_stats(bins_df, region)
            log2_matrix[i, j] = mean
            log2_q1[i, j]     = q1
            log2_q3[i, j]     = q3

    if np.all(np.isnan(log2_matrix)):
        return False

    cn_matrix = 2.0 * np.power(2.0, log2_matrix)
    # 2^x is monotonic, so percentile bounds carry over directly
    cn_q1 = 2.0 * np.power(2.0, log2_q1)
    cn_q3 = 2.0 * np.power(2.0, log2_q3)

    output_dir.mkdir(parents=True, exist_ok=True)
    colors    = _trajectory_palette(len(focal_regions))
    x         = timepoints
    tp_labels = [f"TP{tp}" for tp in timepoints]

    title_main = f"{patient}  ({entity})   Focal amplification trajectories"
    title_sub  = "   ".join(
        f"TP{tp} (TF={tf:.1%}, \u00d7{m:.2f})"
        for tp, tf, m in zip(timepoints, tfs, scale_factors)
    ) + (f"     [common scale: TP{timepoints[ref_idx]} is reference "
         f"(TF={ref_tf:.1%}); error bars = IQR of bins in segment]")

    def _new_axes():
        fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(tp_labels)
        ax.set_xlabel("Timepoint", fontsize=9, color="#444444")
        ax.set_title(f"{title_main}\n{title_sub}", fontsize=10,
                      fontweight="bold", color="#222222", loc="left")
        return fig, ax

    def _plot_lines(ax, matrix, q1_matrix, q3_matrix):
        for j, (label, color) in enumerate(zip(labels, colors)):
            yv  = matrix[:, j]
            lo  = yv - q1_matrix[:, j]
            hi  = q3_matrix[:, j] - yv
            # Clip negative error magnitudes from NaNs/edge cases
            lo  = np.clip(lo, 0, None)
            hi  = np.clip(hi, 0, None)
            ax.errorbar(x, yv, yerr=[lo, hi], fmt="o-", linewidth=1.8,
                         markersize=5, color=color, label=label,
                         ecolor=color, elinewidth=1.0, capsize=3,
                         alpha=0.95, zorder=3)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=True, facecolor="white", edgecolor="#cccccc",
                  labelcolor="#222222")

    # ── Log2-space plot ───────────────────────────────────────────────────
    fig, ax = _new_axes()
    ax.axhline(0,    color="#999999", linewidth=0.8, linestyle="-",  zorder=1)
    for ref in (0.58, -0.58, 1.0, -1.0):
        ax.axhline(ref, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    _plot_lines(ax, log2_matrix, log2_q1, log2_q3)
    ax.set_ylabel("TF-corrected log2 ratio\n(focal amplification region)",
                   fontsize=9, color="#444444")
    fig.tight_layout()
    _save(fig, output_dir / f"{patient}_focal_amp_trajectory_log2")

    # ── Relative copy-number plot ────────────────────────────────────────
    fig, ax = _new_axes()
    ax.axhline(2, color="#999999", linewidth=0.8, linestyle="-", zorder=1)
    for ref in (1, 3, 4, 6, 8):
        ax.axhline(ref, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    _plot_lines(ax, cn_matrix, cn_q1, cn_q3)
    ax.set_ylabel("Relative copy number\n"
                   r"(2 $\times$ 2$^{\mathrm{log2ratio}}$, scaled)",
                   fontsize=9, color="#444444")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, output_dir / f"{patient}_focal_amp_trajectory_cn")

    return True


def plot_all_focal_amp_trajectories(
        focal_tsv:      Path,
        tf_df:          pd.DataFrame,
        seg_files_bins: dict[str, Path],
        output_dir:     Path,
        min_tf:         float = 0.10,
        min_timepoints: int   = 2,
        seg_files_segs: dict[str, Path] | None = None,
) -> int:
    """
    Batch driver: for every patient with >= min_timepoints qualifying
    timepoints AND at least one focal amplification event (in any of their
    samples), generate the log2 and relative-CN trajectory plots.

    Parameters
    ----------
    focal_tsv      : path to focal_amplifications.tsv.
    tf_df          : tumor fraction table (sample, patient, timepoint,
                     tumor_fraction).
    seg_files_bins : {sample: Path} to 50 kbp .cna.seg bin-level files.
    output_dir     : directory for the trajectory plots.
    min_tf         : minimum tumor fraction for a timepoint to be included
                     (default 0.10 = 10%; samples below this are dropped
                     from the trajectory entirely).
    min_timepoints : minimum number of qualifying timepoints per patient.
    seg_files_segs : {sample: Path} to segmented .seg.txt files, used for
                     the empirical common-reference scale factor. If not
                     provided, falls back to naive TF-ratio scaling.

    Returns
    -------
    Number of patients for which plots were generated.
    """
    if not focal_tsv.exists() or focal_tsv.stat().st_size == 0:
        print("        No focal_amplifications.tsv — skipping trajectories.")
        return 0

    try:
        focal_df = pd.read_csv(focal_tsv, sep="\t", low_memory=False)
    except Exception as e:
        print(f"        [WARN] Could not read {focal_tsv}: {e}")
        return 0

    if focal_df.empty or "sample" not in focal_df.columns:
        return 0

    focal_df = focal_df.copy()
    focal_df["chrom"] = focal_df["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}")
    if "cosmic_genes" not in focal_df.columns:
        focal_df["cosmic_genes"] = ""
    focal_df["cosmic_genes"] = focal_df["cosmic_genes"].fillna("").astype(str)
    focal_df.loc[focal_df["cosmic_genes"].str.lower() == "nan", "cosmic_genes"] = ""

    focal_samples = set(focal_df["sample"].unique())

    df = tf_df.dropna(subset=["timepoint", "tumor_fraction"]).copy()
    df = df[df["tumor_fraction"] >= min_tf]
    df["timepoint"] = df["timepoint"].astype(int)
    df = df[df["sample"].isin(seg_files_bins)]

    output_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    for patient, pat_df in df.groupby("patient"):
        pat_df = pat_df.sort_values("timepoint")
        if pat_df["timepoint"].nunique() < min_timepoints:
            continue

        pat_samples = set(pat_df["sample"])
        if not (pat_samples & focal_samples):
            continue

        pat_focal = focal_df[focal_df["sample"].isin(pat_samples)]
        regions = [
            {
                "chrom":        row["chrom"],
                "start":        int(row["start"]),
                "end":          int(row["end"]),
                "cosmic_genes": row["cosmic_genes"],
            }
            for _, row in pat_focal.iterrows()
        ]
        regions = _merge_overlapping(regions)
        if not regions:
            continue

        entity = entity_from_sample(str(patient))
        ok = plot_focal_amp_trajectories(
            patient        = str(patient),
            entity         = entity,
            samples        = pat_df["sample"].tolist(),
            timepoints     = pat_df["timepoint"].tolist(),
            tfs            = pat_df["tumor_fraction"].tolist(),
            seg_files_bins = seg_files_bins,
            focal_regions  = regions,
            output_dir     = output_dir,
            seg_files_segs = seg_files_segs,
        )
        if ok:
            n_plots += 1
            print(f"        {patient}: {len(regions)} focal amp region(s) "
                  f"across {pat_df['timepoint'].nunique()} timepoints")
        gc.collect()

    return n_plots