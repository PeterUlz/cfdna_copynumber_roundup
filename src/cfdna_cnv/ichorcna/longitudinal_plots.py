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
        print(f"      [WARN] No logR column in bins file {seg_path_bins.name}. "
              f"Columns found: {list(df_bins.columns)}")
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
    candidates = [
        "logr", "log2_ratio", "log2ratio", "seg.mean",
        "median", "logr_copy_ratio",
        # ichorCNA-specific variants (case-insensitive check below)
        "logR", "Corrected_logR", "copy.number.corrected",
        "logCopyRatio", "log2CopyRatio",
    ]
    # First try exact match (fast)
    for c in candidates:
        if c in df.columns:
            return c
    # Then try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
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
        # ichorCNA-TF-ratio correction: scale lower-TF sample by tf_a/tf_b
        tf_ratio = tf_a / tf_b if tf_b > 0 else 1.0
        def _scale_tf(df): out=df.copy(); out["logr_corrected"]*=tf_ratio; return out
        bins_b_raw_tfratio = _scale_tf(_load_bins(seg_path_b, sample_b, tf_from=tf_b))
        bins_a_raw_tfratio = _load_bins(seg_path_a, sample_a, tf_from=tf_a)
    else:
        best_m, _ = _optimise_scaling(
            aligned["logr_b"].values, aligned["logr_a"].values)
        def _scale(df): out=df.copy(); out["logr_corrected"]*=best_m; return out
        bins_a_raw = _scale(bins_a_raw)
        # ichorCNA-TF-ratio correction: scale lower-TF sample by tf_b/tf_a
        tf_ratio = tf_b / tf_a if tf_a > 0 else 1.0
        def _scale_tf(df): out=df.copy(); out["logr_corrected"]*=tf_ratio; return out
        bins_a_raw_tfratio = _scale_tf(_load_bins(seg_path_a, sample_a, tf_from=tf_a))
        bins_b_raw_tfratio = _load_bins(seg_path_b, sample_b, tf_from=tf_b)

    # Guard against None from _load_bins in tf-ratio path
    if bins_a_raw_tfratio is None:
        bins_a_raw_tfratio = bins_a_raw
    if bins_b_raw_tfratio is None:
        bins_b_raw_tfratio = bins_b_raw

    # ── Smooth scaled raw bins (empirical) ──────────────────────────────────
    bins_a_s5  = _smooth(bins_a_raw,  5)
    bins_b_s5  = _smooth(bins_b_raw,  5)
    bins_a_s10 = _smooth(bins_a_raw, 10)
    bins_b_s10 = _smooth(bins_b_raw, 10)

    # ── Smooth TF-ratio bins ────────────────────────────────────────────────
    bins_a_s5_tf  = _smooth(bins_a_raw_tfratio,  5)
    bins_b_s5_tf  = _smooth(bins_b_raw_tfratio,  5)
    bins_a_s10_tf = _smooth(bins_a_raw_tfratio, 10)
    bins_b_s10_tf = _smooth(bins_b_raw_tfratio, 10)

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
    title_l1_tf = (f"{patient}  ({entity})   "
                   f"{tp_a_str} (TF={tf_a:.1%})  vs  {tp_b_str} (TF={tf_b:.1%})"
                   f"   TF-ratio={tf_ratio:.2f}  noise={noise_str}  "
                   f"[ichorCNA-TF correction]")
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
    def _render(ba, bb, df_diff, out_path, label, clean=False, pub=False):
        # Clean plots: fixed gain/loss colours, no focal amp shading.
        # Pub plots:   additionally strip title, legend, axis labels
        #              (keep ticks and tick labels for scale reference).
        cur_pal           = CLEAN_PALETTE if (clean or pub) else pal
        cur_color_b       = CLEAN_PALETTE["gain"] if (clean or pub) else color_b
        cur_focal_regions = [] if (clean or pub) else focal_regions

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
        ax_top.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax_top.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        if not pub:
            ax_top.set_ylabel("TF-corrected log2 ratio",
                              fontsize=9, color="#444444", labelpad=5)
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
        ax_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        if not pub:
            ax_bot.set_ylabel(f"\u0394 log2 ratio\n({tp_b_str} \u2212 {tp_a_str})",
                              fontsize=9, color="#444444", labelpad=5)
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
    # Fixed gain/loss colours, no focal amp shading.
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

        # ── Pub (figure-ready) variants ──────────────────────────────────
        # No title, no legend, no axis labels — keep ticks for scale.
        # Drop-in ready for figure assembly in Illustrator/Inkscape.
        pub_dir = output_path.parent / "pub"
        pub_dir.mkdir(parents=True, exist_ok=True)
        pub_out = pub_dir / output_path.name
        _render(bins_a_s5,  bins_b_s5,  diff_s5,
                pub_out.with_name(pub_out.name + "_pub_smooth5"),
                "5-bin smooth", pub=True)
        _render(bins_a_s10, bins_b_s10, diff_s10,
                pub_out.with_name(pub_out.name + "_pub_smooth10"),
                "10-bin smooth", pub=True)

    # ── ichorCNA-TF-ratio correction variants ────────────────────────────────
    # Same plots but using ichorCNA TF ratio (tf_high/tf_low) for correction
    # instead of the empirical WLS scale factor. Useful for direct comparison
    # of the two correction approaches.
    tfratio_dir = output_path.parent / "tf_ratio"
    tfratio_dir.mkdir(parents=True, exist_ok=True)
    tfratio_out = tfratio_dir / output_path.name

    diff_s5_tf  = _smooth(_diff(bins_a_s5_tf,  bins_b_s5_tf),  5,  col="delta_logr")
    diff_s10_tf = _smooth(_diff(bins_a_s10_tf, bins_b_s10_tf), 10, col="delta_logr")

    # Temporarily swap title_l1 for the TF-ratio version
    _orig_title_l1 = title_l1
    title_l1 = title_l1_tf
    _render(bins_a_s5_tf,  bins_b_s5_tf,  diff_s5_tf,
            tfratio_out.with_name(tfratio_out.name + "_tfratio_smooth5"),
            "5-bin smooth [ichorCNA TF ratio]")
    _render(bins_a_s10_tf, bins_b_s10_tf, diff_s10_tf,
            tfratio_out.with_name(tfratio_out.name + "_tfratio_smooth10"),
            "10-bin smooth [ichorCNA TF ratio]")
    title_l1 = _orig_title_l1

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


        # Per-patient scaling diagnostics (after all pairs for this patient)
        _scaling_diagnostic_plots(
            patient        = str(patient),
            entity         = entity,
            samples        = samples,
            timepoints     = timepoints,
            tfs            = tfs,
            seg_files_bins = seg_files,
            seg_files_segs = seg_files_segs,
            output_dir     = output_dir,
        )
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
    """
    Return colours for focal amp trajectory lines and genome map boxes.

    Fixed 10-colour Red-Blue diverging palette. When n < 10, colours are
    picked from the outside in (positions 1, 10, 2, 9, 3, 8, ...) so the
    most visually distinct (most saturated) colours are used first,
    keeping the near-identical centre colours (#fddbc7, #d1e5f0) reserved
    for when there really are close to 10 regions.

    Beyond 10 regions, extra colours fall back to tab20 (the 10-per-patient
    cap makes this unreachable in practice).
    """
    _FIXED = [
        "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
        "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
    ]
    if n >= len(_FIXED):
        extra = n - len(_FIXED)
        cmap  = plt.colormaps.get_cmap("tab20")
        return _FIXED + [cmap(i) for i in range(extra)]

    # Outside-in index order: 0, 9, 1, 8, 2, 7, 3, 6, 4, 5
    order = []
    lo, hi = 0, len(_FIXED) - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo += 1
        hi -= 1
    selected_idx = sorted(order[:n])   # keep palette's natural colour order
    return [_FIXED[i] for i in selected_idx]


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
        total_reads:    list[int | None] | None = None,
        min_reads:      int   = 1_000_000,
        max_scale:      float = 3.0,
) -> tuple[bool, list[float]]:
    """
    For one patient, plot the relative copy-number trajectory of each focal
    amplification region across timepoints as line plots.

    Two figures are written:
      <patient>_focal_amp_trajectory_log2.pdf/.png
      <patient>_focal_amp_trajectory_cn.pdf/.png

    Samples are excluded from the plot if:
      - total_reads < min_reads (default 1,000,000)
      - empirical scale factor > max_scale (default 3.0)

    Returns (success: bool, scale_factors: list[float]) so the caller can
    use the computed scale factors for downstream plots.
    """
    if len(samples) < 2 or not focal_regions:
        print(f"        [SKIP] {patient}: {len(samples)} sample(s), "
              f"{len(focal_regions)} focal region(s) — need >= 2 samples "
              f"and >= 1 region")
        return False, []

    ref_tf = max(tfs)
    if ref_tf <= 0:
        print(f"        [SKIP] {patient}: max TF = {ref_tf} — no valid "
              f"reference sample")
        return False, []
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

    # ── Filter: drop samples that are too noisy to be useful ─────────────
    # 1. total_reads < min_reads  →  too few reads, signal unreliable
    # 2. scale_factor > max_scale →  huge TF difference, correction is
    #    speculative and amplifies noise
    reads_arr = total_reads if total_reads is not None else [None] * len(samples)
    keep_mask = []
    for i, (sample, m, reads) in enumerate(
            zip(samples, scale_factors, reads_arr)):
        if reads is not None and reads < min_reads:
            print(f"        [SKIP] {sample}: {reads:,} reads < "
                  f"{min_reads:,} — excluded from trajectory")
            keep_mask.append(False)
        elif m > max_scale:
            print(f"        [SKIP] {sample}: scale factor {m:.2f} > "
                  f"{max_scale:.1f} — excluded from trajectory")
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    # Always keep the reference timepoint
    keep_mask[ref_idx] = True

    samples    = [s for s, k in zip(samples,    keep_mask) if k]
    timepoints = [t for t, k in zip(timepoints, keep_mask) if k]
    tfs        = [f for f, k in zip(tfs,        keep_mask) if k]
    scale_factors = [m for m, k in zip(scale_factors, keep_mask) if k]

    if len(samples) < 2:
        print(f"        {patient}: fewer than 2 samples pass filters — skipping.")
        return False, []

    # Recompute ref_idx and ref_tf against the filtered sample list
    ref_tf  = max(tfs)
    ref_idx = tfs.index(ref_tf)

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
        print(f"        [SKIP] {patient}: no bin data could be loaded for "
              f"any sample/region combination")
        return False, []

    cn_matrix = 2.0 * np.power(2.0, log2_matrix)
    # 2^x is monotonic, so percentile bounds carry over directly
    cn_q1 = 2.0 * np.power(2.0, log2_q1)
    cn_q3 = 2.0 * np.power(2.0, log2_q3)

    output_dir.mkdir(parents=True, exist_ok=True)
    colors    = _trajectory_palette(len(focal_regions))
    x         = timepoints
    tp_labels = [f"TP{tp}" for tp in timepoints]

    # ── Write per-patient metadata TSV ───────────────────────────────────
    # Contains the information previously crammed into the plot title:
    # timepoint, sample, TF, reads, scale factor, CN per focal amp region.
    reads_arr = total_reads if total_reads is not None else [None] * len(samples)
    meta_rows = []
    for i, (samp, tp, tf, m) in enumerate(
            zip(samples, timepoints, tfs, scale_factors)):
        row: dict = {
            "patient":          patient,
            "sample":           samp,
            "timepoint":        tp,
            "tumor_fraction":   round(tf, 4),
            "scale_factor":     round(m, 4),
            "is_reference":     (m == 1.0),
            "total_reads":      reads_arr[i] if i < len(reads_arr) else None,
        }
        for j, (region, lbl) in enumerate(zip(focal_regions, labels)):
            cn_val  = round(float(cn_matrix[i, j]),  3) \
                      if not np.isnan(cn_matrix[i, j])  else None
            log_val = round(float(log2_matrix[i, j]), 3) \
                      if not np.isnan(log2_matrix[i, j]) else None
            row[f"log2r_{lbl}"] = log_val
            row[f"cn_{lbl}"]    = cn_val
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    meta_tsv = output_dir / f"{patient}_focal_amp_metadata.tsv"
    meta_df.to_csv(meta_tsv, sep="\t", index=False)

    # ── Short title for the plots (patient + entity only) ────────────────
    title_main = f"{patient}  ({entity})   Focal amplification trajectories"

    def _new_axes():
        fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(tp_labels)
        ax.set_xlabel("Timepoint", fontsize=9, color="#444444")
        ax.set_title(title_main, fontsize=10,
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

    # ── Genome map: CN profiles per sample with focal amp boxes ─────────
    _plot_focal_amp_genome_map(
        patient         = patient,
        entity          = entity,
        samples         = samples,
        timepoints      = timepoints,
        tfs             = tfs,
        scale_factors   = scale_factors,
        seg_files_bins  = seg_files_bins,
        seg_files_segs  = seg_files_segs,
        focal_regions   = focal_regions,
        labels          = labels,
        colors          = colors,
        output_dir      = output_dir,
    )

    return True, scale_factors


def _plot_focal_amp_genome_map(
        patient:        str,
        entity:         str,
        samples:        list[str],
        timepoints:     list[int],
        tfs:            list[float],
        scale_factors:  list[float],
        seg_files_bins: dict[str, Path],
        seg_files_segs: dict[str, Path] | None,
        focal_regions:  list[dict],
        labels:         list[str],
        colors:         list,
        output_dir:     Path,
        logr_ylim:      tuple[float, float] = (-2.5, 2.5),
) -> None:
    """
    For each timepoint: a genome-wide 50 kbp bin-level log2-ratio scatter
    (grey, semi-transparent), with segment means overlaid as blue horizontal
    step lines, and coloured upward triangle markers + short stems placed at
    the bottom of the plot (y ≈ logr_ylim[0] + 0.35) to indicate each focal
    amplification region without obscuring the CN data.

    Marker colours match the trajectory line-plot palette.  Gene name labels
    are printed just above each marker.  One panel per timepoint, stacked
    vertically.  Written as <patient>_focal_amp_genome_map.pdf/.png.
    """
    from .plots import _save as _plots_save
    import matplotlib.patches as mpatches_local

    n_tp    = len(samples)
    fig, axes = plt.subplots(
        n_tp, 1,
        figsize=(22, 3.5 * n_tp),
        facecolor="white",
        sharex=True,
        gridspec_kw={"hspace": 0.45},
        constrained_layout=True,
    )
    if n_tp == 1:
        axes = [axes]

    for ax_idx, (sample, tp, tf, m) in enumerate(
        zip(samples, timepoints, tfs, scale_factors)
    ):
        ax = axes[ax_idx]
        _style_ax(ax)
        ax.axhline(0,    color="#999999", linewidth=0.7, linestyle="-",  zorder=1)
        ax.axhline(0.58, color="#cccccc", linewidth=0.4, linestyle=":", zorder=1)
        ax.axhline(-0.58,color="#cccccc", linewidth=0.4, linestyle=":", zorder=1)

        # ── Plot 50 kbp bins (TF-corrected + common-scale factor) ────────
        bins_df = _load_bins(seg_files_bins.get(sample), sample, tf_from=tf)
        if bins_df is not None and not bins_df.empty and m != 1.0:
            bins_df = bins_df.copy()
            bins_df["logr_corrected"] = bins_df["logr_corrected"] * m

        seg_path = seg_files_bins.get(sample)
        segs_df  = None
        if seg_path is not None and sample in (seg_files_segs or {}):
            segs_df = _load_bins_with_seg_means(
                seg_path, seg_files_segs[sample], sample, tf_from=tf
            )
            if segs_df is not None and not segs_df.empty and m != 1.0:
                segs_df = segs_df.copy()
                segs_df["logr_corrected"] = segs_df["logr_corrected"] * m

        if bins_df is not None and not bins_df.empty:
            xv = bins_df.apply(
                lambda r: _genome_x(r["chrom"], (r["start"] + r["end"]) / 2),
                axis=1,
            ).values
            yv = np.clip(bins_df["logr_corrected"].values, *logr_ylim)
            ax.scatter(xv, yv, s=1.5, c="#888888", alpha=0.25,
                       linewidths=0, zorder=2, rasterized=True)

        # ── Segment means as step lines — blue normally, red in focal amps ──
        if segs_df is not None and not segs_df.empty:
            # Build a set of (chrom, start, end) intervals for focal regions
            focal_intervals: dict[str, list[tuple[int,int]]] = {}
            for region in focal_regions:
                focal_intervals.setdefault(region["chrom"], []).append(
                    (int(region["start"]), int(region["end"]))
                )

            def _in_focal(chrom: str, start: int, end: int) -> bool:
                for fs, fe in focal_intervals.get(chrom, []):
                    if start < fe and end > fs:
                        return True
                return False

            for chrom, grp in segs_df.groupby("chrom", sort=False):
                if chrom not in _OFFSETS:
                    continue
                grp = grp.sort_values("start")
                prev_val   = None
                seg_start  = None
                seg_in_foc = False
                for _, row in grp.iterrows():
                    val     = round(float(row["logr_corrected"]), 6)
                    x0r     = _genome_x(chrom, int(row["start"]))
                    x1r     = _genome_x(chrom, int(row["end"]))
                    in_foc  = _in_focal(chrom, int(row["start"]), int(row["end"]))
                    if val != prev_val or in_foc != seg_in_foc:
                        if prev_val is not None and seg_start is not None:
                            ax.plot(
                                [seg_start, x0r],
                                [np.clip(prev_val, *logr_ylim)] * 2,
                                color="#d7191c" if seg_in_foc else "#2c7bb6",
                                linewidth=1.6 if seg_in_foc else 1.2,
                                solid_capstyle="butt", zorder=3,
                            )
                        seg_start  = x0r
                        prev_val   = val
                        seg_in_foc = in_foc
                if prev_val is not None and seg_start is not None:
                    ax.plot(
                        [seg_start, x1r],
                        [np.clip(prev_val, *logr_ylim)] * 2,
                        color="#d7191c" if seg_in_foc else "#2c7bb6",
                        linewidth=1.6 if seg_in_foc else 1.2,
                        solid_capstyle="butt", zorder=3,
                    )

        # ── Focal amp markers at y = -2: triangle pointing up ────────────
        # Placed at a fixed low position so they don't obscure the CN data.
        marker_y = logr_ylim[0] + 0.35   # just above the bottom
        for region, label, color in zip(focal_regions, labels, colors):
            chrom = region["chrom"]
            if chrom not in _OFFSETS:
                continue
            x0    = _genome_x(chrom, region["start"])
            x1    = _genome_x(chrom, region["end"])
            x_mid = (x0 + x1) / 2
            width = max(x1 - x0, GENOME_SIZE * 0.002)

            # Filled triangle marker
            ax.scatter(
                [x_mid], [marker_y],
                marker="^", s=60, c=[color],
                edgecolors="#333333", linewidths=0.4,
                zorder=5, clip_on=True,
            )
            # Thin vertical stem from marker to just above it
            ax.vlines(
                x_mid, marker_y, marker_y + 0.35,
                colors=color, linewidth=1.0, zorder=4,
            )
            # Gene/coordinate label
            genes = str(region.get("cosmic_genes", "")).strip()
            gene_label = (genes.split(",")[0].strip()
                          if genes and genes.lower() != "nan" else label)
            ax.text(
                x_mid, marker_y + 0.40,
                gene_label,
                fontsize=5.5, color=color, ha="center", va="bottom",
                fontweight="bold", zorder=6, clip_on=True,
            )

        ax.set_ylim(logr_ylim)
        ax.set_ylabel("log2R", fontsize=8, color="#444444", labelpad=3)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.set_title(
            f"TP{tp}  —  {sample}  (TF={tf:.1%}, ×{m:.2f})",
            fontsize=9, fontweight="bold", color="#222222",
            loc="left", pad=4,
        )

    # shared legend
    legend_handles = [
        plt.Line2D([0], [0], color="#2c7bb6", linewidth=1.2,
                   label="Segment mean"),
        plt.Line2D([0], [0], color="#d7191c", linewidth=1.6,
                   label="Segment mean (focal amp region)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=5, alpha=0.5, label="50 kbp bins (raw)"),
    ] + [
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=c, markeredgecolor="#333333",
                   markersize=7, label=lbl)
        for c, lbl in zip(colors, labels)
    ]
    axes[0].legend(
        handles=legend_handles,
        fontsize=7.5,
        loc="upper right",
        ncol=min(len(legend_handles), 4),
        frameon=True, facecolor="white", edgecolor="#cccccc",
        labelcolor="#222222",
        title="▲ = focal amp location (at bottom of plot)",
        title_fontsize=7.0,
    )

    fig.suptitle(
        f"{patient}  ({entity})   Focal amplification genome map",
        fontsize=11, fontweight="bold", color="#222222",
    )
    # constrained_layout handles spacing automatically
    _plots_save(fig, output_dir / f"{patient}_focal_amp_genome_map")


def plot_all_focal_amp_trajectories(
        focal_tsv:      Path,
        tf_df:          pd.DataFrame,
        seg_files_bins: dict[str, Path],
        output_dir:     Path,
        min_tf:         float = 0.10,
        min_timepoints: int   = 2,
        seg_files_segs: dict[str, Path] | None = None,
        min_reads:      int   = 1_000_000,
        max_scale:      float = 3.0,
) -> int:
    """
    Batch driver: for every patient with >= min_timepoints qualifying
    timepoints AND at least one focal amplification event, generate:
      1. log2 and relative-CN trajectory plots
      2. genome map with per-timepoint CN profiles and focal amp markers
      3. overlay CN profile plot — all corrected segment mean profiles on
         one genome-wide panel, one line per timepoint, different colours

    Filters applied per timepoint:
      - tumor_fraction >= min_tf  (default 0.10)
      - total_reads   >= min_reads (default 1,000,000)
      - scale_factor  <= max_scale (default 3.0)

    Returns number of patients for which plots were generated.
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

    n_patients_total   = df["patient"].nunique()
    n_patients_seg     = df[df["sample"].isin(seg_files_bins)]["patient"].nunique()
    print(f"      [focal trajectories] {n_patients_total} patients with "
          f"TF >= {min_tf:.0%} in tf_df; {n_patients_seg} have seg files")
    print(f"      [focal trajectories] {len(focal_samples)} samples with "
          f"focal amp events in focal_amplifications.tsv")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0

    for patient, pat_df in df.groupby("patient"):
        pat_df = pat_df.sort_values("timepoint")
        n_tp = pat_df["timepoint"].nunique()
        if n_tp < min_timepoints:
            print(f"      [SKIP] {patient}: only {n_tp} qualifying timepoint(s) "
                  f"(need >= {min_timepoints}, TF >= {min_tf:.0%})")
            continue

        pat_samples = set(pat_df["sample"])
        if not (pat_samples & focal_samples):
            print(f"      [SKIP] {patient}: no focal amplifications detected "
                  f"in any of {len(pat_samples)} qualifying sample(s)")
            continue

        pat_focal = focal_df[focal_df["sample"].isin(pat_samples)]
        regions = [
            {
                "chrom":        row["chrom"],
                "start":        int(row["start"]),
                "end":          int(row["end"]),
                "cosmic_genes": row["cosmic_genes"],
                "logr":         float(row.get("logr", row.get("seg.mean",
                                    row.get("log2_ratio", 0.0))) or 0.0),
            }
            for _, row in pat_focal.iterrows()
        ]
        regions = _merge_overlapping(regions)
        if not regions:
            print(f"      [SKIP] {patient}: focal amp events found but no "
                  f"regions survived merging/filtering")
            continue

        # Cap at 10 regions (matches the 10-colour palette), selecting
        # the highest copy-number (logR) regions.
        if len(regions) > 10:
            def _region_logr(r: dict) -> float:
                return float(r.get("logr", 0.0))
            regions = sorted(regions, key=_region_logr, reverse=True)[:10]

        entity      = entity_from_sample(str(patient))
        reads_list  = pat_df["total_reads"].tolist() \
                      if "total_reads" in pat_df.columns else None

        ok, scale_factors = plot_focal_amp_trajectories(
            patient        = str(patient),
            entity         = entity,
            samples        = pat_df["sample"].tolist(),
            timepoints     = pat_df["timepoint"].tolist(),
            tfs            = pat_df["tumor_fraction"].tolist(),
            seg_files_bins = seg_files_bins,
            focal_regions  = regions,
            output_dir     = output_dir,
            seg_files_segs = seg_files_segs,
            total_reads    = reads_list,
            min_reads      = min_reads,
            max_scale      = max_scale,
        )
        if ok:
            n_plots += 1
            print(f"      [OK]   {patient}: {len(regions)} focal amp region(s) "
                  f"across {pat_df['timepoint'].nunique()} timepoints "
                  f"({len(scale_factors)} pass all filters)")

            # Overlay CN profile plot (filtered samples only)
            # Re-derive which samples passed the filters using the returned
            # scale_factors length (matches the filtered sample list)
            n_passing = len(scale_factors)
            passing_df = pat_df.iloc[:n_passing] \
                         if n_passing <= len(pat_df) else pat_df
            # More robust: re-filter by reads and scale factors
            _plot_focal_amp_cn_overlay(
                patient        = str(patient),
                entity         = entity,
                samples        = pat_df["sample"].tolist(),
                timepoints     = pat_df["timepoint"].tolist(),
                tfs            = pat_df["tumor_fraction"].tolist(),
                total_reads    = reads_list,
                scale_factors  = scale_factors,
                seg_files_segs = seg_files_segs,
                focal_regions  = regions,
                output_dir     = output_dir,
                min_reads      = min_reads,
                max_scale      = max_scale,
            )
        gc.collect()

    return n_plots


def _plot_focal_amp_cn_overlay(
        patient:        str,
        entity:         str,
        samples:        list[str],
        timepoints:     list[int],
        tfs:            list[float],
        total_reads:    list | None,
        scale_factors:  list[float],
        seg_files_segs: dict[str, Path] | None,
        focal_regions:  list[dict],
        output_dir:     Path,
        min_reads:      int   = 1_000_000,
        max_scale:      float = 3.0,
        logr_ylim:      tuple[float, float] = (-2.5, 2.5),
) -> None:
    """
    Single genome-wide panel overlaying the TF-corrected segment mean CN
    profiles for all qualifying timepoints of one patient, with focal amp
    markers at the bottom.  One line per timepoint, coloured from a
    sequential palette (distinct from the focal amp trajectory palette).

    Samples that fail the reads or scale-factor filter are excluded.

    Written as <patient>_focal_amp_cn_overlay.pdf/.png.
    """
    from .plots import _save as _plots_save
    import matplotlib.patches as mpatches_local

    # Palette for overlay timepoint lines — sequential blue-orange scheme,
    # deliberately different from the Red-Blue focal amp colours
    _OVERLAY_COLORS = [
        "#1b7837", "#762a83", "#e08214", "#2166ac",
        "#d6604d", "#35978f", "#bf812d", "#80cdc1",
    ]

    # Re-apply the same filters as plot_focal_amp_trajectories to
    # get the matching (sample, tp, tf, scale_factor) tuples.
    reads_arr = total_reads if total_reads is not None else [None] * len(samples)
    ref_tf  = max(tfs)
    ref_idx = tfs.index(ref_tf)

    # scale_factors list has already been filtered — we need to reconstruct
    # which original samples passed.  Re-derive from the same logic:
    # ref always passes; others pass if reads >= min_reads and m <= max_scale.
    # We use scale_factors as provided (already computed & filtered upstream).
    filt_samples, filt_tps, filt_tfs, filt_m, filt_reads = [], [], [], [], []
    sf_iter = iter(scale_factors)
    for i, (samp, tp, tf, reads) in enumerate(
            zip(samples, timepoints, tfs, reads_arr)):
        passes_reads = (reads is None or reads >= min_reads)
        m_val = 1.0 if i == ref_idx else None
        if i == ref_idx:
            m_val = 1.0
        else:
            try:
                m_val = next(sf_iter)
            except StopIteration:
                break
        passes_scale = (m_val <= max_scale)
        if (i == ref_idx or (passes_reads and passes_scale)):
            filt_samples.append(samp)
            filt_tps.append(tp)
            filt_tfs.append(tf)
            filt_m.append(m_val if m_val is not None else 1.0)
            filt_reads.append(reads)

    if len(filt_samples) < 1 or seg_files_segs is None:
        return

    focal_colors = _trajectory_palette(len(focal_regions))
    focal_labels = []
    seen: dict[str, int] = {}
    for region in focal_regions:
        lbl = _region_label(region)
        if lbl in seen:
            seen[lbl] += 1
            lbl = f"{lbl} ({seen[lbl]})"
        else:
            seen[lbl] = 0
        focal_labels.append(lbl)

    # Build a set of focal amp intervals for red colouring of segs
    focal_intervals: dict[str, list[tuple[int,int]]] = {}
    for region in focal_regions:
        focal_intervals.setdefault(region["chrom"], []).append(
            (int(region["start"]), int(region["end"]))
        )

    def _in_focal(chrom: str, start: int, end: int) -> bool:
        for fs, fe in focal_intervals.get(chrom, []):
            if start < fe and end > fs:
                return True
        return False

    n_colors = len(_OVERLAY_COLORS)

    fig, ax = plt.subplots(figsize=(22, 4.5), facecolor="white")
    _style_ax(ax)
    ax.axhline(0,     color="#999999", linewidth=0.7, linestyle="-",  zorder=1)
    ax.axhline(0.58,  color="#dddddd", linewidth=0.4, linestyle=":", zorder=1)
    ax.axhline(-0.58, color="#dddddd", linewidth=0.4, linestyle=":", zorder=1)

    line_handles = []
    for idx, (samp, tp, tf, m) in enumerate(
            zip(filt_samples, filt_tps, filt_tfs, filt_m)):
        segs_path = seg_files_segs.get(samp)
        seg_path  = None  # bins not needed — seg means only
        # Load seg means using _load_bins_with_seg_means with dummy bins
        # Actually use parse_seg_file to get segment means directly
        from ..common.io import parse_seg_file
        seg_df = parse_seg_file(segs_path, samp) if segs_path else None
        if seg_df is None or seg_df.empty:
            continue

        # Apply TF correction (scale by m)
        logr_col = None
        for c in ["logr", "seg.mean", "log2_ratio", "median"]:
            if c in seg_df.columns:
                logr_col = c
                break
        if logr_col is None:
            continue

        line_color = _OVERLAY_COLORS[idx % n_colors]

        for _, row in seg_df.iterrows():
            chrom = str(row["chrom"])
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            if chrom not in _OFFSETS:
                continue
            val = float(row[logr_col]) * m
            x0  = _genome_x(chrom, int(row["start"]))
            x1  = _genome_x(chrom, int(row["end"]))
            in_foc = _in_focal(chrom, int(row["start"]), int(row["end"]))
            # Focal amp segments: draw in red on top; otherwise line colour
            col = "#d7191c" if in_foc else line_color
            lw  = 2.0      if in_foc else 1.2
            ax.plot([x0, x1], [np.clip(val, *logr_ylim)] * 2,
                    color=col, linewidth=lw,
                    solid_capstyle="butt", zorder=4 if in_foc else 3,
                    alpha=0.9)

        reads_str = (f"{int(filt_reads[idx]/1e6):.1f}M reads"
                     if filt_reads[idx] is not None else "")
        label = f"TP{tp}  TF={tf:.1%}  ×{m:.2f}  {reads_str}".strip()
        line_handles.append(
            plt.Line2D([0], [0], color=line_color, linewidth=2.0, label=label)
        )

    # Focal amp markers at the bottom
    marker_y = logr_ylim[0] + 0.35
    for region, lbl, fc in zip(focal_regions, focal_labels, focal_colors):
        chrom = region["chrom"]
        if chrom not in _OFFSETS:
            continue
        x0   = _genome_x(chrom, region["start"])
        x1   = _genome_x(chrom, region["end"])
        xmid = (x0 + x1) / 2
        ax.scatter([xmid], [marker_y], marker="^", s=55,
                   c=[fc], edgecolors="#333333", linewidths=0.4,
                   zorder=5, clip_on=True)
        ax.vlines(xmid, marker_y, marker_y + 0.35,
                  colors=fc, linewidth=1.0, zorder=4)
        genes = str(region.get("cosmic_genes", "")).strip()
        gene_label = (genes.split(",")[0].strip()
                      if genes and genes.lower() != "nan" else lbl)
        ax.text(xmid, marker_y + 0.42, gene_label,
                fontsize=5.5, color=fc, ha="center", va="bottom",
                fontweight="bold", zorder=6, clip_on=True)

    ax.set_ylim(logr_ylim)
    ax.set_ylabel("TF-corrected log2 ratio", fontsize=9,
                  color="#444444", labelpad=5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Legend: timepoint lines + focal amp triangles
    focal_handles = [
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=fc, markeredgecolor="#333333",
                   markersize=7, label=lbl)
        for fc, lbl in zip(focal_colors, focal_labels)
    ]
    all_handles = line_handles + [
        plt.Line2D([0], [0], color="#d7191c", linewidth=2.0,
                   label="Focal amp segment (all TPs)")
    ] + focal_handles
    ax.legend(handles=all_handles, fontsize=7, loc="upper right",
              ncol=max(1, len(all_handles) // 4),
              frameon=True, facecolor="white", edgecolor="#cccccc",
              labelcolor="#222222",
              title=f"{patient}  ({entity})  overlay — one line per timepoint",
              title_fontsize=7.5)

    fig.suptitle(
        f"{patient}  ({entity})   CN profile overlay — all qualifying timepoints",
        fontsize=10, fontweight="bold", color="#222222",
    )
    fig.tight_layout()
    _plots_save(fig, output_dir / f"{patient}_focal_amp_cn_overlay")


# ---------------------------------------------------------------------------
# Longitudinal scaling diagnostic plots
# ---------------------------------------------------------------------------

def _scaling_diagnostic_plots(
        patient:        str,
        entity:         str,
        samples:        list[str],
        timepoints:     list[int],
        tfs:            list[float],
        seg_files_bins: dict[str, Path],
        seg_files_segs: dict[str, Path] | None,
        output_dir:     Path,
        min_abs:        float = 0.05,
        logr_ylim:      tuple[float, float] = (-2.5, 2.5),
) -> None:
    """
    Two per-patient diagnostic plots for the empirical TF-correction scaling.

    Plot 1: Scatter — log2 ratio before vs after correction (one panel per
    timepoint pair, coloured by timepoint). Shows how well the scaling aligns
    the two samples. Reference (highest-TF) timepoint is grey.

    Plot 2: Weight distribution — genome-wide scatter of the WLS weights
    (w = |logR_reference|) used in the scale-factor fit. Reveals which
    genomic regions drive the estimate.

    Both plots are written to <output_dir>/scaling_diagnostics/.
    """
    from .plots import _save as _plots_save

    if len(samples) < 2:
        return

    ref_tf  = max(tfs)
    ref_idx = tfs.index(ref_tf)
    ref_samp = samples[ref_idx]

    # ── Colour palette for timepoints (avoid red/blue reserved for CN) ───
    _TP_COLORS = [
        "#1b7837", "#762a83", "#e08214", "#2166ac",
        "#d6604d", "#35978f", "#bf812d", "#80cdc1",
    ]
    n = len(samples)
    tp_colors = {s: ("#bbbbbb" if i == ref_idx
                     else _TP_COLORS[(i if i < ref_idx else i - 1) % len(_TP_COLORS)])
                 for i, s in enumerate(samples)}

    diag_dir = output_dir / "scaling_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Load reference bins (raw, unscaled)
    ref_seg_path  = seg_files_bins.get(ref_samp)
    ref_segs_path = seg_files_segs.get(ref_samp) if seg_files_segs else None
    if ref_seg_path is None:
        return

    ref_bins = _load_bins(ref_seg_path, ref_samp, tf_from=ref_tf)
    if ref_bins is None or ref_bins.empty:
        return

    # Load reference segment means for weight computation and scaling fit
    ref_segmeans = None
    if ref_segs_path is not None:
        ref_segmeans = _load_bins_with_seg_means(
            ref_seg_path, ref_segs_path, ref_samp, tf_from=ref_tf
        )

    # ── Compute per-non-ref timepoint scale factors and aligned data ─────
    # We store: (sample, tp, tf, raw_aligned_df, scaled_aligned_df, weight_df, m)
    comparisons = []
    for i, (samp, tp, tf) in enumerate(zip(samples, timepoints, tfs)):
        if i == ref_idx:
            continue
        seg_path  = seg_files_bins.get(samp)
        segs_path = seg_files_segs.get(samp) if seg_files_segs else None
        if seg_path is None:
            continue

        samp_bins = _load_bins(seg_path, samp, tf_from=tf)
        if samp_bins is None or samp_bins.empty:
            continue

        # Align reference and sample segment means for scaling
        m = 1.0
        if ref_segmeans is not None and segs_path is not None:
            samp_segmeans = _load_bins_with_seg_means(
                seg_path, segs_path, samp, tf_from=tf
            )
            if samp_segmeans is not None and not samp_segmeans.empty:
                aligned = _align_bins(ref_segmeans, samp_segmeans)
                if not aligned.empty:
                    m, _ = _optimise_scaling(
                        aligned["logr_a"].values,
                        aligned["logr_b"].values,
                        min_abs=min_abs,
                    )

        # Align raw bins for scatter plots
        raw_aligned = _align_bins(ref_bins, samp_bins)
        if raw_aligned.empty:
            continue

        # Scaled version
        samp_bins_scaled = samp_bins.copy()
        samp_bins_scaled["logr_corrected"] *= m
        scaled_aligned = _align_bins(ref_bins, samp_bins_scaled)

        # Weights: |logR_reference| on the segment-mean aligned data
        if ref_segmeans is not None and segs_path is not None:
            samp_segmeans = _load_bins_with_seg_means(
                seg_path, segs_path, samp, tf_from=tf
            )
            if samp_segmeans is not None and not samp_segmeans.empty:
                w_aligned = _align_bins(ref_segmeans, samp_segmeans)
                w_aligned["weight"] = np.abs(w_aligned["logr_a"])
                w_aligned = w_aligned[w_aligned["weight"] >= min_abs]
            else:
                w_aligned = None
        else:
            # Fall back to raw bin weights
            w_aligned = raw_aligned.copy()
            w_aligned["weight"] = np.abs(w_aligned["logr_a"])
            w_aligned = w_aligned[w_aligned["weight"] >= min_abs]

        comparisons.append((samp, tp, tf, raw_aligned, scaled_aligned,
                             w_aligned, m))

    if not comparisons:
        return

    # ── Plot 1: Before/After correction scatter ──────────────────────────
    # One row per non-reference timepoint, two columns (before / after).
    n_comp = len(comparisons)
    fig, axes = plt.subplots(
        n_comp, 2,
        figsize=(9, 3.5 * n_comp),
        facecolor="white",
        constrained_layout=True,
    )
    if n_comp == 1:
        axes = [axes]   # make always a list of rows

    for row_idx, (samp, tp, tf, raw_al, scl_al, _, m) in enumerate(comparisons):
        color = tp_colors[samp]
        ref_label = f"TP{timepoints[ref_idx]} (ref, TF={ref_tf:.1%})"
        samp_label = f"TP{tp}  TF={tf:.1%}"

        for col_idx, (al, ttl) in enumerate([
            (raw_al,  f"Before correction  (m=1.00)\n{samp_label}"),
            (scl_al,  f"After correction   (m={m:.2f})\n{samp_label}"),
        ]):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.5)

            lf = np.clip(al["logr_a"].values, *logr_ylim)
            la = np.clip(al["logr_b"].values, *logr_ylim)
            ax.scatter(lf, la, s=1.5, c=color, alpha=0.3,
                       linewidths=0, rasterized=True, zorder=2)

            # y=x reference line
            lim = logr_ylim
            ax.plot(lim, lim, color="#999999", linewidth=0.8,
                    linestyle="--", zorder=3)
            # Regression line through origin (slope = m_applied)
            m_applied = 1.0 if col_idx == 0 else m
            ax.plot(lim, [v * m_applied for v in lim],
                    color=color, linewidth=1.2, linestyle="-", zorder=4)

            ax.set_xlim(logr_ylim); ax.set_ylim(logr_ylim)
            ax.set_xlabel(f"log2R  {ref_label}", fontsize=7.5,
                          color="#444444")
            ax.set_ylabel(f"log2R  {samp_label}", fontsize=7.5,
                          color="#444444")
            ax.set_title(ttl, fontsize=8, color="#222222",
                         fontweight="bold", loc="left")
            ax.axhline(0, color="#cccccc", linewidth=0.5)
            ax.axvline(0, color="#cccccc", linewidth=0.5)

            # Correlation annotation
            mask = np.isfinite(lf) & np.isfinite(la)
            if mask.sum() > 10:
                r = float(np.corrcoef(lf[mask], la[mask])[0, 1])
                ax.text(0.97, 0.04, f"r = {r:.3f}", transform=ax.transAxes,
                        fontsize=7, ha="right", va="bottom", color="#444444")

    fig.suptitle(
        f"{patient}  ({entity})   Log2R before/after TF correction",
        fontsize=10, fontweight="bold", color="#222222",
    )
    _plots_save(fig, diag_dir / f"{patient}_scaling_scatter")

    # ── Plot 2: Weight distribution genome-wide ──────────────────────────
    # One panel per non-reference timepoint: x = genome position,
    # y = WLS weight (|logR_reference|) at the segment level.
    fig2, axes2 = plt.subplots(
        n_comp, 1,
        figsize=(22, 3.0 * n_comp),
        facecolor="white",
        sharex=True,
        constrained_layout=True,
    )
    if n_comp == 1:
        axes2 = [axes2]

    for row_idx, (samp, tp, tf, _, _, w_al, m) in enumerate(comparisons):
        ax = axes2[row_idx]
        _style_ax(ax)
        ax.axhline(min_abs, color="#cccccc", linewidth=0.6,
                   linestyle=":", zorder=1,
                   label=f"min_abs threshold ({min_abs})")

        color = tp_colors[samp]

        if w_al is not None and not w_al.empty:
            xv = w_al.apply(
                lambda r: _genome_x(r["chrom"],
                                    (r["start"] + r["end"]) / 2), axis=1
            ).values
            yv = w_al["weight"].values
            ax.scatter(xv, yv, s=2.5, c=color, alpha=0.5,
                       linewidths=0, zorder=2, rasterized=True)

        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"Weight\n|logR ref|", fontsize=8,
                      color="#444444", labelpad=4)
        ax.set_title(
            f"TP{tp}  {samp}  (TF={tf:.1%}, scale ×{m:.2f})"
            f"  —  WLS weights used in scale-factor fit",
            fontsize=8, fontweight="bold", color="#222222", loc="left",
        )

    fig2.suptitle(
        f"{patient}  ({entity})   WLS weight distribution",
        fontsize=10, fontweight="bold", color="#222222",
    )
    _plots_save(fig2, diag_dir / f"{patient}_scaling_weights")


# ---------------------------------------------------------------------------
# Gene-centric focal amplification trajectory plot
# ---------------------------------------------------------------------------

def plot_gene_focal_trajectories(
        gene:           str,
        focal_tsv:      Path | None,
        tf_df:          pd.DataFrame,
        seg_files_bins: dict[str, Path],
        output_dir:     Path,
        seg_files_segs: dict[str, Path] | None = None,
        entity_filter:  str | None = None,
        min_tf:         float = 0.10,
        min_reads:      int   = 1_000_000,
        max_scale:      float = 3.0,
        min_timepoints: int   = 2,
        logr_ylim:      tuple[float, float] = (-0.5, 3.5),
        fixed_region:   dict | None = None,
        patient_filter: list[str] | None = None,
) -> int:
    """
    Gene- or region-centric CN trajectory plot.

    Two modes:
      Gene mode  (fixed_region=None): searches focal_tsv for samples with
                 cosmic_genes matching `gene`, derives patient-specific
                 focal amp coordinates per patient.
      Region mode (fixed_region set): uses the supplied region dict
                 {chrom, start, end, cosmic_genes} for ALL patients.
                 focal_tsv is used only to identify which patients have
                 any focal amp overlapping the region (optional — if None,
                 all patients with >= min_timepoints qualifying timepoints
                 are included).
    """

    from .plots import _save as _plots_save

    # ── Region mode: fixed_region supplied directly ───────────────────────
    if fixed_region is not None:
        gene_region  = fixed_region
        gene_samples: set[str] = set()   # empty = include all patients
        # If focal_tsv supplied, restrict to patients overlapping the region
        if focal_tsv is not None and focal_tsv.exists():
            try:
                fdf = pd.read_csv(focal_tsv, sep="\t", low_memory=False)
                fdf["chrom"] = fdf["chrom"].astype(str).apply(
                    lambda c: c if c.startswith("chr") else f"chr{c}")
                overlap = fdf[
                    (fdf["chrom"] == gene_region["chrom"]) &
                    (fdf["start"] < gene_region["end"]) &
                    (fdf["end"]   > gene_region["start"])
                ]
                gene_samples = set(overlap["sample"].unique())
                print(f"  [gene trajectories] Region {gene}: "
                      f"{len(gene_samples)} samples with overlapping focal amp")
            except Exception as e:
                print(f"  [gene trajectories] [WARN] Could not read focal TSV: {e}")
        if not gene_samples:
            print(f"  [gene trajectories] No focal-tsv or no overlap found — "
                  f"including ALL patients with >= {min_timepoints} timepoints")
        gene_df      = pd.DataFrame()   # not used in region mode
        merged       = [gene_region]

    # ── Gene mode: look up focal_tsv by gene name ─────────────────────────
    else:
        if focal_tsv is None or not focal_tsv.exists():
            print(f"  [gene trajectories] focal_amplifications.tsv not found.")
            return 0

        focal_df = pd.read_csv(focal_tsv, sep="\t", low_memory=False)
        if focal_df.empty or "cosmic_genes" not in focal_df.columns:
            print(f"  [gene trajectories] No focal amp data or no cosmic_genes column.")
            return 0

        gene_mask = (
            focal_df["cosmic_genes"]
            .fillna("").astype(str)
            .str.lower()
            .str.contains(gene.lower(), regex=False)
        )
        gene_df = focal_df[gene_mask].copy()
        if gene_df.empty:
            print(f"  [gene trajectories] No focal amps found for gene '{gene}'.")
            return 0

        gene_df["chrom"] = gene_df["chrom"].astype(str).apply(
            lambda c: c if c.startswith("chr") else f"chr{c}")

        print(f"  [gene trajectories] '{gene}': {len(gene_df)} events in "
              f"{gene_df['sample'].nunique()} samples")

        candidate_regions = [
            {"chrom": r["chrom"], "start": int(r["start"]), "end": int(r["end"]),
             "cosmic_genes": str(r.get("cosmic_genes", ""))}
            for _, r in gene_df.iterrows()
        ]
        merged = _merge_overlapping(candidate_regions)
        if not merged:
            print(f"  [gene trajectories] No valid regions after merging for '{gene}'.")
            return 0
        gene_region = max(merged, key=lambda r: r["end"] - r["start"])
        gene_samples = set(gene_df["sample"].unique())

        print(f"  [gene trajectories] Region: {gene_region['chrom']}:"
              f"{gene_region['start']:,}-{gene_region['end']:,}")

    # Filter tf_df to qualifying timepoints
    df = tf_df.dropna(subset=["timepoint", "tumor_fraction"]).copy()
    df = df[
        (df["tumor_fraction"] >= min_tf) &
        df["sample"].isin(seg_files_bins)
    ]
    df["timepoint"] = df["timepoint"].astype(int)

    # Ensure entity column is present — derive from patient ID if missing
    if "entity" not in df.columns or df["entity"].isna().all():
        df["entity"] = df["patient"].apply(
            lambda p: entity_from_sample(str(p))
        )

    # Optionally filter by entity
    if entity_filter:
        mask = df["entity"].str.lower() == entity_filter.lower()
        if not mask.any():
            # Also try capitalised form in case entity comes from sample name
            mask = df["entity"].str.lower().str.contains(
                entity_filter.lower(), regex=False
            )
        df = df[mask]
        if df.empty:
            print(f"  [gene trajectories] No qualifying samples for entity "
                  f"'{entity_filter}'.")
            return 0
        print(f"  [gene trajectories] Entity filter '{entity_filter}': "
              f"{df['patient'].nunique()} patients")

    # Find reads column
    has_reads = "total_reads" in df.columns

    # Keep only patients who have at least one sample with the gene amp
    if patient_filter:
        print(f"  [gene trajectories] Patient filter: {patient_filter}")

    qualifying_patients = []
    for patient, pat_df in df.groupby("patient"):
        if patient_filter and str(patient) not in patient_filter:
            continue
        pat_df = pat_df.sort_values("timepoint")
        # Gene mode: skip patients without the gene amp.
        # Region mode with empty gene_samples: include all patients.
        if gene_samples and not (set(pat_df["sample"]) & gene_samples):
            continue
        if pat_df["timepoint"].nunique() < min_timepoints:
            continue
        qualifying_patients.append((patient, pat_df))

    if not qualifying_patients:
        print(f"  [gene trajectories] No patients with >= {min_timepoints} "
              f"qualifying timepoint(s) and '{gene}' amplification.")
        return 0

    print(f"  [gene trajectories] {len(qualifying_patients)} patients with "
          f"'{gene}' amplification.")

    # ── For each patient: compute common-reference scale factors ─────────
    # Use the same logic as plot_focal_amp_trajectories
    patient_data: list[dict] = []

    for patient, pat_df in qualifying_patients:
        samples    = pat_df["sample"].tolist()
        timepoints = pat_df["timepoint"].tolist()
        tfs        = pat_df["tumor_fraction"].tolist()
        reads_list = pat_df["total_reads"].tolist() if has_reads else [None]*len(samples)

        ref_tf  = max(tfs)
        ref_idx = tfs.index(ref_tf)
        ref_samp = samples[ref_idx]

        # Compute scale factors relative to highest-TF timepoint
        scale_factors = [1.0] * len(samples)
        ref_seg_path  = seg_files_bins.get(ref_samp)
        ref_segs_path = (seg_files_segs or {}).get(ref_samp)
        ref_segmeans  = None
        if ref_seg_path is not None and ref_segs_path is not None:
            ref_segmeans = _load_bins_with_seg_means(
                ref_seg_path, ref_segs_path, ref_samp, tf_from=ref_tf
            )

        for i, (samp, tf) in enumerate(zip(samples, tfs)):
            if i == ref_idx:
                continue
            m = None
            seg_path  = seg_files_bins.get(samp)
            segs_path = (seg_files_segs or {}).get(samp)
            if ref_segmeans is not None and seg_path is not None and segs_path is not None:
                samp_segmeans = _load_bins_with_seg_means(
                    seg_path, segs_path, samp, tf_from=tf
                )
                if samp_segmeans is not None and not samp_segmeans.empty:
                    aligned = _align_bins(ref_segmeans, samp_segmeans)
                    if not aligned.empty:
                        m, _ = _optimise_scaling(
                            aligned["logr_a"].values, aligned["logr_b"].values
                        )
            if m is None:
                m = (ref_tf / tf) if tf > 0 else 1.0
            scale_factors[i] = m

        # Apply reads + scale filter (keep reference always)
        keep = []
        for i, (samp, tf, m, reads) in enumerate(
                zip(samples, timepoints, scale_factors, reads_list)):
            if i == ref_idx:
                keep.append(True)
                continue
            if reads is not None and reads < min_reads:
                keep.append(False)
            elif scale_factors[i] > max_scale:
                keep.append(False)
            else:
                keep.append(True)

        samples_f    = [s for s, k in zip(samples,       keep) if k]
        timepoints_f = [t for t, k in zip(timepoints,    keep) if k]
        tfs_f        = [f for f, k in zip(tfs,           keep) if k]
        scales_f     = [m for m, k in zip(scale_factors, keep) if k]

        if not samples_f:
            continue

        # Determine this patient's own focal amp region for the gene.
        # In region mode (fixed_region set), use the fixed region for all patients.
        if fixed_region is not None:
            pat_region = gene_region
        else:
            pat_amp_rows = gene_df[gene_df["sample"].isin(set(samples_f) & gene_samples)]
            if pat_amp_rows.empty:
                pat_amp_rows = gene_df[gene_df["sample"].isin(set(samples) & gene_samples)]
            if not pat_amp_rows.empty:
                pat_regions = [
                    {"chrom": r["chrom"], "start": int(r["start"]), "end": int(r["end"]),
                     "cosmic_genes": str(r.get("cosmic_genes", ""))}
                    for _, r in pat_amp_rows.iterrows()
                ]
                pat_merged  = _merge_overlapping(pat_regions)
                pat_region  = max(pat_merged, key=lambda r: r["end"] - r["start"]) \
                              if pat_merged else gene_region
            else:
                pat_region = gene_region

        # Compute log2 + IQR for this patient's focal amp region at each timepoint
        log2_vals, q1_vals, q3_vals = [], [], []
        for samp, tf, m in zip(samples_f, tfs_f, scales_f):
            seg_path = seg_files_bins.get(samp)
            if seg_path is None:
                log2_vals.append(np.nan); q1_vals.append(np.nan); q3_vals.append(np.nan)
                continue
            bins_df = _load_bins(seg_path, samp, tf_from=tf)
            if bins_df is None or bins_df.empty:
                log2_vals.append(np.nan); q1_vals.append(np.nan); q3_vals.append(np.nan)
                continue
            if m != 1.0:
                bins_df = bins_df.copy()
                bins_df["logr_corrected"] *= m
            mean, q1, q3 = _region_logr_stats(bins_df, pat_region)
            log2_vals.append(mean); q1_vals.append(q1); q3_vals.append(q3)

        # Drop timepoints where no bin data was available (NaN log2)
        # and renumber the remaining ones starting from 1 so all patients
        # are on the same relative time axis regardless of their actual
        # timepoint indices.
        valid_mask = [not np.isnan(v) for v in log2_vals]
        samples_f    = [s for s, k in zip(samples_f,    valid_mask) if k]
        tfs_f        = [f for f, k in zip(tfs_f,        valid_mask) if k]
        scales_f     = [m for m, k in zip(scales_f,     valid_mask) if k]
        orig_tps     = [t for t, k in zip(timepoints_f, valid_mask) if k]
        log2_vals    = [v for v, k in zip(log2_vals, valid_mask) if k]
        q1_vals      = [v for v, k in zip(q1_vals,   valid_mask) if k]
        q3_vals      = [v for v, k in zip(q3_vals,   valid_mask) if k]

        if not samples_f:
            continue

        # Renumber: 1, 2, 3, ... regardless of original timepoint labels
        norm_tps = list(range(1, len(samples_f) + 1))

        if len(norm_tps) < min_timepoints:
            print(f"        [SKIP] {patient}: only {len(norm_tps)} timepoint(s) "
                  f"with valid bin data after filtering (need >= {min_timepoints})")
            continue

        cn_vals  = [2.0 * 2**v for v in log2_vals]
        cn_q1v   = [2.0 * 2**v for v in q1_vals]
        cn_q3v   = [2.0 * 2**v for v in q3_vals]

        patient_data.append({
            "patient":      str(patient),
            "samples":      samples_f,
            "timepoints":   norm_tps,       # normalised: 1, 2, 3, ...
            "orig_tps":     orig_tps,       # original timepoint indices (for metadata)
            "tfs":          tfs_f,
            "scales":       scales_f,
            "log2":         log2_vals,
            "q1":           q1_vals,
            "q3":           q3_vals,
            "cn":           cn_vals,
            "cn_q1":        cn_q1v,
            "cn_q3":        cn_q3v,
            "has_amp":      set(samples_f) & gene_samples,
            "region":       pat_region,
        })

    if not patient_data:
        print(f"  [gene trajectories] No data loaded for any patient.")
        return 0

    # ── Colour palette — one colour per patient ───────────────────────────
    n_patients = len(patient_data)
    _PAT_CMAP  = plt.colormaps.get_cmap("tab20" if n_patients > 10 else "tab10")
    pat_colors = {d["patient"]: _PAT_CMAP(i % _PAT_CMAP.N)
                  for i, d in enumerate(patient_data)}

    entity_label   = f" — {entity_filter}" if entity_filter else ""
    region_str     = (f"{gene_region['chrom']}:{gene_region['start']//1_000_000:.2f}"
                      f"–{gene_region['end']//1_000_000:.2f} Mb")
    n_amp_patients = sum(1 for d in patient_data if d["has_amp"])
    print(f"  [gene trajectories] {n_amp_patients} patients with '{gene}' amp "
          f"in qualifying samples (of {len(patient_data)} with longitudinal data)")
    title = (f"{gene} focal amplification trajectories{entity_label}\n"
             f"Cohort region: {region_str}   |   "
             f"{n_amp_patients} patients with {gene} amp   |   "
             f"each patient uses their own focal amp coordinates   |   "
             f"error bars = IQR of 50 kbp bins")

    output_dir.mkdir(parents=True, exist_ok=True)

    def _draw(ax, key_mean, key_q1, key_q3):
        for d in patient_data:
            # Only include patients who have at least one amp-positive timepoint
            # in the filtered (post-reads/scale) sample list
            if not d["has_amp"]:
                continue
            x    = d["timepoints"]
            yv   = d[key_mean]
            lo   = [np.clip(m - q, 0, None) if not np.isnan(m) and not np.isnan(q)
                    else 0 for m, q in zip(yv, d[key_q1])]
            hi   = [np.clip(q - m, 0, None) if not np.isnan(m) and not np.isnan(q)
                    else 0 for m, q in zip(yv, d[key_q3])]
            col  = pat_colors[d["patient"]]
            for i, (xi, yi) in enumerate(zip(x, yv)):
                if np.isnan(yi):
                    continue
                in_amp = d["samples"][i] in d["has_amp"]
                ax.errorbar(
                    [xi], [yi],
                    yerr=[[lo[i]], [hi[i]]],
                    fmt="o",
                    markersize=7 if in_amp else 5,
                    color=col,
                    markeredgecolor="#333333" if in_amp else col,
                    markeredgewidth=0.8 if in_amp else 0,
                    ecolor=col, elinewidth=0.8, capsize=2,
                    linewidth=0, zorder=4,
                )
            valid = [(xi, yi) for xi, yi in zip(x, yv) if not np.isnan(yi)]
            if len(valid) > 1:
                ax.plot([v[0] for v in valid], [v[1] for v in valid],
                        color=col, linewidth=1.4, alpha=0.8,
                        zorder=3, label=d["patient"])
            elif len(valid) == 1:
                ax.plot(valid[0][0], valid[0][1], "o",
                        color=col, markersize=7, zorder=4,
                        label=d["patient"])

    def _add_outside_legend(fig, ax):
        """Deduplicate handles and place legend to the right of the axes."""
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        ncol = max(1, len(l2) // 30)
        leg = ax.legend(
            h2, l2,
            fontsize=6.5,
            ncol=ncol,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            title=f"{gene}-amplified patients",
            title_fontsize=7.0,
        )
        # Expand figure width to accommodate legend
        fig.subplots_adjust(right=max(0.60, 1.0 - 0.15 * ncol))
        return leg

    # ── Log2 plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.axhline(0,    color="#999999", linewidth=0.8, zorder=1)
    ax.axhline(0.58, color="#dddddd", linewidth=0.5, linestyle=":", zorder=1)
    ax.axhline(1.0,  color="#dddddd", linewidth=0.5, linestyle=":", zorder=1)
    _draw(ax, "log2", "q1", "q3")
    ax.set_ylabel("TF-corrected log2 ratio", fontsize=9, color="#444444")
    ax.set_xlabel("Relative timepoint (1 = first qualifying sample per patient)",
                  fontsize=9, color="#444444")
    ax.set_title(title, fontsize=9, fontweight="bold", color="#222222", loc="left")
    _add_outside_legend(fig, ax)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    _plots_save(fig, output_dir / f"{gene}_trajectories_log2")

    # ── CN plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.axhline(2, color="#999999", linewidth=0.8, zorder=1)
    for ref in (1, 3, 4, 6, 8):
        ax.axhline(ref, color="#dddddd", linewidth=0.5, linestyle=":", zorder=1)
    _draw(ax, "cn", "cn_q1", "cn_q3")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Relative copy number\n"
                  r"(2 $\times$ 2$^{\mathrm{log2ratio}}$)", fontsize=9, color="#444444")
    ax.set_xlabel("Relative timepoint (1 = first qualifying sample per patient)",
                  fontsize=9, color="#444444")
    ax.set_title(title, fontsize=9, fontweight="bold", color="#222222", loc="left")
    _add_outside_legend(fig, ax)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    _plots_save(fig, output_dir / f"{gene}_trajectories_cn")

    # ── Metadata TSV ──────────────────────────────────────────────────────
    rows = []
    for d in patient_data:
        for i, (samp, tp, orig_tp, tf, m, l2v, q1v, q3v, cnv) in enumerate(zip(
                d["samples"], d["timepoints"], d["orig_tps"],
                d["tfs"], d["scales"],
                d["log2"], d["q1"], d["q3"], d["cn"])):
            rows.append({
                "gene": gene, "patient": d["patient"],
                "sample": samp,
                "timepoint_norm": tp,
                "timepoint_orig": orig_tp,
                "tumor_fraction": round(tf, 4), "scale_factor": round(m, 4),
                "has_amp": samp in d["has_amp"],
                "region_chrom": d["region"]["chrom"],
                "region_start": d["region"]["start"],
                "region_end":   d["region"]["end"],
                "log2r": round(l2v, 4) if not np.isnan(l2v) else None,
                "log2r_q1": round(q1v, 4) if not np.isnan(q1v) else None,
                "log2r_q3": round(q3v, 4) if not np.isnan(q3v) else None,
                "cn": round(cnv, 4) if not np.isnan(cnv) else None,
            })
    meta_tsv = output_dir / f"{gene}_trajectories_metadata.tsv"
    pd.DataFrame(rows).to_csv(meta_tsv, sep="\t", index=False)
    print(f"  [gene trajectories] Written: {meta_tsv}")
    print(f"  [gene trajectories] Plots: {gene}_trajectories_log2/cn.pdf/png")

    return n_patients


# ---------------------------------------------------------------------------
# Tumor fraction estimation from empirical scaling
# ---------------------------------------------------------------------------

def compute_scaled_tf_estimates(
        tf_df:          pd.DataFrame,
        seg_files:      dict[str, Path],
        seg_files_segs: dict[str, Path] | None,
        output_dir:     Path,
        min_tf:         float = 0.03,
        min_abs:        float = 0.05,
) -> pd.DataFrame:
    """
    For every patient with >= 2 timepoints, estimate the tumor fraction of
    each non-reference timepoint from the empirical scale factor relative to
    the patient's highest-TF timepoint.

    The model: if the reference (highest-TF) timepoint has ichorCNA TF = TF_ref,
    and the empirical scale factor for a later timepoint is m (meaning that
    timepoint's log2R values are m× smaller), then:

        TF_estimated = TF_ref / m

    Uncertainty: the RMSE of the WLS fit provides a residual error. We
    propagate this via a simple first-order approximation:

        TF_estimated_lower = TF_ref / (m + delta_m)
        TF_estimated_upper = TF_ref / (m - delta_m)

    where delta_m = rmse / sqrt(n_informative_bins) is the standard error
    on m estimated from the WLS residuals.

    Outputs
    -------
    <output_dir>/tf_estimates_from_scaling.tsv
        Columns: patient, entity, sample, timepoint, tf_ichorCNA,
                 tf_estimated, tf_lower, tf_upper, scale_factor, scale_rmse,
                 is_reference, n_bins_used

    <output_dir>/tf_estimates_scatter.pdf/.png
        Scatter: ichorCNA TF (x) vs estimated TF (y) coloured by entity,
        with identity line y=x.
    """
    from .plots import _save as _plots_save, ICHORCNA_ENTITY_PALETTE, entity_from_sample

    output_dir.mkdir(parents=True, exist_ok=True)

    df = tf_df.dropna(subset=["timepoint", "tumor_fraction"]).copy()
    df = df[df["tumor_fraction"] >= min_tf]
    df["timepoint"] = df["timepoint"].astype(int)

    rows: list[dict] = []

    for patient, pat_df in df.groupby("patient"):
        pat_df = pat_df[pat_df["sample"].isin(seg_files)].sort_values("timepoint")
        if len(pat_df) < 2:
            continue

        entity   = entity_from_sample(str(patient))
        samples    = pat_df["sample"].tolist()
        timepoints = pat_df["timepoint"].tolist()
        tfs        = pat_df["tumor_fraction"].tolist()

        # Reference = highest TF timepoint
        ref_idx  = tfs.index(max(tfs))
        ref_samp = samples[ref_idx]
        ref_tf   = tfs[ref_idx]

        # Load reference segment means
        ref_path      = seg_files[ref_samp]
        ref_segs_path = (seg_files_segs or {}).get(ref_samp)
        ref_segmeans  = None
        if ref_segs_path is not None:
            ref_segmeans = _load_bins_with_seg_means(
                ref_path, ref_segs_path, ref_samp, tf_from=ref_tf
            )

        for i, (samp, tp, tf) in enumerate(zip(samples, timepoints, tfs)):
            is_ref = (i == ref_idx)
            row = {
                "patient":      str(patient),
                "entity":       entity,
                "sample":       samp,
                "timepoint":    tp,
                "tf_ichorCNA":  round(tf, 4),
                "is_reference": is_ref,
                "scale_factor": 1.0 if is_ref else None,
                "scale_rmse":   0.0  if is_ref else None,
                "n_bins_used":  None,
                "tf_estimated": round(ref_tf, 4),
                "tf_lower":     round(ref_tf, 4),
                "tf_upper":     round(ref_tf, 4),
            }

            if not is_ref and ref_segmeans is not None:
                seg_path  = seg_files.get(samp)
                segs_path = (seg_files_segs or {}).get(samp)
                if seg_path is not None and segs_path is not None:
                    samp_segmeans = _load_bins_with_seg_means(
                        seg_path, segs_path, samp, tf_from=tf
                    )
                    if samp_segmeans is not None and not samp_segmeans.empty:
                        aligned = _align_bins(ref_segmeans, samp_segmeans)
                        if not aligned.empty:
                            lf   = aligned["logr_a"].values
                            la   = aligned["logr_b"].values
                            m, rmse = _optimise_scaling(lf, la, min_abs=min_abs)

                            # n informative bins used in the fit
                            n_bins = int(np.sum(
                                np.isfinite(lf) & np.isfinite(la) &
                                (np.abs(lf) >= min_abs)
                            ))

                            # TF estimate: TF_ref / m
                            tf_est = ref_tf / m if m > 0 else float("nan")

                            # Uncertainty: propagate RMSE as standard error on m
                            if n_bins > 0 and not np.isnan(rmse):
                                delta_m = rmse / np.sqrt(n_bins)
                                m_lo = max(1.0, m - delta_m)
                                m_hi = m + delta_m
                                tf_lo = ref_tf / m_hi   # lower TF when m is larger
                                tf_hi = ref_tf / m_lo   # upper TF when m is smaller
                            else:
                                tf_lo = tf_hi = tf_est

                            row.update({
                                "scale_factor": round(m, 4),
                                "scale_rmse":   round(rmse, 4),
                                "n_bins_used":  n_bins,
                                "tf_estimated": round(tf_est, 4),
                                "tf_lower":     round(max(0, tf_lo), 4),
                                "tf_upper":     round(min(1, tf_hi), 4),
                            })
            rows.append(row)

    if not rows:
        print("  [tf estimates] No data — no patients with >= 2 timepoints.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    tsv_path = output_dir / "tf_estimates_from_scaling.tsv"
    result.to_csv(tsv_path, sep="\t", index=False)
    print(f"  [tf estimates] Written: {tsv_path}  ({len(result)} rows, "
          f"{result['patient'].nunique()} patients)")

    # ── Scatter: ichorCNA TF vs estimated TF ─────────────────────────────
    non_ref = result[~result["is_reference"]].dropna(subset=["tf_estimated"])
    if non_ref.empty:
        return result

    fig, ax = plt.subplots(figsize=(6.5, 6.0), facecolor="white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)

    # Identity line
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.9,
            linestyle="--", zorder=1, label="y = x (perfect agreement)")

    for entity, grp in non_ref.groupby("entity"):
        pal   = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
        color = pal["gain"]
        # Error bars from tf_lower / tf_upper
        yerr_lo = (grp["tf_estimated"] - grp["tf_lower"]).clip(lower=0).values
        yerr_hi = (grp["tf_upper"] - grp["tf_estimated"]).clip(lower=0).values
        ax.errorbar(
            grp["tf_ichorCNA"].values,
            grp["tf_estimated"].values,
            yerr=[yerr_lo, yerr_hi],
            fmt="o", markersize=4, color=color,
            ecolor=color, elinewidth=0.7, capsize=2,
            alpha=0.7, zorder=3, label=entity,
        )

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Tumor fraction (ichorCNA, per-sample)", fontsize=9,
                  color="#444444")
    ax.set_ylabel("Tumor fraction estimated from\nempirical scale factor",
                  fontsize=9, color="#444444")
    ax.set_title("TF comparison: ichorCNA vs scaling-based estimate\n"
                 "(non-reference timepoints only; reference = highest TF per patient)",
                 fontsize=9, fontweight="bold", color="#222222", loc="left")

    # Pearson r annotation (non-reference only)
    valid = non_ref.dropna(subset=["tf_ichorCNA", "tf_estimated"])
    if len(valid) > 5:
        r = float(np.corrcoef(valid["tf_ichorCNA"].values,
                               valid["tf_estimated"].values)[0, 1])
        ax.text(0.97, 0.04,
                f"Pearson r = {r:.3f}  (n = {len(valid)})",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                color="#444444")

    ax.legend(fontsize=7.5, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    _plots_save(fig, output_dir / "tf_estimates_scatter")
    print(f"  [tf estimates] Scatter plot written.")

    return result