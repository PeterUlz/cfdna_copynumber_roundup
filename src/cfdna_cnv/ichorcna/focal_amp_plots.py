"""
Per-sample diagnostic plots for focal amplification detection.

For every sample that had any candidate segments (CN >= 4, size 50-500 kbp,
num.mark >= 3) — including samples that were ultimately discarded for having
too many events — this module generates a genome-wide scatter plot showing:

  - All 50 kbp bin-level logR values (grey dots)
  - Candidate segments coloured by filter outcome:
      green  = PASSED all filters
      orange = FAILED the log2 elevation / background filter
      red    = sample DISCARDED (too many focal amps)
      grey   = no background data (filter not evaluable)
  - Background logR level shown as a horizontal bar per candidate
  - Segment annotations: CN, num.mark, delta_log2
  - Filter threshold lines

Output: one PDF/PNG per sample in <output_dir>/focal_amp_diagnostics/
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
from .focal_amp import (
    detect_focal_amplifications,
    _get_log2_col,
    _is_amp_call,
)
from ..common.constants import FOCAL_MIN_BP, FOCAL_MAX_BP


# ---------------------------------------------------------------------------
# Diagnostic detection: return all candidates with per-filter outcome columns
# ---------------------------------------------------------------------------

def detect_focal_amplifications_diagnostic(
        seg_df:       pd.DataFrame,
        segs_1mb:     pd.DataFrame | None = None,
        min_bp:       int   = FOCAL_MIN_BP,
        max_bp:       int   = FOCAL_MAX_BP,
        min_cn:       int   = 4,
        min_bins:     int   = 3,
        log2_delta:   float = 0.2,
        flank_bins:   int   = 2,
        noise_factor: float = 2.0,
        max_focal_per_sample: int = 5,
) -> pd.DataFrame:
    """
    Like detect_focal_amplifications but returns ALL candidates that passed
    the size/CN/bin filters, annotated with per-row filter outcome columns:

        passed_size     : bool  — segment length in [min_bp, max_bp]
        passed_cn_call  : bool  — CN >= min_cn AND call is AMP/HLAMP
        passed_bins     : bool  — num.mark >= min_bins
        passed_background: bool | None  — logR delta >= log2_delta
                           (None = background not evaluable)
        bg_log2         : float — median background logR
        delta_log2      : float — segment logR - background logR
        filter_outcome  : str   — "PASS" | "FAIL_BACKGROUND" |
                                   "NO_BACKGROUND" | "DISCARDED_TOO_MANY"

    Segments that fail the size/CN/bin filters are NOT included — those are
    truly not candidates. Everything that reaches the background filter is
    shown with its outcome.
    """
    df = seg_df.copy()
    has_call = "call" in df.columns
    has_cn   = "copy.number" in df.columns

    def _passes_cn_call(row) -> bool:
        if has_call and pd.notna(row["call"]):
            if not _is_amp_call(row["call"]):
                return False
        if has_cn and pd.notna(row["copy.number"]):
            return float(row["copy.number"]) >= min_cn
        logr_col = _get_log2_col(df)
        if logr_col and pd.notna(row.get(logr_col)):
            return float(row[logr_col]) >= 1.0
        return False

    # Size + CN/call
    candidates = df[
        (df["seg_length"] >= min_bp) &
        (df["seg_length"] <= max_bp)
    ].copy()
    candidates = candidates[candidates.apply(_passes_cn_call, axis=1)]

    if candidates.empty:
        return pd.DataFrame()

    # Bin count
    if "num.mark" in candidates.columns:
        candidates = candidates[
            pd.to_numeric(candidates["num.mark"], errors="coerce") >= min_bins
        ].copy()
    else:
        candidates = candidates[
            (candidates["seg_length"] / 50_000) >= min_bins
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Background filter — annotate each candidate
    # Uses 50 kbp bins (same file as candidate detection) with
    # length-weighted median of flanking bins within flank_bins * 1 Mbp.
    log2_col_50k = _get_log2_col(df)

    bg_out    = []
    delta_out = []
    passed_bg = []

    if log2_col_50k is None:
        bg_out    = [np.nan] * len(candidates)
        delta_out = [np.nan] * len(candidates)
        passed_bg = [None]   * len(candidates)
    else:
        flank_bp = flank_bins * 1_000_000

        # Compute sample noise for the noise_factor check
        _noise_diffs = []
        for _chrom, _grp in df.groupby("chrom"):
            _vals = pd.to_numeric(
                _grp[log2_col_50k], errors="coerce"
            ).dropna().values
            if len(_vals) >= 2:
                _noise_diffs.append(np.abs(np.diff(_vals)))
        sample_noise: float | None = (
            float(np.median(np.concatenate(_noise_diffs)))
            if _noise_diffs else None
        )

        bins_by_chrom: dict[str, list] = {}
        for _, r in df.iterrows():
            lv = r.get(log2_col_50k, np.nan)
            if pd.isna(lv):
                continue
            bins_by_chrom.setdefault(r["chrom"], []).append(
                (int(r["start"]), int(r["end"]), float(lv))
            )
        for k in bins_by_chrom:
            bins_by_chrom[k].sort()

        def _weighted_median(vals, wts):
            pairs = sorted(zip(vals, wts), key=lambda x: x[0])
            total = sum(w for _, w in pairs)
            cumsum = 0.0
            for v, w in pairs:
                cumsum += w
                if cumsum >= total / 2:
                    return float(v)
            return float(pairs[-1][0])

        def _background(chrom, seg_start, seg_end):
            upstream, downstream = [], []
            for s, e, lv in bins_by_chrom.get(chrom, []):
                overlap = min(e, seg_end) - max(s, seg_start)
                if overlap > 0:
                    continue
                bin_len = e - s
                if e <= seg_start and seg_start - e <= flank_bp:
                    upstream.append((lv, bin_len))
                elif s >= seg_end and s - seg_end <= flank_bp:
                    downstream.append((lv, bin_len))
            combined = upstream + downstream
            if not combined:
                return None
            vals, wts = zip(*combined)
            return _weighted_median(list(vals), list(wts))

        for _, row in candidates.iterrows():
            seg_l2 = row.get(log2_col_50k, np.nan)
            if pd.isna(seg_l2):
                bg_out.append(np.nan)
                delta_out.append(np.nan)
                passed_bg.append(None)
                continue
            bg = _background(row["chrom"], int(row["start"]), int(row["end"]))
            if bg is None:
                bg_out.append(np.nan)
                delta_out.append(np.nan)
                # No background — apply noise filter only
                if sample_noise is not None and sample_noise > 0:
                    passes_noise = float(seg_l2) >= noise_factor * sample_noise
                else:
                    passes_noise = True
                passed_bg.append(None if passes_noise else False)
            else:
                delta = float(seg_l2) - bg
                bg_out.append(round(bg, 4))
                delta_out.append(round(delta, 4))
                passes = (float(seg_l2) >= log2_delta) and (delta >= log2_delta)
                if passes and sample_noise is not None and sample_noise > 0:
                    passes = float(seg_l2) >= noise_factor * sample_noise
                passed_bg.append(passes)

    candidates = candidates.copy()
    candidates["bg_log2"]    = bg_out
    candidates["delta_log2"] = delta_out
    candidates["passed_bg"]  = passed_bg

    # Determine per-sample final outcome
    n_passed = sum(
        1 for p in passed_bg
        if p is True or p is None  # None = no background = kept
    )
    discarded = n_passed > max_focal_per_sample

    def _outcome(row):
        if discarded:
            return "DISCARDED_TOO_MANY"
        if pd.isna(row["passed_bg"]):
            return "NO_BACKGROUND"
        return "PASS" if row["passed_bg"] else "FAIL_BACKGROUND"

    candidates["filter_outcome"] = candidates.apply(_outcome, axis=1)
    candidates["n_passed_total"] = n_passed
    candidates["was_discarded"]  = discarded

    return candidates.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-sample diagnostic plot
# ---------------------------------------------------------------------------

_OUTCOME_COLORS = {
    "PASS":               "#2a9d8f",   # teal green
    "FAIL_BACKGROUND":    "#f4a261",   # orange
    "NO_BACKGROUND":      "#adb5bd",   # grey
    "DISCARDED_TOO_MANY": "#e63946",   # red
}
_OUTCOME_LABELS = {
    "PASS":               "Passed all filters",
    "FAIL_BACKGROUND":    "Failed background elevation filter",
    "NO_BACKGROUND":      "No background bins (kept by default)",
    "DISCARDED_TOO_MANY": "Sample discarded (too many focal amps)",
}


def plot_focal_amp_diagnostic(
        sample:            str,
        seg_path_50k:      Path,
        seg_path_50k_segs: Path | None,
        tf:                float,
        candidates:        pd.DataFrame,
        output_path:       Path,
        entity:            str = "",
        min_bp:            int   = FOCAL_MIN_BP,
        max_bp:            int   = FOCAL_MAX_BP,
        min_cn:            int   = 4,
        log2_delta:        float = 0.2,
        logr_ylim:         tuple[float, float] = (-2.0, 3.0),
) -> None:
    """
    Generate a per-sample diagnostic plot showing:
      - 50 kbp bin-level logR scatter (light grey, from .cna.seg)
      - 50 kbp segment logR (step lines, blue, from .seg.txt)
      - Candidate focal amp segments coloured by filter outcome
      - Background level bar for each candidate
      - Delta and CN annotations per candidate
      - Filter threshold lines
    """
    # Load bin-level 50 kbp data for background scatter
    bins_50k  = parse_seg_file(seg_path_50k, sample) if seg_path_50k.exists() else None
    # Load segmented 50 kbp data for step lines
    segs_50k  = parse_seg_file(seg_path_50k_segs, sample) \
                if seg_path_50k_segs is not None and seg_path_50k_segs.exists() else None

    fig, ax = plt.subplots(figsize=(22, 5.5), facecolor="white")
    _style_ax(ax)

    logr_col_50k      = _get_log2_col(bins_50k)  if bins_50k  is not None else None
    logr_col_50k_segs = _get_log2_col(segs_50k)  if segs_50k  is not None else None

    # ── Grey scatter: 50 kbp bin-level logR ────────────────────────────────
    if bins_50k is not None and logr_col_50k is not None:
        bins_50k = bins_50k[bins_50k["chrom"].isin(_OFFSETS)].copy()
        x_bins = bins_50k.apply(
            lambda r: _genome_x(r["chrom"], (r["start"] + r["end"]) / 2),
            axis=1,
        ).values
        y_bins = np.clip(
            pd.to_numeric(bins_50k[logr_col_50k], errors="coerce").values,
            logr_ylim[0], logr_ylim[1],
        )
        ax.scatter(x_bins, y_bins, s=1.5, c="#cccccc", alpha=0.5,
                   linewidths=0, zorder=2, rasterized=True)

    # ── Blue step lines: 50 kbp segment logR (from .seg.txt) ──────────────
    # Shows the segmented CN landscape at the same 50 kbp resolution as
    # the focal amp candidates — identical segmentation, so candidate bars
    # and background segments are directly comparable.
    if segs_50k is not None and logr_col_50k_segs is not None:
        segs_50k_plot = segs_50k[segs_50k["chrom"].isin(_OFFSETS)].copy()
        for _, r in segs_50k_plot.iterrows():
            x0   = _genome_x(r["chrom"], int(r["start"]))
            x1   = _genome_x(r["chrom"], int(r["end"]))
            logr = pd.to_numeric(r[logr_col_50k_segs], errors="coerce")
            if pd.isna(logr):
                continue
            logr = float(np.clip(logr, logr_ylim[0], logr_ylim[1]))
            ax.hlines(logr, x0, x1, colors="#457b9d", linewidth=1.5,
                      alpha=0.7, zorder=3)

    # ── Candidate segments coloured by filter outcome ───────────────────────
    for _, row in candidates.iterrows():
        chrom = row["chrom"]
        if chrom not in _OFFSETS:
            continue
        x0     = _genome_x(chrom, int(row["start"]))
        x1     = _genome_x(chrom, int(row["end"]))
        outcome = row["filter_outcome"]
        color   = _OUTCOME_COLORS.get(outcome, "#888888")

        # Candidate bar height: segment logR from the 50 kbp .seg.txt file
        # (seg.median.logR — the median logR across all 50 kbp bins in the
        # segment, as reported by ichorCNA). This is what the filter uses.
        seg_logr_col = logr_col_50k or logr_col_50k_segs
        seg_logr = float(np.clip(
            pd.to_numeric(row.get(seg_logr_col, np.nan), errors="coerce"),
            logr_ylim[0], logr_ylim[1],
        )) if seg_logr_col else float(np.clip(
            np.log2(max(float(row.get("copy.number", 2)), 0.01) / 2),
            logr_ylim[0], logr_ylim[1],
        ))

        ax.hlines(seg_logr, x0, x1, colors=color, linewidth=4.0,
                  alpha=0.9, zorder=5)

        # Background level bar (dashed, same x extent)
        bg = row.get("bg_log2", np.nan)
        if pd.notna(bg):
            bg_clipped = float(np.clip(bg, logr_ylim[0], logr_ylim[1]))
            ax.hlines(bg_clipped, x0, x1, colors=color, linewidth=1.5,
                      linestyle="--", alpha=0.6, zorder=4)
            # Vertical arrow showing the delta
            ax.annotate(
                "", xy=(x0, seg_logr), xytext=(x0, bg_clipped),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                zorder=6,
            )

        # Annotation: CN and delta above the segment
        cn_val    = row.get("copy.number", "?")
        delta_val = row.get("delta_log2", np.nan)
        n_bins_val = int(pd.to_numeric(row.get("num.mark", 0), errors="coerce") or 0)
        delta_str = f"Δ{delta_val:+.2f}" if pd.notna(delta_val) else "Δ?"
        ann_y     = min(float(np.clip(seg_logr + 0.12, logr_ylim[0], logr_ylim[1])),
                        logr_ylim[1] - 0.15)
        ax.text(
            (x0 + x1) / 2, ann_y,
            f"CN{cn_val}  {delta_str}\n{n_bins_val}bins",
            fontsize=5.5, color=color, ha="center", va="bottom",
            fontweight="bold", zorder=7, clip_on=True,
        )

    # ── Threshold lines ─────────────────────────────────────────────────────
    # log2_delta threshold shown relative to 0 (approximate guide)
    ax.axhline(log2_delta,  color="#f4a261", linewidth=0.8,
               linestyle=":", alpha=0.8, zorder=1)
    ax.axhline(-log2_delta, color="#f4a261", linewidth=0.8,
               linestyle=":", alpha=0.8, zorder=1)
    ax.text(GENOME_SIZE * 1.001, log2_delta,
            f"+{log2_delta}", fontsize=6, color="#f4a261", va="center")
    ax.text(GENOME_SIZE * 1.001, -log2_delta,
            f"-{log2_delta}", fontsize=6, color="#f4a261", va="center")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_ylim(logr_ylim)
    ax.set_ylabel("log2 ratio", fontsize=9, color="#444444", labelpad=5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Title ────────────────────────────────────────────────────────────────
    n_pass     = (candidates["filter_outcome"] == "PASS").sum()
    n_fail_bg  = (candidates["filter_outcome"] == "FAIL_BACKGROUND").sum()
    n_no_bg    = (candidates["filter_outcome"] == "NO_BACKGROUND").sum()
    n_discard  = (candidates["filter_outcome"] == "DISCARDED_TOO_MANY").sum()
    discarded  = candidates["was_discarded"].any() if "was_discarded" in candidates else False

    status = "SAMPLE DISCARDED (too many events)" if discarded else "SAMPLE KEPT"
    tf_str = f"{tf:.1%}" if tf else "?"
    ax.set_title(
        f"{sample}  ({entity},  TF={tf_str})   {status}\n"
        f"Candidates: {len(candidates)}  |  "
        f"PASS={n_pass}  FAIL_BG={n_fail_bg}  "
        f"NO_BG={n_no_bg}  DISCARDED={n_discard}",
        fontsize=10, fontweight="bold", color="#222222",
        pad=6, loc="left",
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color="#cccccc",  label="50 kbp bin-level logR (raw signal)"),
        mpatches.Patch(color="#457b9d",  label="50 kbp segment logR (.seg.txt)"),
    ] + [
        mpatches.Patch(color=_OUTCOME_COLORS[k], label=_OUTCOME_LABELS[k])
        for k in _OUTCOME_COLORS
        if k in candidates["filter_outcome"].values
    ] + [
        plt.Line2D([0], [0], color="#888888", linewidth=1.5,
                   linestyle="--", label="Background level"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left",
              facecolor="white", edgecolor="#cccccc",
              labelcolor="#222222", framealpha=1.0, ncol=2)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Batch: one diagnostic plot per sample with any candidates
# ---------------------------------------------------------------------------

def plot_all_focal_amp_diagnostics(
        seg_files_50k:      dict[str, Path],
        high_tf_samples:    list[str],
        tf_df:              pd.DataFrame,
        output_dir:         Path,
        seg_files_50k_bins: dict[str, Path] | None = None,
        min_bp:             int   = FOCAL_MIN_BP,
        max_bp:             int   = FOCAL_MAX_BP,
        min_cn:             int   = 4,
        min_bins:           int   = 3,
        log2_delta:         float = 0.2,
        flank_bins:         int   = 2,
        noise_factor:       float = 2.0,
        max_focal_per_sample: int = 5,
) -> int:
    """
    For every high-TF sample that has at least one candidate segment,
    generate a diagnostic plot regardless of whether the sample was
    ultimately kept or discarded.

    Parameters
    ----------
    seg_files_50k       : segmented .seg.txt files — used for candidate detection.
    seg_files_50k_bins  : bin-level .cna.seg files — used for logR scatter
                          background in the plot. Falls back to seg_files_50k
                          if not provided.
    high_tf_samples     : samples that passed the TF threshold.
    tf_df               : tumor fraction table for TF lookup.
    output_dir          : plots written here.
    All other params mirror detect_focal_amplifications.

    Returns
    -------
    Number of plots generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    high_tf_set = set(high_tf_samples)
    tf_map      = tf_df.set_index("sample")["tumor_fraction"].to_dict()
    n_plots     = 0
    candidates_total = 0

    samples = [s for s in seg_files_50k if s in high_tf_set]
    n       = len(samples)

    for i, sample in enumerate(samples, 1):
        if i % 100 == 0 or i == n:
            print(f"        focal diag: {i}/{n} samples …", flush=True)

        seg_path_50k = seg_files_50k[sample]

        df_50k = parse_seg_file(seg_path_50k, sample)
        if df_50k is None or df_50k.empty:
            continue

        candidates = detect_focal_amplifications_diagnostic(
            df_50k,
            segs_1mb             = None,  # background now uses 50kb bins
            min_bp               = min_bp,
            max_bp               = max_bp,
            min_cn               = min_cn,
            min_bins             = min_bins,
            log2_delta           = log2_delta,
            flank_bins           = flank_bins,
            noise_factor         = noise_factor,
            max_focal_per_sample = max_focal_per_sample,
        )

        if candidates.empty:
            del df_50k
            gc.collect()
            continue

        candidates_total += len(candidates)
        entity      = entity_from_sample(sample)
        tf          = tf_map.get(sample, float("nan"))
        safe_name   = sample.replace("/", "_").replace(" ", "_")
        output_path = output_dir / safe_name

        # bin-level .cna.seg → grey scatter; segmented .seg.txt → blue steps
        _bin_path  = (seg_files_50k_bins or seg_files_50k).get(sample, seg_path_50k)
        _segs_path = seg_files_50k.get(sample)  # segmented, for step lines
        plot_focal_amp_diagnostic(
            sample            = sample,
            seg_path_50k      = _bin_path,
            seg_path_50k_segs = _segs_path,
            tf                = tf,
            candidates        = candidates,
            output_path       = output_path,
            entity            = entity,
            min_bp            = min_bp,
            max_bp            = max_bp,
            min_cn            = min_cn,
            log2_delta        = log2_delta,
        )
        n_plots += 1
        del df_50k, candidates
        gc.collect()

    print(f"        Diagnostic plots: {n_plots} samples, "
          f"{candidates_total} total candidates visualised.")
    return n_plots