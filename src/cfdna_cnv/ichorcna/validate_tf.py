"""
validate_tf.py  —  Tumor fraction validation against in-silico dilution series.

Compares:
  1. ichorCNA per-sample TF (from .params.txt files, 1 Mbp run)
  2. Our empirical scaling-based TF (TF_ref / scale_factor, from 50 kbp run)
  3. Expected TF (from user-supplied metadata TSV)

Usage via cfdna-cnv validate-tf (see cli.py for the subcommand).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .ichorcna.longitudinal_plots import (
    _load_bins_with_seg_means,
    _align_bins,
    _optimise_scaling,
)
from .ichorcna.plots import _save, ICHORCNA_ENTITY_PALETTE


def load_dilution_metadata(tsv: Path) -> pd.DataFrame:
    """
    Load and validate the dilution metadata TSV.

    Required columns:
        sample          : sample name matching ichorCNA file names
        expected_tf     : known dilution fraction (0–1), e.g. 0.5, 0.25
        series          : identifier for the original undiluted sample that
                          this dilution derives from (groups replicates at
                          the same dilution step together AND links dilutions
                          back to their source)

    Optional columns:
        replicate       : replicate label (e.g. 1–5); auto-assigned if absent
        label           : free-text label for the plot

    Example TSV:
        sample              expected_tf  series     replicate
        SAMPLE_A_dil50_r1   0.50         SERIES_A   1
        SAMPLE_A_dil50_r2   0.50         SERIES_A   2
        SAMPLE_A_dil25_r1   0.25         SERIES_A   1
        SAMPLE_A_undil_r1   1.00         SERIES_A   1   ← or actual known TF
    """
    df = pd.read_csv(tsv, sep="\t")
    required = {"sample", "expected_tf", "series"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dilution metadata TSV missing required columns: {missing}\n"
            f"Required: sample, expected_tf, series"
        )
    df["expected_tf"] = df["expected_tf"].astype(float)
    if "replicate" not in df.columns:
        df["replicate"] = (
            df.groupby(["series", "expected_tf"]).cumcount() + 1
        )
    return df


def _find_file(directory: Path, sample: str, suffix: str) -> Path | None:
    """Find a file matching sample name + suffix in directory."""
    # Try exact match
    exact = directory / f"{sample}{suffix}"
    if exact.exists():
        return exact
    # Try glob (handles extra suffixes like _S1_L001)
    matches = sorted(directory.glob(f"*{sample}*{suffix}"))
    return matches[0] if matches else None


def compute_ichorCNA_tf(
        meta_df:  pd.DataFrame,
        dir_1mb:  Path,
) -> dict[str, float]:
    """Extract TF from ichorCNA .params.txt files."""
    import re
    tf_map: dict[str, float] = {}
    for sample in meta_df["sample"]:
        p = _find_file(dir_1mb, sample, ".params.txt")
        if p is None:
            continue
        txt = p.read_text()
        m   = re.search(r"tumor_fraction\s*=\s*([0-9.]+)", txt)
        if m:
            tf_map[sample] = float(m.group(1))
    return tf_map


def compute_scaling_tf(
        meta_df:       pd.DataFrame,
        dir_50kb:      Path,
        min_abs:       float = 0.05,
) -> pd.DataFrame:
    """
    Estimate TF via empirical scaling for each dilution sample, using the
    highest expected-TF sample within each series as the reference.

    Returns DataFrame with columns:
        sample, series, tf_scaling, tf_scaling_lower, tf_scaling_upper,
        scale_factor, scale_rmse, n_bins_used
    """
    rows = []

    for series, grp in meta_df.groupby("series"):
        grp = grp.sort_values("expected_tf", ascending=False)

        # Reference = highest expected_tf sample in this series
        ref_row    = grp.iloc[0]
        ref_sample = ref_row["sample"]
        ref_tf     = float(ref_row["expected_tf"])

        ref_bins_path  = _find_file(dir_50kb, ref_sample, ".cna.seg")
        ref_segs_path  = _find_file(dir_50kb, ref_sample, ".seg.txt")
        if ref_bins_path is None or ref_segs_path is None:
            print(f"  [WARN] {series}: reference sample {ref_sample} "
                  f"files not found — skipping series")
            continue

        ref_segmeans = _load_bins_with_seg_means(
            ref_bins_path, ref_segs_path, ref_sample, tf_from=ref_tf
        )
        if ref_segmeans is None or ref_segmeans.empty:
            print(f"  [WARN] {series}: could not load ref segmeans — skipping")
            continue

        for _, row in grp.iterrows():
            samp    = row["sample"]
            exp_tf  = float(row["expected_tf"])
            is_ref  = (samp == ref_sample)

            out = {
                "sample": samp, "series": series,
                "is_reference": is_ref,
                "tf_scaling": ref_tf,
                "tf_scaling_lower": ref_tf,
                "tf_scaling_upper": ref_tf,
                "scale_factor": 1.0,
                "scale_rmse":   0.0,
                "n_bins_used":  None,
            }

            if not is_ref:
                bins_path = _find_file(dir_50kb, samp, ".cna.seg")
                segs_path = _find_file(dir_50kb, samp, ".seg.txt")
                if bins_path is None or segs_path is None:
                    rows.append(out); continue

                samp_segmeans = _load_bins_with_seg_means(
                    bins_path, segs_path, samp, tf_from=exp_tf
                )
                if samp_segmeans is None or samp_segmeans.empty:
                    rows.append(out); continue

                aligned = _align_bins(ref_segmeans, samp_segmeans)
                if aligned.empty:
                    rows.append(out); continue

                m, rmse = _optimise_scaling(
                    aligned["logr_a"].values,
                    aligned["logr_b"].values,
                    min_abs=min_abs,
                )
                n_bins = int(np.sum(
                    np.isfinite(aligned["logr_a"].values) &
                    np.isfinite(aligned["logr_b"].values) &
                    (np.abs(aligned["logr_a"].values) >= min_abs)
                ))
                tf_est = ref_tf / m if m > 0 else float("nan")
                if n_bins > 0 and not np.isnan(rmse):
                    delta_m = rmse / np.sqrt(n_bins)
                    tf_lo   = ref_tf / (m + delta_m)
                    tf_hi   = ref_tf / max(1.0, m - delta_m)
                else:
                    tf_lo = tf_hi = tf_est

                out.update({
                    "tf_scaling":       round(tf_est, 4),
                    "tf_scaling_lower": round(max(0, tf_lo), 4),
                    "tf_scaling_upper": round(min(1, tf_hi), 4),
                    "scale_factor":     round(m, 4),
                    "scale_rmse":       round(rmse, 4),
                    "n_bins_used":      n_bins,
                })

            rows.append(out)

    return pd.DataFrame(rows)


def run_tf_validation(
        meta_tsv:  Path,
        dir_1mb:   Path,
        dir_50kb:  Path,
        output_dir: Path,
        min_abs:   float = 0.05,
) -> pd.DataFrame:
    """
    Full validation pipeline.

    Parameters
    ----------
    meta_tsv   : dilution metadata TSV (sample, expected_tf, series, replicate)
    dir_1mb    : ichorCNA 1 Mbp run directory (for TF from params files)
    dir_50kb   : ichorCNA 50 kbp run directory (for scaling-based TF)
    output_dir : directory for outputs
    min_abs    : minimum |log2R| for bins used in WLS fit (default 0.05)

    Returns
    -------
    Combined results DataFrame (also written to tf_validation.tsv)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading metadata from {meta_tsv} …")
    meta_df = load_dilution_metadata(meta_tsv)
    n_samples = len(meta_df)
    n_series  = meta_df["series"].nunique()
    print(f"  {n_samples} samples in {n_series} dilution series")

    # ── ichorCNA TFs ──────────────────────────────────────────────────────
    print(f"  Reading ichorCNA TFs from {dir_1mb} …")
    ichorcna_tf = compute_ichorCNA_tf(meta_df, dir_1mb)
    meta_df["tf_ichorCNA"] = meta_df["sample"].map(ichorcna_tf)
    n_found = meta_df["tf_ichorCNA"].notna().sum()
    print(f"  Found ichorCNA TF for {n_found}/{n_samples} samples")

    # ── Scaling-based TFs ─────────────────────────────────────────────────
    print(f"  Computing scaling-based TFs from {dir_50kb} …")
    scaling_df = compute_scaling_tf(meta_df, dir_50kb, min_abs=min_abs)

    # ── Merge ─────────────────────────────────────────────────────────────
    result = meta_df.merge(
        scaling_df[[
            "sample", "tf_scaling", "tf_scaling_lower", "tf_scaling_upper",
            "scale_factor", "scale_rmse", "n_bins_used", "is_reference",
        ]],
        on="sample", how="left",
    )

    # ── Write TSV ─────────────────────────────────────────────────────────
    out_tsv = output_dir / "tf_validation.tsv"
    result.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Written: {out_tsv}")

    # ── Scatter plot ──────────────────────────────────────────────────────
    _plot_tf_validation(result, output_dir)

    return result


def _plot_tf_validation(result: pd.DataFrame, output_dir: Path) -> None:
    """
    Three-panel scatter: expected TF (x) vs each estimate (y).
    Panel A: ichorCNA TF
    Panel B: Scaling-based TF
    Panel C: both overlaid on a single axes

    Each series gets a distinct colour; replicates are shown as individual
    points at the same x-position with slight jitter.
    """
    # Series colour palette
    series_list = sorted(result["series"].unique())
    cmap        = plt.colormaps.get_cmap(
        "tab10" if len(series_list) <= 10 else "tab20"
    )
    series_color = {s: cmap(i % cmap.N) for i, s in enumerate(series_list)}

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.5), facecolor="white",
        constrained_layout=True,
    )

    panels = [
        ("tf_ichorCNA",  None,              None,              "ichorCNA TF (per-sample)"),
        ("tf_scaling",   "tf_scaling_lower","tf_scaling_upper","Scaling-based TF (TF_ref / m)"),
    ]

    # ── Panels A and B ────────────────────────────────────────────────────
    for ax, (col, lo_col, hi_col, title) in zip(axes[:2], panels):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8,
                linestyle="--", zorder=1)
        ax.axhline(0, color="#eeeeee", linewidth=0.5, zorder=0)

        for series, grp in result.groupby("series"):
            color = series_color[series]
            x = grp["expected_tf"].values + rng.uniform(-0.005, 0.005, len(grp))
            y = grp[col].values
            valid = np.isfinite(x) & np.isfinite(y)
            if not valid.any():
                continue
            if lo_col and hi_col:
                lo = grp[lo_col].values
                hi = grp[hi_col].values
                ye_lo = np.clip(y - lo, 0, None)
                ye_hi = np.clip(hi - y, 0, None)
                ax.errorbar(x[valid], y[valid],
                            yerr=[ye_lo[valid], ye_hi[valid]],
                            fmt="o", markersize=5, color=color,
                            ecolor=color, elinewidth=0.8, capsize=2,
                            alpha=0.8, zorder=3, label=series)
            else:
                ax.scatter(x[valid], y[valid], s=28, color=color,
                           alpha=0.8, zorder=3, label=series)

        # Stats
        sub = result[["expected_tf", col]].dropna()
        if len(sub) > 3:
            r  = float(np.corrcoef(sub["expected_tf"], sub[col])[0, 1])
            ax.text(0.97, 0.04, f"r = {r:.3f}  (n={len(sub)})",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="bottom", color="#444444")

        ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Expected TF (dilution)", fontsize=9, color="#444444")
        ax.set_ylabel("Measured TF", fontsize=9, color="#444444")
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="#222222", loc="left")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=7, loc="upper left", frameon=True,
                  facecolor="white", edgecolor="#cccccc", title="Series")

    # ── Panel C: overlay ──────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8,
            linestyle="--", zorder=1, label="y = x")

    for series, grp in result.groupby("series"):
        color = series_color[series]
        x = grp["expected_tf"].values + rng.uniform(-0.005, 0.005, len(grp))
        for col, marker, label_sfx in [
            ("tf_ichorCNA", "o", " ichorCNA"),
            ("tf_scaling",  "^", " scaling"),
        ]:
            y     = grp[col].values
            valid = np.isfinite(x) & np.isfinite(y)
            if not valid.any():
                continue
            ax.scatter(x[valid], y[valid], s=22, c=[color]*valid.sum(),
                       marker=marker, alpha=0.75, zorder=3,
                       label=f"{series}{label_sfx}")

    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Expected TF (dilution)", fontsize=9, color="#444444")
    ax.set_ylabel("Measured TF", fontsize=9, color="#444444")
    ax.set_title("Overlay: ○ ichorCNA  △ scaling", fontsize=10,
                 fontweight="bold", color="#222222", loc="left")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=6.5, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#cccccc",
              ncol=max(1, len(series_list) // 5))

    fig.suptitle("Tumor fraction validation — in-silico dilution series",
                 fontsize=12, fontweight="bold", color="#222222")
    _save(fig, output_dir / "tf_validation_scatter")
    print(f"  Plot written: tf_validation_scatter.pdf/png")