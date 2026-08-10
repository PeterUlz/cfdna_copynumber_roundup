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
    df = pd.read_csv(tsv, sep=None, engine="python")
    required = {"sample", "expected_tf", "series"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dilution metadata TSV missing required columns: {missing}\n"
            f"Required: sample, expected_tf, series\n"
            f"Found columns: {list(df.columns)}"
        )
    df["expected_tf"] = df["expected_tf"].astype(float)
    if "replicate" not in df.columns:
        df["replicate"] = (
            df.groupby(["series", "expected_tf"]).cumcount() + 1
        )
    return df


def _find_file(directory: Path, sample: str, suffix: str) -> Path | None:
    """
    Find a file matching sample name + suffix in directory.
    Tries several strategies in order:
      1. Subdirectory: <directory>/<sample>/<sample><suffix>
      2. Subdirectory glob: <directory>/<sample>/*<suffix>
      3. Flat exact: <directory>/<sample><suffix>
      4. Flat glob: <directory>/*<sample>*<suffix>
      5. Normalised (hyphens ↔ underscores) variants of the above
    """
    candidates = [sample, sample.replace("-", "_"), sample.replace("_", "-")]

    for samp in candidates:
        # 1. Subdir exact
        sub = directory / samp / f"{samp}{suffix}"
        if sub.exists():
            return sub
        # 2. Subdir glob (handles cases where file name differs from dir name)
        subdir = directory / samp
        if subdir.is_dir():
            matches = sorted(subdir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        # 3. Flat exact
        flat = directory / f"{samp}{suffix}"
        if flat.exists():
            return flat
        # 4. Flat glob
        matches = sorted(directory.glob(f"*{samp}*{suffix}"))
        if matches:
            return matches[0]

    print(f"    [NOT FOUND] {sample}{suffix}  "
          f"(searched subdirs and flat in {directory})")
    return None


def compute_ichorCNA_tf(
        meta_df:  pd.DataFrame,
        dir_1mb:  Path,
) -> dict[str, float]:
    """Extract TF from ichorCNA .params.txt files (handles subdir structure)."""
    import re
    tf_map: dict[str, float] = {}
    n_warned = 0
    for sample in meta_df["sample"]:
        p = _find_file(dir_1mb, sample, ".params.txt")
        if p is None:
            continue
        try:
            txt = p.read_text()
            # Try both formats:
            # "Tumor Fraction:\t0.07773"  (ichorCNA .params.txt)
            # "tumor_fraction = 0.07773"  (older ichorCNA format)
            m = (re.search(r"Tumor Fraction:\s+([0-9.]+)", txt) or
                 re.search(r"tumor_fraction\s*=\s*([0-9.]+)", txt))
            if m:
                tf_map[sample] = float(m.group(1))
            else:
                if n_warned < 3:
                    lines = txt.strip().splitlines()[:8]
                    print(f"    [WARN] No tumor_fraction found in {p.name}")
                    print(f"           File contents: {lines}")
                    n_warned += 1
                elif n_warned == 3:
                    print(f"    [WARN] (further missing tumor_fraction warnings suppressed)")
                    n_warned += 1
        except Exception as e:
            print(f"    [WARN] Could not read {p}: {e}")
    n_found = len(tf_map)
    n_total = len(meta_df["sample"].unique())
    print(f"  ichorCNA TF found for {n_found}/{n_total} samples")
    if n_found == 0:
        # Help diagnose — show what params files exist
        found = list(dir_1mb.rglob("*.params.txt"))[:5]
        print(f"  [WARN] First .params.txt files found under {dir_1mb}:")
        for f in found:
            print(f"    {f.relative_to(dir_1mb)}")
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
    print(f"  Looking for .params.txt files (e.g. {dir_1mb}/<sample>/<sample>.params.txt)")
    ichorcna_tf = compute_ichorCNA_tf(meta_df, dir_1mb)
    meta_df["tf_ichorCNA"] = meta_df["sample"].map(ichorcna_tf)
    n_found = meta_df["tf_ichorCNA"].notna().sum()
    print(f"  Found ichorCNA TF for {n_found}/{n_samples} samples")

    # ── Scaling-based TFs ─────────────────────────────────────────────────
    print(f"  Computing scaling-based TFs from {dir_50kb} …")
    scaling_df = compute_scaling_tf(meta_df, dir_50kb, min_abs=min_abs)
    if scaling_df.empty:
        print(f"  [WARN] No scaling TFs computed — check that sample names "
              f"in the metadata match the file names in {dir_50kb}")
        print(f"  [WARN] Metadata sample names:")
        for s in meta_df["sample"].tolist():
            print(f"    {s}")
        actual = sorted(dir_50kb.glob("*.cna.seg"))[:10]
        print(f"  [WARN] First {len(actual)} .cna.seg files in {dir_50kb}:")
        for f in actual:
            print(f"    {f.name}")
        # Create empty scaling_df with required columns so merge doesn't crash
        scaling_df = pd.DataFrame(columns=[
            "sample", "tf_scaling", "tf_scaling_lower", "tf_scaling_upper",
            "scale_factor", "scale_rmse", "n_bins_used", "is_reference",
        ])

    # ── Merge ─────────────────────────────────────────────────────────────
    result = meta_df.merge(
        scaling_df[[
            "sample", "tf_scaling", "tf_scaling_lower", "tf_scaling_upper",
            "scale_factor", "scale_rmse", "n_bins_used", "is_reference",
        ]],
        on="sample", how="left",
    )

    # Mark reference samples: those where sample name == series name
    # (these are the undiluted anchors and should be excluded from evaluation)
    result["is_reference"] = result["sample"] == result["series"]

    # ── Write full TSV (including reference) ──────────────────────────────
    out_tsv = output_dir / "tf_validation.tsv"
    result.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Written: {out_tsv}")

    # ── Compute deviation statistics (non-reference only) ─────────────────
    dil = result[~result["is_reference"]].copy()
    dil["expected_tf"]  = pd.to_numeric(dil["expected_tf"],  errors="coerce")
    dil["tf_ichorCNA"]  = pd.to_numeric(dil["tf_ichorCNA"],  errors="coerce")
    dil["tf_scaling"]   = pd.to_numeric(dil["tf_scaling"],   errors="coerce")

    dil["dev_ichorCNA"] = dil["tf_ichorCNA"] - dil["expected_tf"]
    dil["dev_scaling"]  = dil["tf_scaling"]  - dil["expected_tf"]

    stats: dict[str, dict] = {}
    for method, col in [("ichorCNA", "dev_ichorCNA"), ("scaling", "dev_scaling")]:
        vals = dil[col].dropna()
        if len(vals) == 0:
            continue
        stats[method] = {
            "n":          len(vals),
            "mean_dev":   round(float(vals.mean()), 4),
            "median_dev": round(float(vals.median()), 4),
            "mae":        round(float(vals.abs().mean()), 4),
            "rmse":       round(float(np.sqrt((vals**2).mean())), 4),
            "std":        round(float(vals.std()), 4),
        }

    print(f"\n  Deviation from expected TF (non-reference dilution samples only):")
    print(f"  {'Method':<12} {'N':>5} {'Mean dev':>10} {'Median dev':>12} "
          f"{'MAE':>8} {'RMSE':>8} {'SD':>8}")
    print(f"  {'-'*65}")
    for method, s in stats.items():
        print(f"  {method:<12} {s['n']:>5} {s['mean_dev']:>+10.4f} "
              f"{s['median_dev']:>+12.4f} {s['mae']:>8.4f} "
              f"{s['rmse']:>8.4f} {s['std']:>8.4f}")

    # Write deviation stats TSV
    stats_rows = [{"method": k, **v} for k, v in stats.items()]
    pd.DataFrame(stats_rows).to_csv(
        output_dir / "tf_validation_stats.tsv", sep="\t", index=False
    )
    print(f"\n  Written: {output_dir / 'tf_validation_stats.tsv'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_tf_validation(dil, output_dir)
    _plot_tf_deviation_histogram(dil, output_dir)
    _plot_dilution_ratios(dil, output_dir, full_result=result)

    return result


def _plot_tf_validation(result: pd.DataFrame, output_dir: Path) -> None:
    """
    Three-panel scatter: expected TF (x) vs each estimate (y).
    Reference samples (sample == series) are excluded before this is called.
    Panel A: ichorCNA TF
    Panel B: Scaling-based TF (with error bars)
    Panel C: both overlaid
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
            x = grp["expected_tf"].values.astype(float) + rng.uniform(-0.005, 0.005, len(grp))
            y = pd.to_numeric(grp[col], errors="coerce").values.astype(float)
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
        sub = pd.DataFrame({
            "expected_tf": pd.to_numeric(result["expected_tf"], errors="coerce"),
            col:           pd.to_numeric(result[col],           errors="coerce"),
        }).dropna()
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
        x = grp["expected_tf"].values.astype(float) + rng.uniform(-0.005, 0.005, len(grp))
        for col, marker, label_sfx in [
            ("tf_ichorCNA", "o", " ichorCNA"),
            ("tf_scaling",  "^", " scaling"),
        ]:
            y     = pd.to_numeric(grp[col], errors="coerce").values.astype(float)
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


def _plot_tf_deviation_histogram(dil: pd.DataFrame, output_dir: Path) -> None:
    """
    Histogram of per-sample deviations (measured - expected) for ichorCNA
    and scaling-based TF estimates, excluding reference samples.
    One panel per method + one overlay panel.
    """
    dil = dil.copy()
    dil["dev_ichorCNA"] = (pd.to_numeric(dil["tf_ichorCNA"], errors="coerce")
                           - pd.to_numeric(dil["expected_tf"], errors="coerce"))
    dil["dev_scaling"]  = (pd.to_numeric(dil["tf_scaling"],  errors="coerce")
                           - pd.to_numeric(dil["expected_tf"], errors="coerce"))

    dev_ichorcna = dil["dev_ichorCNA"].dropna().values
    dev_scaling  = dil["dev_scaling"].dropna().values

    if len(dev_ichorcna) == 0 and len(dev_scaling) == 0:
        print("  [WARN] No deviation data to plot.")
        return

    # Shared x range
    all_devs = np.concatenate([dev_ichorcna, dev_scaling])
    x_min    = float(np.nanmin(all_devs)) - 0.02
    x_max    = float(np.nanmax(all_devs)) + 0.02
    bins     = np.linspace(x_min, x_max, 30)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white",
                             constrained_layout=True)

    def _hist_panel(ax, devs, color, label, show_stats=True):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
        ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
        ax.hist(devs, bins=bins, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_xlabel("Deviation from expected TF\n(measured − expected)",
                      fontsize=9, color="#444444")
        ax.set_ylabel("Count", fontsize=9, color="#444444")
        ax.set_title(label, fontsize=10, fontweight="bold",
                     color="#222222", loc="left")
        if show_stats and len(devs) > 0:
            mae  = float(np.abs(devs).mean())
            rmse = float(np.sqrt((devs**2).mean()))
            md   = float(np.median(devs))
            ax.text(0.97, 0.97,
                    f"MAE  = {mae:.4f}\nRMSE = {rmse:.4f}\nMedian = {md:+.4f}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.9))

    _hist_panel(axes[0], dev_ichorcna, "#d7191c",
                f"ichorCNA TF deviation  (n={len(dev_ichorcna)})")
    _hist_panel(axes[1], dev_scaling,  "#2c7bb6",
                f"Scaling-based TF deviation  (n={len(dev_scaling)})")

    # Overlay
    ax = axes[2]
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.hist(dev_ichorcna, bins=bins, color="#d7191c", alpha=0.55,
            edgecolor="white", linewidth=0.5, zorder=2, label="ichorCNA")
    ax.hist(dev_scaling,  bins=bins, color="#2c7bb6", alpha=0.55,
            edgecolor="white", linewidth=0.5, zorder=2, label="Scaling")
    ax.set_xlabel("Deviation from expected TF\n(measured − expected)",
                  fontsize=9, color="#444444")
    ax.set_ylabel("Count", fontsize=9, color="#444444")
    ax.set_title("Overlay", fontsize=10, fontweight="bold",
                 color="#222222", loc="left")
    ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.suptitle(
        "Per-sample TF deviation from expected dilution fraction\n"
        "(reference samples excluded)",
        fontsize=11, fontweight="bold", color="#222222",
    )
    _save(fig, output_dir / "tf_deviation_histogram")
    print(f"  Histogram written: tf_deviation_histogram.pdf/png")


def _plot_dilution_ratios(dil: pd.DataFrame, output_dir: Path,
                          full_result: pd.DataFrame | None = None) -> None:
    """
    Dilution-ratio plot: compares the expected dilution ratio against the
    ratio actually observed by each method — without depending on the
    absolute ichorCNA TF of the reference sample.

    For each non-reference sample:
        expected_ratio  = expected_tf / expected_tf_reference   (from metadata)
        ichorCNA_ratio  = tf_ichorCNA / tf_ichorCNA_reference   (per-series)
        scaling_ratio   = 1 / scale_factor                      (m = TF_ref/TF_dil)

    Three-panel scatter (expected ratio on x-axis):
        Panel A: ichorCNA ratio
        Panel B: scaling ratio (1/m) with uncertainty
        Panel C: overlay
    Plus a two-panel histogram of deviations from the expected ratio.
    """
    dil = dil.copy()
    dil["expected_tf"] = pd.to_numeric(dil["expected_tf"], errors="coerce")
    dil["tf_ichorCNA"] = pd.to_numeric(dil["tf_ichorCNA"], errors="coerce")
    dil["tf_scaling"]  = pd.to_numeric(dil["tf_scaling"],  errors="coerce")
    dil["scale_factor"]= pd.to_numeric(dil["scale_factor"],errors="coerce")

    # ── Reference values come from the FULL result (reference rows) ───────
    # The reference sample has sample == series and was excluded from `dil`.
    # Use full_result if available; otherwise fall back to max in dil (wrong
    # but better than crashing).
    if full_result is not None:
        ref_rows = full_result[full_result["sample"] == full_result["series"]].copy()
        ref_rows["expected_tf"]  = pd.to_numeric(ref_rows["expected_tf"],  errors="coerce")
        ref_rows["tf_ichorCNA"]  = pd.to_numeric(ref_rows["tf_ichorCNA"],  errors="coerce")
        ref_expected = ref_rows.set_index("series")["expected_tf"].rename("ref_expected_tf")
        ref_ichorcna = ref_rows.set_index("series")["tf_ichorCNA"].rename("ref_tf_ichorCNA")
    else:
        # Fallback — should not happen if called correctly
        ref_expected = (dil.groupby("series")["expected_tf"].max()
                          .rename("ref_expected_tf"))
        ref_ichorcna = (dil.groupby("series")["tf_ichorCNA"].max()
                          .rename("ref_tf_ichorCNA"))

    dil = dil.join(ref_expected, on="series")
    dil = dil.join(ref_ichorcna, on="series")

    dil["expected_ratio"] = dil["expected_tf"]  / dil["ref_expected_tf"]
    dil["ichorcna_ratio"] = dil["tf_ichorCNA"]  / dil["ref_tf_ichorCNA"]
    dil["scaling_ratio"]  = 1.0 / dil["scale_factor"]

    # Uncertainty on scaling ratio: propagate delta_m
    # delta(1/m) ≈ delta_m / m^2 = (rmse/sqrt(n)) / m^2
    if "scale_rmse" in dil.columns and "n_bins_used" in dil.columns:
        rmse  = pd.to_numeric(dil["scale_rmse"],  errors="coerce")
        n_bins= pd.to_numeric(dil["n_bins_used"], errors="coerce")
        m     = dil["scale_factor"]
        delta_m = rmse / np.sqrt(n_bins.clip(lower=1))
        dil["scaling_ratio_lo"] = 1.0 / (m + delta_m)
        dil["scaling_ratio_hi"] = 1.0 / (m - delta_m).clip(lower=1.0)
    else:
        dil["scaling_ratio_lo"] = dil["scaling_ratio"]
        dil["scaling_ratio_hi"] = dil["scaling_ratio"]

    dil["dev_ichorcna_ratio"] = dil["ichorcna_ratio"] - dil["expected_ratio"]
    dil["dev_scaling_ratio"]  = dil["scaling_ratio"]  - dil["expected_ratio"]

    # ── Print summary stats ───────────────────────────────────────────────
    print(f"\n  Dilution ratio deviation (non-reference samples only):")
    print(f"  {'Method':<12} {'N':>5} {'Mean dev':>10} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*47}")
    for method, col in [("ichorCNA", "dev_ichorcna_ratio"),
                        ("scaling",  "dev_scaling_ratio")]:
        vals = dil[col].dropna()
        if len(vals) == 0:
            continue
        print(f"  {method:<12} {len(vals):>5} {vals.mean():>+10.4f} "
              f"{vals.abs().mean():>8.4f} {np.sqrt((vals**2).mean()):>8.4f}")

    pd.DataFrame({
        "sample":                 dil["sample"],
        "series":                 dil["series"],
        "expected_ratio":         dil["expected_ratio"].round(4),
        "ichorcna_ratio":         dil["ichorcna_ratio"].round(4),
        "scaling_ratio":          dil["scaling_ratio"].round(4),
        "scaling_ratio_lo":       dil["scaling_ratio_lo"].round(4),
        "scaling_ratio_hi":       dil["scaling_ratio_hi"].round(4),
        "dev_ichorcna_ratio":     dil["dev_ichorcna_ratio"].round(4),
        "dev_scaling_ratio":      dil["dev_scaling_ratio"].round(4),
    }).to_csv(output_dir / "tf_dilution_ratios.tsv", sep="\t", index=False)
    print(f"  Written: tf_dilution_ratios.tsv")

    # ── Colour palette ────────────────────────────────────────────────────
    series_list  = sorted(dil["series"].dropna().unique())
    cmap         = plt.colormaps.get_cmap(
        "tab10" if len(series_list) <= 10 else "tab20")
    series_color = {s: cmap(i % cmap.N) for i, s in enumerate(series_list)}
    rng          = np.random.default_rng(42)

    # ── Scatter panels ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="white",
                             constrained_layout=True)

    def _scatter(ax, ratio_col, lo_col, hi_col, title):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8,
                linestyle="--", zorder=1, label="y = x")
        for series, grp in dil.groupby("series"):
            color = series_color[series]
            x  = grp["expected_ratio"].values.astype(float) + \
                 rng.uniform(-0.005, 0.005, len(grp))
            y  = pd.to_numeric(grp[ratio_col], errors="coerce").values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            if lo_col and hi_col:
                lo = pd.to_numeric(grp[lo_col], errors="coerce").values.astype(float)
                hi = pd.to_numeric(grp[hi_col], errors="coerce").values.astype(float)
                ax.errorbar(x[ok], y[ok],
                            yerr=[np.clip(y-lo, 0, None)[ok],
                                  np.clip(hi-y, 0, None)[ok]],
                            fmt="o", markersize=5, color=color,
                            ecolor=color, elinewidth=0.8, capsize=2,
                            alpha=0.8, zorder=3, label=series)
            else:
                ax.scatter(x[ok], y[ok], s=28, color=color,
                           alpha=0.8, zorder=3, label=series)
        sub = dil[["expected_ratio", ratio_col]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if len(sub) > 3:
            r = float(np.corrcoef(sub["expected_ratio"], sub[ratio_col])[0, 1])
            ax.text(0.97, 0.04, f"r = {r:.3f}  (n={len(sub)})",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="bottom", color="#444444")
        lim = max(1.05, float(dil[ratio_col].dropna().max()) * 1.1)
        ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim)
        ax.set_xlabel("Expected dilution ratio\n(expected_tf / ref_expected_tf)",
                      fontsize=9, color="#444444")
        ax.set_ylabel("Observed ratio", fontsize=9, color="#444444")
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="#222222", loc="left")
        ax.legend(fontsize=7, loc="upper left", frameon=True,
                  facecolor="white", edgecolor="#cccccc", title="Series")

    _scatter(axes[0], "ichorcna_ratio", None, None,
             "ichorCNA ratio\n(tf_dil / tf_ref, per series)")
    _scatter(axes[1], "scaling_ratio", "scaling_ratio_lo", "scaling_ratio_hi",
             "Scaling ratio\n(1 / scale_factor)")

    # Overlay
    ax = axes[2]
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8,
            linestyle="--", zorder=1)
    for series, grp in dil.groupby("series"):
        color = series_color[series]
        x = grp["expected_ratio"].values.astype(float) + \
            rng.uniform(-0.005, 0.005, len(grp))
        for col, marker in [("ichorcna_ratio", "o"), ("scaling_ratio", "^")]:
            y  = pd.to_numeric(grp[col], errors="coerce").values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                ax.scatter(x[ok], y[ok], s=22, c=[color]*ok.sum(),
                           marker=marker, alpha=0.75, zorder=3)
    lim = max(1.05, float(dil[["ichorcna_ratio", "scaling_ratio"]]
                          .apply(pd.to_numeric, errors="coerce")
                          .max().max()) * 1.1)
    ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim)
    ax.set_xlabel("Expected dilution ratio", fontsize=9, color="#444444")
    ax.set_ylabel("Observed ratio", fontsize=9, color="#444444")
    ax.set_title("Overlay: ○ ichorCNA  △ scaling", fontsize=10,
                 fontweight="bold", color="#222222", loc="left")
    import matplotlib.lines as mlines
    ax.legend(handles=[
        mlines.Line2D([], [], marker="o", color="grey", ls="none",
                      markersize=6, label="ichorCNA"),
        mlines.Line2D([], [], marker="^", color="grey", ls="none",
                      markersize=6, label="Scaling (1/m)"),
    ], fontsize=8, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.suptitle(
        "Dilution ratio accuracy — no dependence on absolute TF\n"
        "(reference samples excluded; ratio = diluted / reference)",
        fontsize=11, fontweight="bold", color="#222222",
    )
    _save(fig, output_dir / "tf_dilution_ratio_scatter")
    print(f"  Scatter written: tf_dilution_ratio_scatter.pdf/png")

    # ── Histogram of ratio deviations ─────────────────────────────────────
    dev_i = dil["dev_ichorcna_ratio"].dropna().values
    dev_s = dil["dev_scaling_ratio"].dropna().values
    all_d = np.concatenate([dev_i, dev_s])
    bins  = np.linspace(float(all_d.min())-0.02, float(all_d.max())+0.02, 30)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white",
                               constrained_layout=True)

    def _hist(ax, devs, color, label):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
        ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
        ax.hist(devs, bins=bins, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_xlabel("Ratio deviation (observed − expected)",
                      fontsize=9, color="#444444")
        ax.set_ylabel("Count", fontsize=9, color="#444444")
        ax.set_title(label, fontsize=10, fontweight="bold",
                     color="#222222", loc="left")
        if len(devs) > 0:
            ax.text(0.97, 0.97,
                    f"MAE  = {np.abs(devs).mean():.4f}\n"
                    f"RMSE = {np.sqrt((devs**2).mean()):.4f}\n"
                    f"Median = {np.median(devs):+.4f}",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.9))

    _hist(axes2[0], dev_i, "#d7191c",
          f"ichorCNA ratio deviation  (n={len(dev_i)})")
    _hist(axes2[1], dev_s, "#2c7bb6",
          f"Scaling ratio deviation  (n={len(dev_s)})")

    ax = axes2[2]
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.6)
    ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.hist(dev_i, bins=bins, color="#d7191c", alpha=0.55,
            edgecolor="white", linewidth=0.5, zorder=2, label="ichorCNA")
    ax.hist(dev_s, bins=bins, color="#2c7bb6", alpha=0.55,
            edgecolor="white", linewidth=0.5, zorder=2, label="Scaling")
    ax.set_xlabel("Ratio deviation", fontsize=9, color="#444444")
    ax.set_ylabel("Count", fontsize=9, color="#444444")
    ax.set_title("Overlay", fontsize=10, fontweight="bold",
                 color="#222222", loc="left")
    ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig2.suptitle(
        "Dilution ratio deviation histograms\n"
        "(reference samples excluded)",
        fontsize=11, fontweight="bold", color="#222222",
    )
    _save(fig2, output_dir / "tf_dilution_ratio_histogram")
    print(f"  Histogram written: tf_dilution_ratio_histogram.pdf/png")