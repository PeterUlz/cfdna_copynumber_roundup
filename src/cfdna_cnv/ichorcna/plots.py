"""
Genome-wide copy-number frequency plots for ichorCNA data.

Two plot types:
  plot_broad_cna_frequencies  – multi-entity, multi-panel bar plot of AMP/DEL
                                 frequencies binned into fixed genomic windows
  plot_focal_amp_frequencies  – multi-entity lollipop plot of focal amp
                                 frequencies from focal_amp_frequencies.tsv

The broad CNA plot re-bins the variable-width segments from
broad_cna_frequencies.tsv into fixed windows (default 5 Mbp) to avoid
chromosome-spanning bars. For each window the frequency is the fraction of
samples whose segment overlaps that window, using the alteration call
(freq_amp / freq_del) from the stored per-locus counts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd

from ..common.constants import (
    CHROM_ORDER, CHROM_LENGTHS, GENOME_SIZE, _OFFSETS,
)

# ---------------------------------------------------------------------------
# Entity definitions
# ---------------------------------------------------------------------------

ICHORCNA_ENTITY_PALETTE: dict[str, dict] = {
    "Breast":     {"gain": "#e63946", "loss": "#457b9d"},
    "Colorectal": {"gain": "#2a9d8f", "loss": "#e9c46a"},
    "Prostate":   {"gain": "#f4a261", "loss": "#264653"},
    "Lung":       {"gain": "#a8dadc", "loss": "#1d3557"},
    "Ovarian":    {"gain": "#c77dff", "loss": "#5a189a"},
    "Other":      {"gain": "#adb5bd", "loss": "#495057"},
}

ENTITY_MAP: dict[str, str] = {
    "B": "Breast",
    "C": "Colorectal",
    "P": "Prostate",
    "L": "Lung",
    "E": "Ovarian",
}


def entity_from_sample(sample: str) -> str:
    return ENTITY_MAP.get(str(sample)[0].upper(), "Other")


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    for fmt in ("pdf", "png"):
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=180, facecolor="white",
                    bbox_inches="tight")
        print(f"      Plot saved: {out}")
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    for ci, chrom in enumerate(CHROM_ORDER):
        x0 = _OFFSETS[chrom]
        x1 = x0 + CHROM_LENGTHS[chrom]
        # Alternate white / very light grey for chromosome bands
        ax.axvspan(x0, x1,
                   color="#f5f5f5" if ci % 2 == 0 else "white",
                   alpha=1.0, zorder=0)
        ax.axvline(x0, color="#cccccc", linewidth=0.4,
                   linestyle="-", zorder=1)
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.set_xlim(0, GENOME_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.6)
    ax.set_facecolor("white")
    tick_x     = [_OFFSETS[c] + CHROM_LENGTHS[c] / 2 for c in CHROM_ORDER]
    tick_labels = [c.replace("chr", "") for c in CHROM_ORDER]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=7, color="#444444")
    ax.tick_params(axis="x", length=3, color="#cccccc")
    ax.tick_params(axis="y", length=2, color="#cccccc", colors="#444444")


def _genome_x(chrom: str, pos: float) -> float:
    return _OFFSETS.get(chrom, 0) + pos


# Broad CNA frequency plot
# ---------------------------------------------------------------------------

def plot_broad_cna_frequencies(
        broad_tsv:           Path,
        tf_tsv:              Path,
        output_path:         Path,
        bin_size:            int   = 5_000_000,
        min_freq:            float = 0.05,
        entities:            list[str] | None = None,
        exact_entity_counts: bool  = False,
) -> None:
    """
    Genome-wide broad CNA frequency plot from ichorCNA output.
    One panel per entity, gains upward, losses downward.

    Variable-width ichorCNA segments are re-binned into fixed *bin_size*
    windows before plotting to avoid chromosome-spanning bars.

    Parameters
    ----------
    broad_tsv           : path to broad_cna_frequencies.tsv
    tf_tsv              : path to tf_summary.tsv
    output_path         : stem path for output (.pdf and .png added)
    bin_size            : fixed bin width in bp for re-binning (default 5 Mbp)
    min_freq            : bins where max(freq_amp, freq_del) < min_freq are
                          drawn as transparent (still plotted for continuity)
    entities            : restrict to these entities (None = all)
    exact_entity_counts : when True, the broad_tsv already has per-entity rows
                          (entity column != "All") with exact freq values.
                          When False, global counts are scaled by entity fraction.
    """
    broad_df = pd.read_csv(broad_tsv, sep="\t", low_memory=False)
    tf_df    = pd.read_csv(tf_tsv,    sep="\t", low_memory=False)

    broad_df["chrom"] = broad_df["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )
    broad_df = broad_df[broad_df["chrom"].isin(_OFFSETS)]

    tf_df["entity"] = tf_df["patient"].apply(lambda p: entity_from_sample(str(p)))
    entity_n: dict[str, int] = tf_df.groupby("entity")["sample"].nunique().to_dict()

    all_entities = sorted(
        [e for e in entity_n if entity_n[e] > 0 and e != "All"],
        key=lambda e: list(ICHORCNA_ENTITY_PALETTE).index(e)
        if e in ICHORCNA_ENTITY_PALETTE else 99,
    )
    if entities:
        all_entities = [e for e in all_entities if e in entities]
    if not all_entities:
        print("      [WARN] No entities found in tf_summary — skipping broad CNA plot.")
        return

    n = len(all_entities)
    fig = plt.figure(figsize=(22, 4.2 * n + 1.5), facecolor="white")
    gs  = GridSpec(n, 1, figure=fig,
                   hspace=0.55, top=0.93, bottom=0.06, left=0.07, right=0.96)

    for ax_idx, entity in enumerate(all_entities):
        ax  = fig.add_subplot(gs[ax_idx])
        pal = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
        _style_ax(ax)

        n_entity   = entity_n.get(entity, 0)
        n_patients = tf_df[tf_df["entity"] == entity]["patient"].nunique()
        if n_entity == 0:
            continue

        # Select the right rows from broad_df for this entity.
        # Data is already binned (bin_start/bin_end columns) by
        # stream_broad_cna_frequencies, so we plot directly.
        if exact_entity_counts and "entity" in broad_df.columns:
            loci = broad_df[broad_df["entity"] == entity].copy()
        else:
            all_rows = broad_df[
                broad_df["entity"] == "All"
            ].copy() if "entity" in broad_df.columns else broad_df.copy()
            n_total_global = all_rows["n_total"].iloc[0] if len(all_rows) else 1
            scale = n_entity / n_total_global if n_total_global > 0 else 1.0
            loci  = all_rows.copy()
            loci["freq_amp"] = (loci["freq_amp"] * scale).clip(upper=1.0)
            loci["freq_del"] = (loci["freq_del"] * scale).clip(upper=1.0)

        for _, row in loci.iterrows():
            chrom = row["chrom"]
            x0    = _genome_x(chrom, int(row["bin_start"]))
            width = int(row["bin_end"]) - int(row["bin_start"])
            fa    = row["freq_amp"]
            fd    = row["freq_del"]
            if fa >= min_freq:
                ax.bar(x0, fa, width=width, align="edge",
                       color=pal["gain"], alpha=0.85, linewidth=0, zorder=2)
            if fd >= min_freq:
                ax.bar(x0, -fd, width=width, align="edge",
                       color=pal["loss"], alpha=0.85, linewidth=0, zorder=2)

        y_max = max(
            loci["freq_amp"].max() if not loci.empty else 0,
            loci["freq_del"].max() if not loci.empty else 0,
            0.1,
        )
        y_cap = 0.60   # cap at 60% so shallow events don't compress the scale
        ax.set_ylim(-y_cap, y_cap)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("% patients", fontsize=8, color="#444444", labelpad=5)
        ax.set_title(
            f"{entity}   (n = {n_patients} patients, one sample per patient)",
            fontsize=11, fontweight="bold", color="#222222", pad=6, loc="left",
        )
        ax.text(GENOME_SIZE * 1.002,  y_cap * 0.5, "Gain",
                fontsize=8, color=pal["gain"], va="center", ha="left", fontweight="bold")
        ax.text(GENOME_SIZE * 1.002, -y_cap * 0.5, "Loss",
                fontsize=8, color=pal["loss"], va="center", ha="left", fontweight="bold")

    fig.suptitle("Genome-Wide Copy-Number Frequencies — ichorCNA",
                 fontsize=14, fontweight="bold", color="#222222", y=0.98)
    _save(fig, output_path)


def plot_broad_cna_frequencies_lowtf(
        broad_tsv:   Path,
        tf_tsv:      Path,
        output_path: Path,
        bin_size:    int   = 5_000_000,
        min_freq:    float = 0.05,
        tf_low:      float = 0.03,
        tf_high:     float = 0.10,
        entities:    list[str] | None = None,
) -> None:
    """
    Genome-wide broad CNA frequency plot restricted to samples with
    tf_low ≤ TF < tf_high (default 3 %–10 %, the near-LOD window).

    The broad_tsv must already contain per-entity rows produced by
    stream_broad_cna_frequencies run on the low-TF sample subset.
    One panel per entity, gains upward, losses downward, y-axis capped
    at ±60 %.  The subtitle carries the TF window so the plot is
    self-documenting.

    Parameters
    ----------
    broad_tsv   : path to the low-TF broad_cna_frequencies_lowtf.tsv
    tf_tsv      : path to tf_summary.tsv  (used to count patients)
    output_path : stem path for output (.pdf and .png added)
    bin_size    : fixed bin width in bp for re-binning (default 5 Mbp)
    min_freq    : bins below this frequency are not drawn (default 5 %)
    tf_low      : lower TF bound (inclusive, default 0.03)
    tf_high     : upper TF bound (exclusive, default 0.10)
    entities    : restrict to these entities (None = all)
    """
    broad_df = pd.read_csv(broad_tsv, sep="\t", low_memory=False)
    tf_df    = pd.read_csv(tf_tsv,    sep="\t", low_memory=False)

    broad_df["chrom"] = broad_df["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )
    broad_df = broad_df[broad_df["chrom"].isin(_OFFSETS)]

    # Restrict tf_df to the low-TF window to get accurate patient counts
    tf_window = tf_df[
        (tf_df["tumor_fraction"] >= tf_low) &
        (tf_df["tumor_fraction"] <  tf_high)
    ].copy()
    tf_window["entity"] = tf_window["patient"].apply(
        lambda p: entity_from_sample(str(p))
    )

    # One sample per patient (highest TF in window)
    tf_window = (
        tf_window.sort_values("tumor_fraction", ascending=False)
                 .drop_duplicates(subset=["patient"], keep="first")
    )

    all_entities = sorted(
        tf_window["entity"].unique().tolist(),
        key=lambda e: list(ICHORCNA_ENTITY_PALETTE).index(e)
        if e in ICHORCNA_ENTITY_PALETTE else 99,
    )
    if entities:
        all_entities = [e for e in all_entities if e in entities]
    if not all_entities:
        print("      [WARN] No entities found in low-TF window — skipping low-TF broad CNA plot.")
        return

    n = len(all_entities)
    fig = plt.figure(figsize=(22, 4.2 * n + 1.5), facecolor="white")
    gs  = GridSpec(n, 1, figure=fig,
                   hspace=0.55, top=0.90, bottom=0.06, left=0.07, right=0.96)

    for ax_idx, entity in enumerate(all_entities):
        ax  = fig.add_subplot(gs[ax_idx])
        pal = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
        _style_ax(ax)

        n_patients = int(
            (tf_window["entity"] == entity).sum()
        )
        if n_patients == 0:
            continue

        # Per-entity rows from the pre-filtered broad_tsv
        if "entity" in broad_df.columns:
            loci = broad_df[broad_df["entity"] == entity].copy()
        else:
            loci = broad_df.copy()

        if loci.empty:
            ax.set_title(
                f"{entity}   (n = {n_patients} patients, "
                f"TF {tf_low*100:.0f}–{tf_high*100:.0f}%) — no altered loci",
                fontsize=11, fontweight="bold", color="#222222", pad=6, loc="left",
            )
            continue

        for _, row in loci.iterrows():
            chrom = row["chrom"]
            x0    = _genome_x(chrom, int(row["bin_start"]))
            width = int(row["bin_end"]) - int(row["bin_start"])
            fa    = row["freq_amp"]
            fd    = row["freq_del"]
            if fa >= min_freq:
                ax.bar(x0, fa, width=width, align="edge",
                       color=pal["gain"], alpha=0.85, linewidth=0, zorder=2)
            if fd >= min_freq:
                ax.bar(x0, -fd, width=width, align="edge",
                       color=pal["loss"], alpha=0.85, linewidth=0, zorder=2)

        y_cap = 0.60
        ax.set_ylim(-y_cap, y_cap)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("% patients", fontsize=8, color="#444444", labelpad=5)
        ax.set_title(
            f"{entity}   (n = {n_patients} patients, one sample per patient, "
            f"TF {tf_low*100:.0f}–{tf_high*100:.0f}%)",
            fontsize=11, fontweight="bold", color="#222222", pad=6, loc="left",
        )
        ax.text(GENOME_SIZE * 1.002,  y_cap * 0.5, "Gain",
                fontsize=8, color=pal["gain"], va="center", ha="left", fontweight="bold")
        ax.text(GENOME_SIZE * 1.002, -y_cap * 0.5, "Loss",
                fontsize=8, color=pal["loss"], va="center", ha="left", fontweight="bold")

    tf_range_str = f"TF {tf_low*100:.0f}%–{tf_high*100:.0f}%"
    fig.suptitle(
        f"Genome-Wide Copy-Number Frequencies — ichorCNA  ({tf_range_str})",
        fontsize=14, fontweight="bold", color="#222222", y=0.98,
    )
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Focal amplification frequency plot
# ---------------------------------------------------------------------------

def plot_focal_amp_frequencies(
        focal_freq_tsv: Path,
        tf_tsv:         Path,
        output_path:    Path,
        min_freq:       float = 0.0,
        top_n_labels:   int   = 30,
        entities:       list[str] | None = None,
) -> None:
    """
    Genome-wide coverage-style bar plot of focal amplification frequencies.

    Each 50 kbp bin is drawn as a bar whose height = fraction of patients
    in that entity with a focal amp overlapping that bin.  Adjacent bins
    from the same focal amp region form a visible plateau, making the
    minimal common region (MCR) immediately apparent as the tallest plateau.

    COSMIC CGC gene loci are highlighted in a contrasting colour and
    labelled above the plot.
    """
    freq_df = pd.read_csv(focal_freq_tsv, sep="\t", low_memory=False)
    tf_df   = pd.read_csv(tf_tsv,         sep="\t", low_memory=False)

    if freq_df.empty:
        print("      [WARN] focal_amp_frequencies.tsv is empty — skipping plot.")
        return

    freq_df["chrom"] = freq_df["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )

    # Normalise column names for both old and new schema
    if "bin_start" in freq_df.columns:
        freq_df["start"] = freq_df["bin_start"]
        freq_df["end"]   = freq_df["bin_end"]

    pat_col = "patients" if "patients" in freq_df.columns else "samples"

    # Use entity column if already present (e.g. cBioPortal data where
    # patient IDs don't follow the B/P/C naming convention), else derive.
    if "entity" in tf_df.columns and tf_df["entity"].notna().any():
        tf_df = tf_df.copy()
        tf_df["entity"] = tf_df["entity"].str.capitalize()
    else:
        tf_df["entity"] = tf_df["patient"].apply(
            lambda p: entity_from_sample(str(p)))

    # The 'patients' column stores patient IDs (e.g. B85, P10).
    # Match against tf_df["patient"] — not tf_df["sample"].
    entity_patients_map: dict[str, set] = {}
    entity_n_patients:   dict[str, int] = {}
    for _, row in tf_df.iterrows():
        ent = row["entity"]
        entity_patients_map.setdefault(ent, set()).add(str(row["patient"]))
    for ent, grp in tf_df.groupby("entity"):
        entity_n_patients[ent] = grp["patient"].nunique()

    all_entities = sorted(
        [e for e in entity_patients_map if entity_patients_map[e]],
        key=lambda e: list(ICHORCNA_ENTITY_PALETTE).index(e)
        if e in ICHORCNA_ENTITY_PALETTE else 99,
    )
    if entities:
        all_entities = [e for e in all_entities if e in entities]
    if not all_entities:
        print("      [WARN] No entities found — skipping focal amp plot.")
        return

    # Debug: confirm patient ID overlap
    all_freq_patients = set(
        s.strip()
        for ss in freq_df[pat_col].dropna()
        for s in str(ss).split(",") if s.strip()
    )
    all_tf_patients = set(tf_df["patient"].astype(str))
    overlap_count   = len(all_freq_patients & all_tf_patients)
    print(f"      [focal freq] {len(all_freq_patients)} unique patients in freq TSV, "
          f"{len(all_tf_patients)} in tf_summary, {overlap_count} overlap")
    if overlap_count == 0 and all_freq_patients:
        print(f"      [focal freq] freq patients: {sorted(all_freq_patients)[:3]}")
        print(f"      [focal freq] tf patients:   {sorted(all_tf_patients)[:3]}")

    n = len(all_entities)
    fig = plt.figure(figsize=(22, 4.2 * n + 1.5), facecolor="white")
    gs  = GridSpec(n, 1, figure=fig,
                   hspace=0.55, top=0.93, bottom=0.06, left=0.07, right=0.96)

    for ax_idx, entity in enumerate(all_entities):
        ax  = fig.add_subplot(gs[ax_idx])
        pal = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
        _style_ax(ax)

        ent_patients = entity_patients_map.get(entity, set())
        n_patients   = entity_n_patients.get(entity, len(ent_patients))
        n_samples    = tf_df[tf_df["entity"] == entity]["sample"].nunique()

        # Count how many patients from this entity appear in each bin
        def _entity_freq(patients_str) -> float:
            if pd.isna(patients_str) or not patients_str:
                return 0.0
            n = sum(1 for p in str(patients_str).split(",")
                    if p.strip() in ent_patients)
            return n / n_patients if n_patients > 0 else 0.0

        loci = freq_df[freq_df["chrom"].isin(_OFFSETS)].copy()
        loci["freq_entity"] = loci[pat_col].apply(_entity_freq)

        # Diagnostic: show per-entity hit counts
        n_nonzero = (loci["freq_entity"] > 0).sum()
        max_freq  = loci["freq_entity"].max() if not loci.empty else 0
        print(f"      [{entity}] samples={n_samples}  patients={n_patients}  "
              f"bins_with_signal={n_nonzero}  max_freq={max_freq:.4f}")

        # Fallback: if per-entity counting yields nothing use global frequency
        if loci["freq_entity"].sum() == 0 and "frequency" in loci.columns:
            print(f"      [WARN] {entity}: per-entity sample matching yielded "
                  f"zero — falling back to global frequency.")
            n_global = int(freq_df["n_total_patients"].iloc[0]) \
                       if "n_total_patients" in freq_df.columns \
                       else int(freq_df.get("n_total", pd.Series(1)).iloc[0])
            scale = n_patients / n_global if n_global > 0 else 1.0
            loci["freq_entity"] = loci["frequency"] * scale

        loci = loci[loci["freq_entity"] >= min_freq].copy()

        if loci.empty:
            ax.set_title(
                f"{entity}   (n = {n_samples} samples, {n_patients} patients)"
                f"  —  no focal amps >= {min_freq:.0%}",
                fontsize=10, color="#444444", pad=6, loc="left",
            )
            continue

        loci["has_gene"] = (
            loci.get("cosmic_genes", pd.Series("", index=loci.index))
                .fillna("").astype(str) != ""
        )

        # Draw as bars (50 kbp wide) — adjacent bins from the same focal amp
        # region form a continuous plateau revealing the MCR.
        # Fixed two-colour scheme: CGC driver gene = #d7191c, no known driver = #2c7bb6
        for _, row in loci.iterrows():
            x0    = _genome_x(row["chrom"], int(row["start"]))
            width = int(row["end"]) - int(row["start"])
            freq  = row["freq_entity"]
            color = "#d7191c" if row["has_gene"] else "#2c7bb6"
            ax.bar(x0, freq, width=width, align="edge",
                   color=color, alpha=0.85, linewidth=0, zorder=2)

        # Gene labels at the highest-frequency bin per gene
        gene_peaks: dict[str, tuple] = {}   # gene → (freq, x_mid)
        for _, row in loci[loci["has_gene"]].iterrows():
            x_mid = _genome_x(row["chrom"],
                               (int(row["start"]) + int(row["end"])) / 2)
            for gene in str(row["cosmic_genes"]).split(","):
                gene = gene.strip()
                if not gene or gene == "nan":
                    continue
                if gene not in gene_peaks or row["freq_entity"] > gene_peaks[gene][0]:
                    gene_peaks[gene] = (row["freq_entity"], x_mid)

        # Sort by frequency and label top N, skipping overlapping labels
        label_xs: list[float] = []
        label_gap = GENOME_SIZE * 0.010
        for gene, (freq, x_mid) in sorted(
                gene_peaks.items(), key=lambda kv: -kv[1][0])[:top_n_labels]:
            if any(abs(x_mid - lx) < label_gap for lx in label_xs):
                continue
            label_xs.append(x_mid)
            ax.text(x_mid, freq + 0.008, gene,
                    fontsize=6.5, color="#d7191c",
                    ha="center", va="bottom", rotation=90,
                    fontweight="bold", zorder=4, clip_on=True)

        y_max = max(loci["freq_entity"].max(), 0.05)
        ax.set_ylim(0, min(1.05, y_max * 1.45 + 0.02))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("% patients", fontsize=8, color="#444444", labelpad=5)

        n_bins = len(loci)
        ax.set_title(
            f"{entity}   (n = {n_samples} samples, {n_patients} patients, "
            f"{n_bins} focal amp bins >= {min_freq:.0%})",
            fontsize=11, fontweight="bold", color="#222222", pad=6, loc="left",
        )
        handles = [
            mpatches.Patch(color="#2c7bb6", label="Focal amp (no known CGC driver)"),
            mpatches.Patch(color="#d7191c", label="Overlaps COSMIC CGC driver gene"),
        ]
        ax.legend(handles=handles, fontsize=7.5, loc="lower left",
                  facecolor="white", edgecolor="#cccccc",
                  labelcolor="#222222", framealpha=0.85)

    fig.suptitle("Focal Amplification Frequencies — ichorCNA",
                 fontsize=14, fontweight="bold", color="#222222", y=0.98)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Per-entity focal frequency helper (kept for emergent amp plot compatibility)
# ---------------------------------------------------------------------------

def _per_entity_focal_freq(
        freq_df:        pd.DataFrame,
        ent_patients:   set[str],
        patient_entity: "dict[str, str] | None" = None,
        entity:         str = "",
) -> pd.DataFrame:
    """
    Re-compute per-entity focal amp frequency from a focal frequency DataFrame.
    The 'patients' column is used if present, otherwise falls back to 'samples'.
    """
    df = freq_df.copy()
    pat_col = "patients" if "patients" in df.columns else "samples"

    def _count(patients_str: str) -> int:
        if pd.isna(patients_str) or patients_str == "":
            return 0
        return sum(1 for p in str(patients_str).split(",")
                   if p.strip() in ent_patients)

    n_total           = len(ent_patients) if ent_patients else 1
    df["n_entity"]    = df[pat_col].apply(_count)
    df["freq_entity"] = df["n_entity"] / n_total
    return df


# ---------------------------------------------------------------------------
# Emergent focal amplification plot
# ---------------------------------------------------------------------------

def plot_emergent_focal_amps(
        emergent_tsv: Path,
        output_path:  Path,
        min_patients: int   = 1,
        top_n_labels: int   = 30,
        entities:     list[str] | None = None,
) -> None:
    """
    Genome-wide lollipop plot of focal amplifications that emerge in
    subsequent samples (absent at baseline) per entity.

    Each locus is shown as a stem+circle at its genomic midpoint.
    Height = number of patients in which the event emerged.
    Loci overlapping COSMIC CGC genes are highlighted and labelled.

    Parameters
    ----------
    emergent_tsv  : path to emergent_focal_amps.tsv
    output_path   : stem path for output (.pdf and .png added)
    min_patients  : minimum number of patients for a locus to be plotted
                    (default 1 = show all emergent events)
    top_n_labels  : max gene labels per panel
    entities      : restrict to these entities (None = all)
    """
    df = pd.read_csv(emergent_tsv, sep="\t", low_memory=False)
    if df.empty:
        print("      [WARN] No emergent focal amps — skipping plot.")
        return

    df["chrom"] = df["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )

    all_entities = sorted(
        [e for e in df["entity"].unique() if pd.notna(e)],
        key=lambda e: list(ICHORCNA_ENTITY_PALETTE).index(e)
        if e in ICHORCNA_ENTITY_PALETTE else 99,
    )
    if entities:
        all_entities = [e for e in all_entities if e in entities]
    if not all_entities:
        print("      [WARN] No entities in emergent focal amp table — skipping plot.")
        return

    n   = len(all_entities)
    fig = plt.figure(figsize=(22, 4.5 * n + 1.5), facecolor="white")
    gs  = GridSpec(n, 1, figure=fig,
                   hspace=0.6, top=0.93, bottom=0.05, left=0.07, right=0.96)

    for ax_idx, entity in enumerate(all_entities):
        ax   = fig.add_subplot(gs[ax_idx])
        pal  = ICHORCNA_ENTITY_PALETTE.get(entity, ICHORCNA_ENTITY_PALETTE["Other"])
        _style_ax(ax)

        ent_df = df[df["entity"] == entity]

        # Aggregate: per locus, count distinct patients with an emergent event
        loci = (
            ent_df.groupby(["chrom", "start", "end"])
                  .agg(
                      n_patients   = ("patient",     "nunique"),
                      cosmic_genes = ("cosmic_genes",
                                      lambda x: ",".join(
                                          sorted({g for v in x
                                                  for g in str(v).split(",")
                                                  if g and g != "nan"})
                                      )),
                      mean_cn      = ("copy_number",  "mean"),
                  )
                  .reset_index()
        )
        loci = loci[
            (loci["n_patients"] >= min_patients) &
            (loci["chrom"].isin(_OFFSETS))
        ].copy()

        if loci.empty:
            ax.set_title(
                f"{entity}  —  no emergent focal amps in "
                f">= {min_patients} patient(s)",
                fontsize=10, color="#444444", pad=6, loc="left",
            )
            continue

        loci["mid_genome"] = loci.apply(
            lambda r: _genome_x(r["chrom"], (r["start"] + r["end"]) / 2),
            axis=1,
        )
        loci["has_gene"] = loci["cosmic_genes"].fillna("").astype(str) != ""

        # Draw lollipops — height = n_patients
        # Fixed two-colour scheme matching the focal amp frequency plots:
        # CGC driver gene = #d7191c, no known driver = #2c7bb6
        for _, row in loci.iterrows():
            x     = row["mid_genome"]
            h     = row["n_patients"]
            color = "#d7191c" if row["has_gene"] else "#2c7bb6"
            ax.vlines(x, 0, h, colors=color, linewidth=1.4,
                      alpha=0.8, zorder=2)
            ax.scatter(x, h, color=color, s=22, zorder=3,
                       edgecolors="none", alpha=0.95)

        # Gene labels (non-overlapping, top loci by patient count)
        labelled  = loci[loci["has_gene"]].nlargest(top_n_labels, "n_patients")
        label_xs: list[float] = []
        label_gap = GENOME_SIZE * 0.012

        for _, row in labelled.iterrows():
            x    = row["mid_genome"]
            h    = row["n_patients"]
            gene = str(row["cosmic_genes"])
            if not gene or gene == "nan":
                continue
            if any(abs(x - lx) < label_gap for lx in label_xs):
                continue
            label_xs.append(x)
            ax.text(x, h + 0.3, gene.split(",")[0],
                    fontsize=6.5, color="#d7191c",
                    ha="center", va="bottom", rotation=90,
                    fontweight="bold", zorder=4)

        # Y axis — integer patient counts
        y_max    = int(loci["n_patients"].max())
        y_ticks  = list(range(0, y_max + 2))
        ax.set_ylim(-0.5, y_max + 1.5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in y_ticks],
                           fontsize=7, color="#444444")
        ax.set_ylabel("Patients with\nemergent amp", fontsize=8,
                      color="#444444", labelpad=5)

        n_pts   = ent_df["patient"].nunique()
        n_loci  = len(loci)
        ax.set_title(
            f"{entity}  —  Emergent Focal Amplifications   "
            f"(tracking {n_pts} patients, {n_loci} emergent loci "
            f"in >= {min_patients} patient(s))",
            fontsize=11, fontweight="bold", color="#222222",
            pad=6, loc="left",
        )

        handles = [
            mpatches.Patch(color="#2c7bb6", label="Emergent focal amp (no known CGC driver)"),
            mpatches.Patch(color="#d7191c", label="Overlaps COSMIC CGC driver gene"),
        ]
        ax.legend(handles=handles, fontsize=7.5, loc="lower left",
                  facecolor="white",   edgecolor="#cccccc",
                  labelcolor="#222222", framealpha=0.85)

    fig.suptitle(
        "Emergent Focal Amplifications in Subsequent Samples — ichorCNA",
        fontsize=14, fontweight="bold", color="#222222", y=0.98,
    )
    _save(fig, output_path)