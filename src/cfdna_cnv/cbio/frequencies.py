"""
CNA frequency computation for cBioPortal segmentation data:
  - Broad (binned) CNA frequencies
  - Focal amplification detection and population-level frequencies
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.constants import FOCAL_MIN_BP, FOCAL_MAX_BP
from ..common.cosmic import annotate_with_cgc
from .loader import make_bins


# ---------------------------------------------------------------------------
# Broad CNA frequencies (fixed genomic bins)
# ---------------------------------------------------------------------------

def assign_seg_call(seg_mean: float, gain_thr: float, loss_thr: float) -> str:
    if seg_mean >= gain_thr:
        return "GAIN"
    if seg_mean <= loss_thr:
        return "LOSS"
    return "NEUTRAL"


def compute_frequencies(
        seg_df:          pd.DataFrame,
        bins_df:         pd.DataFrame,
        gain_threshold:  float = 0.2,
        loss_threshold:  float = -0.2,
        min_overlap_bp:  int   = 1,
) -> pd.DataFrame:
    """
    Compute per-bin GAIN/LOSS frequencies for one entity.
    Uses an overlap-length-weighted mean seg.mean per patient per bin.

    Returns bins_df extended with:
        n_patients, n_gain, n_loss, n_neutral,
        freq_gain, freq_loss, mean_seg_mean, entity, n_total
    """
    entity      = seg_df["entity"].iloc[0]
    n_total     = seg_df["patient"].nunique()
    seg_by_chrom = {c: grp for c, grp in seg_df.groupby("chrom")}
    results     = []

    for _, bin_row in bins_df.iterrows():
        chrom     = bin_row["chrom"]
        bin_start = bin_row["bin_start"]
        bin_end   = bin_row["bin_end"]
        segs      = seg_by_chrom.get(chrom)

        if segs is None:
            results.append(_empty_bin(n_total))
            continue

        overlap = segs[
            (segs["end"] > bin_start) & (segs["start"] < bin_end)
        ].copy()
        if overlap.empty:
            results.append(_empty_bin(n_total))
            continue

        overlap["overlap_bp"] = overlap.apply(
            lambda r: min(r["end"], bin_end) - max(r["start"], bin_start),
            axis=1,
        )
        overlap = overlap[overlap["overlap_bp"] >= min_overlap_bp]

        patient_calls = []
        for pat, pat_segs in overlap.groupby("patient"):
            w      = pat_segs["overlap_bp"].values
            vals   = pat_segs["seg_mean"].values
            w_mean = float(np.average(vals, weights=w))
            patient_calls.append({
                "patient":  pat,
                "seg_mean": w_mean,
                "call":     assign_seg_call(w_mean, gain_threshold, loss_threshold),
            })

        call_df    = pd.DataFrame(patient_calls)
        n_patients = len(call_df)
        n_gain     = int((call_df["call"] == "GAIN").sum())
        n_loss     = int((call_df["call"] == "LOSS").sum())
        results.append({
            "n_patients":    n_patients,
            "n_gain":        n_gain,
            "n_loss":        n_loss,
            "n_neutral":     int((call_df["call"] == "NEUTRAL").sum()),
            "freq_gain":     n_gain / n_total,
            "freq_loss":     n_loss / n_total,
            "mean_seg_mean": float(call_df["seg_mean"].mean()),
        })

    result_df = pd.concat(
        [bins_df.reset_index(drop=True), pd.DataFrame(results)], axis=1
    )
    result_df["entity"]  = entity
    result_df["n_total"] = n_total
    return result_df


def _empty_bin(n_total: int) -> dict:
    return {
        "n_patients": 0, "n_gain": 0, "n_loss": 0, "n_neutral": 0,
        "freq_gain":  0.0, "freq_loss": 0.0, "mean_seg_mean": float("nan"),
    }


# ---------------------------------------------------------------------------
# Focal amplification detection for cBioPortal data
# ---------------------------------------------------------------------------

def detect_focal_amps_cbio(
        seg_df:       pd.DataFrame,
        focal_min_bp: int   = FOCAL_MIN_BP,
        focal_max_bp: int   = FOCAL_MAX_BP,
        log2_min:     float = 0.5,
        log2_delta:   float = 0.2,
        flank_bp:     int   = 10_000_000,
) -> pd.DataFrame:
    """
    Detect focal amplifications in cBioPortal seg data.

    Criteria:
      1. Segment length in [focal_min_bp, focal_max_bp].
      2. seg.mean >= log2_min.
      3. seg.mean exceeds median of flanking segments within flank_bp by
         at least log2_delta.
    """
    df = seg_df.copy()
    df["seg_length"] = df["end"] - df["start"]

    candidates = df[
        (df["seg_length"] >= focal_min_bp) &
        (df["seg_length"] <= focal_max_bp) &
        (df["seg_mean"]   >= log2_min)
    ].copy()
    if candidates.empty:
        return candidates.reset_index(drop=True)

    # Build per-(patient, chrom) sorted segment list for background lookup
    seg_index: dict[tuple, list] = {}
    for _, r in df.iterrows():
        if pd.isna(r["seg_mean"]):
            continue
        key = (r["patient"], r["chrom"])
        seg_index.setdefault(key, []).append(
            (int(r["start"]), int(r["end"]), float(r["seg_mean"]))
        )
    for k in seg_index:
        seg_index[k].sort()

    def _bg(patient: str, chrom: str, s: int, e: int) -> float | None:
        entries   = seg_index.get((patient, chrom), [])
        win_start = s - flank_bp
        win_end   = e + flank_bp
        bg_vals   = [lv for ss, ee, lv in entries
                     if win_start <= (ss + ee) / 2 <= win_end
                     and min(ee, e) - max(ss, s) <= 0]
        return float(np.median(bg_vals)) if bg_vals else None

    keep, bg_out, delta_out = [], [], []
    for _, row in candidates.iterrows():
        bg = _bg(row["patient"], row["chrom"], int(row["start"]), int(row["end"]))
        if bg is None:
            keep.append(True); bg_out.append(np.nan); delta_out.append(np.nan)
        else:
            delta = float(row["seg_mean"]) - bg
            keep.append(delta >= log2_delta)
            bg_out.append(round(bg, 4))
            delta_out.append(round(delta, 4))

    candidates           = candidates[keep].copy()
    candidates["bg_seg_mean"]    = [v for v, k in zip(bg_out,   keep) if k]
    candidates["delta_seg_mean"] = [v for v, k in zip(delta_out, keep) if k]

    n_removed = sum(not k for k in keep)
    if n_removed:
        print(f"        Elevation filter removed {n_removed} candidates "
              f"(delta < {log2_delta}).")
    return candidates.reset_index(drop=True)


def compute_focal_amp_frequencies(
        focal_df:  pd.DataFrame,
        n_patients: int,
        entity:     str,
) -> pd.DataFrame:
    """Population-level frequency of each unique focal amp locus."""
    if focal_df.empty:
        return pd.DataFrame()

    agg: dict[str, tuple] = {
        "n_patients_with_amp": ("patient",  "nunique"),
        "mean_seg_mean":       ("seg_mean", "mean"),
        "max_seg_mean":        ("seg_mean", "max"),
    }
    for col in ("cosmic_genes", "cosmic_roles", "cosmic_roles_raw",
                "cosmic_tiers", "cosmic_hallmarks", "cosmic_chr_bands",
                "cosmic_tumour_types", "cosmic_tissue_types"):
        if col in focal_df.columns:
            agg[col] = (col, lambda x: ",".join(
                sorted({g for v in x
                        for g in str(v).split(",") if g and g != "nan"})
            ))

    grp = (focal_df
           .groupby(["chrom", "start", "end"])
           .agg(**agg)
           .reset_index())
    grp["entity"]     = entity
    grp["n_total"]    = n_patients
    grp["frequency"]  = grp["n_patients_with_amp"] / n_patients
    grp["seg_length"] = grp["end"] - grp["start"]

    col_order = ["entity", "chrom", "start", "end", "seg_length",
                 "n_patients_with_amp", "n_total", "frequency",
                 "mean_seg_mean", "max_seg_mean",
                 "cosmic_genes", "cosmic_roles", "cosmic_roles_raw",
                 "cosmic_tiers", "cosmic_hallmarks", "cosmic_chr_bands",
                 "cosmic_tumour_types", "cosmic_tissue_types"]
    col_order = [c for c in col_order if c in grp.columns]
    return grp[col_order].sort_values("frequency", ascending=False).reset_index(drop=True)
