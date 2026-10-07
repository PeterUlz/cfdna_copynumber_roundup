"""
Longitudinal copy-number comparison across serial samples of the same patient.
"""
from __future__ import annotations

import pandas as pd

from ..common.io import parse_patient_timepoint


def longitudinal_comparison(
        focal_df: pd.DataFrame,
        tf_df:    pd.DataFrame,
) -> pd.DataFrame:
    """
    For each patient with > 1 timepoint, compare CN of focal amplification
    loci across subsequent samples.

    Returns a long-format DataFrame with one row per
    (patient, chrom, start, end, timepoint), including a cn_delta_from_prev
    column showing change from the previous timepoint.
    """
    if focal_df.empty or "sample" not in focal_df.columns:
        return pd.DataFrame()

    pt_map = tf_df.set_index("sample")[["patient", "timepoint"]].to_dict("index")

    df = focal_df.copy()
    df["patient"]   = df["sample"].map(lambda s: pt_map.get(s, {}).get("patient", s))
    df["timepoint"] = df["sample"].map(lambda s: pt_map.get(s, {}).get("timepoint"))

    # Keep only patients with > 1 timepoint
    multi = df.groupby("patient")["timepoint"].nunique()
    multi_patients = multi[multi > 1].index.tolist()
    df = df[df["patient"].isin(multi_patients)].copy()
    if df.empty:
        return pd.DataFrame()

    # Aggregation — copy.number is required; gene annotation is optional
    agg_dict: dict = {
        "copy_number": ("copy.number", "mean"),
        "sample":      ("sample", "first"),
    }
    for gene_col in ("cosmic_genes", "oncogenes"):
        if gene_col in df.columns:
            agg_dict["genes"] = (gene_col, "first")
            break

    pivot = (
        df.groupby(["patient", "chrom", "start", "end", "timepoint"])
          .agg(**agg_dict)
          .reset_index()
          .sort_values(["patient", "chrom", "start", "end", "timepoint"])
    )

    pivot["cn_delta_from_prev"] = (
        pivot.groupby(["patient", "chrom", "start", "end"])["copy_number"]
             .diff()
    )

    return pivot.reset_index(drop=True)


def emergent_focal_amps(
        focal_df: pd.DataFrame,
        tf_df:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify focal amplifications that emerge in subsequent samples relative
    to each patient's baseline (the sample with the highest tumor fraction).

    A focal amp is considered "emergent" at a given timepoint if:
      - It is present in that sample's focal amp events, AND
      - It is absent from the baseline sample's focal amp events
        (no overlapping segment with an AMP call in the baseline).

    Overlap is defined as any coordinate intersection between the emergent
    segment and any baseline segment on the same chromosome.

    Parameters
    ----------
    focal_df : focal amplification events (from focal_amplifications.tsv).
               Required columns: sample, chrom, start, end.
               Optional: copy.number, cosmic_genes, subclone.status.
    tf_df    : tumor fraction table (from tf_summary.tsv).
               Required columns: sample, patient, timepoint, tumor_fraction.

    Returns
    -------
    Long-format DataFrame with one row per emergent event, columns:
        patient, entity,
        baseline_sample, baseline_tf,
        comparison_sample, comparison_timepoint, comparison_tf,
        chrom, start, end, seg_length,
        copy_number, cosmic_genes, subclone_status,
        is_emergent  (always True here, for downstream filtering convenience)
    """
    if focal_df.empty or "sample" not in focal_df.columns:
        return pd.DataFrame()

    # Build sample metadata lookup
    tf_indexed = tf_df.set_index("sample")
    required   = {"patient", "timepoint", "tumor_fraction"}
    if not required.issubset(tf_indexed.columns):
        missing = required - set(tf_indexed.columns)
        raise ValueError(f"tf_df missing columns: {missing}")

    def _meta(s: str) -> dict:
        if s in tf_indexed.index:
            r = tf_indexed.loc[s]
            return {
                "patient":        r["patient"],
                "timepoint":      r["timepoint"],
                "tumor_fraction": r["tumor_fraction"],
            }
        return {"patient": s, "timepoint": None, "tumor_fraction": None}

    df = focal_df.copy()
    df["patient"]        = df["sample"].map(lambda s: _meta(s)["patient"])
    df["timepoint"]      = df["sample"].map(lambda s: _meta(s)["timepoint"])
    df["tumor_fraction"] = df["sample"].map(lambda s: _meta(s)["tumor_fraction"])

    # Keep only patients with > 1 timepoint in the focal amp table
    multi = df.groupby("patient")["timepoint"].nunique()
    multi_patients = multi[multi > 1].index.tolist()
    df = df[df["patient"].isin(multi_patients)].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []

    for patient, pat_df in df.groupby("patient"):
        # Baseline = sample with highest tumor fraction among this patient's
        # samples that appear in the focal amp table
        tf_per_sample = (
            pat_df.groupby("sample")["tumor_fraction"]
                  .first()
                  .dropna()
        )
        if tf_per_sample.empty:
            continue
        baseline_sample = tf_per_sample.idxmax()
        baseline_tf     = float(tf_per_sample[baseline_sample])

        # Baseline focal amp loci: set of (chrom, start, end)
        baseline_segs = pat_df[pat_df["sample"] == baseline_sample][
            ["chrom", "start", "end"]
        ].copy()

        # Build per-chromosome interval list for fast overlap check
        baseline_by_chrom: dict[str, list[tuple[int, int]]] = {}
        for _, r in baseline_segs.iterrows():
            baseline_by_chrom.setdefault(r["chrom"], []).append(
                (int(r["start"]), int(r["end"]))
            )

        def _overlaps_baseline(chrom: str, start: int, end: int) -> bool:
            for bs, be in baseline_by_chrom.get(chrom, []):
                if start < be and end > bs:
                    return True
            return False

        # Comparison samples = all other samples for this patient
        other_samples = pat_df[pat_df["sample"] != baseline_sample]["sample"].unique()

        for comp_sample in other_samples:
            comp_meta = _meta(comp_sample)
            comp_tf   = comp_meta["tumor_fraction"]
            comp_tp   = comp_meta["timepoint"]
            comp_segs = pat_df[pat_df["sample"] == comp_sample]

            for _, row in comp_segs.iterrows():
                chrom = row["chrom"]
                start = int(row["start"])
                end   = int(row["end"])

                if _overlaps_baseline(chrom, start, end):
                    continue  # not emergent — already present at baseline

                rows.append({
                    "patient":              patient,
                    "baseline_sample":      baseline_sample,
                    "baseline_tf":          baseline_tf,
                    "comparison_sample":    comp_sample,
                    "comparison_timepoint": comp_tp,
                    "comparison_tf":        comp_tf,
                    "chrom":                chrom,
                    "start":                start,
                    "end":                  end,
                    "seg_length":           end - start,
                    "copy_number":          row.get("copy.number", None),
                    "cosmic_genes":         row.get("cosmic_genes", ""),
                    "subclone_status":      row.get("subclone.status", ""),
                    "is_emergent":          True,
                })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Attach entity from patient first letter
    _ENTITY_MAP = {
        "B": "Breast", "C": "Colorectal", "P": "Prostate",
        "L": "Lung",   "E": "Ovarian",
    }
    result["entity"] = result["patient"].str[0].str.upper().map(
        _ENTITY_MAP
    ).fillna("Other")

    return result.sort_values(
        ["entity", "patient", "comparison_timepoint", "chrom", "start"]
    ).reset_index(drop=True)