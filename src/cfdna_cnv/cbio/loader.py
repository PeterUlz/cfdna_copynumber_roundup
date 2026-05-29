"""
cBioPortal segmentation file loading and genomic bin creation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.constants import CHROM_LENGTHS, CHROM_ORDER


def normalize_chrom(c: str) -> str:
    c = str(c).strip()
    return c if c.startswith("chr") else f"chr{c}"


def load_seg_file(path: Path, entity: str) -> pd.DataFrame:
    """
    Load a cBioPortal .seg file. Flexible column detection handles the
    standard cBioPortal format (ID, chrom, loc.start, loc.end, num.mark,
    seg.mean) as well as common variants.

    Returns a DataFrame with columns:
        patient, entity, chrom, start, end, num_mark, seg_mean
    """
    try:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    except Exception as e:
        raise ValueError(f"Could not read {path}: {e}") from e

    df.columns = [c.strip() for c in df.columns]
    col_lower  = {c.lower().replace(".", "_").replace(" ", "_"): c
                  for c in df.columns}

    def _get(*candidates: str) -> str | None:
        for c in candidates:
            if c in col_lower:
                return col_lower[c]
        return None

    id_col    = _get("id", "sample", "sample_id", "patient_id")
    chr_col   = _get("chrom", "chr", "chromosome", "seqnames")
    start_col = _get("loc_start", "start", "chromstart")
    end_col   = _get("loc_end",   "end",   "chromend")
    mean_col  = _get("seg_mean",  "log2",  "value",  "mean")

    missing = [lbl for lbl, col in [
        ("ID", id_col), ("chrom", chr_col),
        ("loc.start", start_col), ("loc.end", end_col), ("seg.mean", mean_col),
    ] if col is None]
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}. Found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "patient":  df[id_col].astype(str).str.strip(),
        "entity":   entity,
        "chrom":    df[chr_col].apply(normalize_chrom),
        "start":    pd.to_numeric(df[start_col], errors="coerce"),
        "end":      pd.to_numeric(df[end_col],   errors="coerce"),
        "seg_mean": pd.to_numeric(df[mean_col],  errors="coerce"),
    })

    out = (out[out["chrom"].isin(CHROM_LENGTHS)]
           .dropna(subset=["start", "end", "seg_mean"]))
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)

    n_patients = out["patient"].nunique()
    print(f"      {entity:12s}: {len(out):6d} segments, "
          f"{n_patients} patients  ({path.name})")
    return out


def load_entity_files(
        entity_files: dict[str, list[Path]],
        min_seg_size: int = 0,
) -> dict[str, pd.DataFrame]:
    """
    Load and optionally merge multiple seg files per entity.
    Returns {entity: DataFrame}.
    """
    seg_by_entity: dict[str, pd.DataFrame] = {}
    for entity, paths in entity_files.items():
        frames = [load_seg_file(p, entity) for p in paths]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        if len(paths) > 1:
            print(f"      {entity}: merged {len(paths)} files → "
                  f"{df['patient'].nunique()} unique patients, "
                  f"{len(df)} segments total.")
        if min_seg_size > 0:
            before = len(df)
            df = df[(df["end"] - df["start"]) >= min_seg_size]
            if before > len(df):
                print(f"      {entity}: removed {before - len(df)} segments "
                      f"< {min_seg_size} bp")
        seg_by_entity[entity] = df
    return seg_by_entity


def make_bins(bin_size: int = 5_000_000) -> pd.DataFrame:
    """
    Create fixed-size genomic bins across all chromosomes in CHROM_ORDER.
    Returns DataFrame with columns: chrom, bin_start, bin_end, bin_id.
    """
    rows = []
    for chrom in CHROM_ORDER:
        chrom_len = CHROM_LENGTHS[chrom]
        for s in range(0, chrom_len, bin_size):
            rows.append({
                "chrom":     chrom,
                "bin_start": s,
                "bin_end":   min(s + bin_size, chrom_len),
            })
    df = pd.DataFrame(rows)
    df["bin_id"] = range(len(df))
    return df
