"""
Quality-control checks for ichorCNA output directories.
  - find_incomplete_runs : detect sample folders missing params or seg files
  - validate_seg_file    : per-sample sanity checks on .cna.seg content
  - validate_all_seg_files : run validation across a cohort, stream to disk
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pandas as pd

from ..common.constants import (
    VALID_CHROMS, CHROM_LENGTHS, CN_MAX, SEG_MIN_BP, AMP_MIN_CN, DIPLOID_CN,
)
from ..common.io import normalize_sample_name, parse_seg_file


# ---------------------------------------------------------------------------
# Incomplete run detection
# ---------------------------------------------------------------------------

def find_incomplete_runs(
        directory:  Path,
        params_ext: str = ".params.txt",
        seg_ext:    str = ".cna.seg",
) -> pd.DataFrame:
    """
    Scan *directory* for sample sub-folders missing either the .params.txt
    or .cna.seg output file. Returns a DataFrame with columns:
        sample, directory, has_params, has_seg, status
    """
    rows = []
    try:
        top_entries = list(os.scandir(directory))
    except PermissionError:
        return pd.DataFrame()

    for entry in sorted(top_entries, key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        subdir = Path(entry.path)
        try:
            files = [f.name for f in os.scandir(subdir) if os.DirEntry.is_file(f)]
        except PermissionError:
            continue
        if not files:
            continue   # skip empty directories

        sample    = normalize_sample_name(entry.name)
        has_params = any(f.endswith(params_ext) for f in files)
        has_seg    = any(
            f.endswith(seg_ext) or
            (seg_ext == ".cna.seg" and f.endswith(".seg"))
            for f in files
        )
        if not has_params or not has_seg:
            if not has_params and not has_seg:
                status = "missing params + seg"
            elif not has_params:
                status = "missing params"
            else:
                status = "missing seg"
            rows.append({
                "sample":     sample,
                "directory":  str(subdir),
                "has_params": has_params,
                "has_seg":    has_seg,
                "status":     status,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Segment file validation
# ---------------------------------------------------------------------------

def validate_seg_file(seg_path: Path, sample: str) -> dict:
    """
    Run sanity checks on a .cna.seg file and return a report dict with keys:
        sample, seg_file, n_rows, issues, issue_count, has_errors, details

    Checks:
        1.  Required columns present
        2.  Missing / NaN values
        3.  Unrecognized chromosomes
        4.  Negative start coordinates
        5.  end <= start
        6.  Coordinates exceeding hg38 chromosome length
        7.  Suspiciously short segments (< 1 kbp)
        8.  Copy number out of range (< 0 or > 20)
        9.  Non-integer copy numbers
        10. Overlapping segments on the same chromosome
        11. Entire autosomes absent
    """
    report: dict = {
        "sample":      sample,
        "seg_file":    str(seg_path),
        "n_rows":      0,
        "issues":      [],
        "issue_count": 0,
        "has_errors":  False,
        "details":     {},
    }

    df = parse_seg_file(seg_path, sample)
    if df is None or df.empty:
        report["issues"].append("FATAL: file could not be parsed or is empty")
        report["has_errors"] = True
        return report

    report["n_rows"] = len(df)
    issues:  list[str]      = []
    details: dict           = {}

    # 1. Required columns
    missing_cols = [c for c in ("chrom", "start", "end") if c not in df.columns]
    if missing_cols:
        issues.append(f"FATAL: missing required columns: {missing_cols}")
        report["has_errors"] = True
        report["issues"] = issues
        return report

    # 2. Missing values
    for col in ("chrom", "start", "end", "copy.number"):
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            issues.append(f"missing values in '{col}': {n_null} rows")
            details[f"null_{col}"] = n_null

    # 3. Unrecognized chromosomes
    unknown = set(df["chrom"].unique()) - VALID_CHROMS
    if unknown:
        n = int(df["chrom"].isin(unknown).sum())
        issues.append(f"unrecognized chromosome(s) {sorted(unknown)}: {n} rows")
        details["unknown_chroms"] = sorted(unknown)

    # 4. Negative starts
    n_neg = int((df["start"] < 0).sum())
    if n_neg:
        issues.append(f"negative start coordinates: {n_neg} rows")
        details["neg_start"] = n_neg

    # 5. end <= start
    n_bad = int((df["end"] <= df["start"]).sum())
    if n_bad:
        issues.append(f"end <= start: {n_bad} rows")
        details["end_le_start"] = n_bad

    # 6. Exceeds chromosome length
    n_exceeds = 0
    for chrom, max_len in CHROM_LENGTHS.items():
        sub = df[df["chrom"] == chrom]
        if not sub.empty:
            n_exceeds += int((sub["end"] > max_len).sum())
    if n_exceeds:
        issues.append(f"coordinates exceed hg38 length: {n_exceeds} rows")
        details["exceeds_chrom_length"] = n_exceeds

    # 7. Short segments
    if "seg_length" in df.columns:
        n_tiny = int((df["seg_length"] < SEG_MIN_BP).sum())
        if n_tiny:
            issues.append(f"segments < {SEG_MIN_BP} bp: {n_tiny} rows")
            details["tiny_segments"] = n_tiny
        details.update({
            "seg_length_min":    int(df["seg_length"].min()),
            "seg_length_max":    int(df["seg_length"].max()),
            "seg_length_median": int(df["seg_length"].median()),
        })

    # 8–9. Copy number checks
    if "copy.number" in df.columns:
        cn = df["copy.number"].dropna()
        n_neg_cn  = int((cn < 0).sum())
        n_high_cn = int((cn > CN_MAX).sum())
        n_nonint  = int((cn != cn.round()).sum())
        if n_neg_cn:
            issues.append(f"negative copy numbers: {n_neg_cn} rows")
            details["neg_copy_number"] = n_neg_cn
        if n_high_cn:
            issues.append(f"copy number > {CN_MAX}: {n_high_cn} rows (max={cn.max():.0f})")
            details["high_copy_number"] = n_high_cn
        if n_nonint:
            issues.append(f"non-integer copy numbers: {n_nonint} rows")
            details["noninteger_cn"] = n_nonint
        if len(cn):
            details.update({
                "cn_min":    float(cn.min()),
                "cn_max":    float(cn.max()),
                "cn_median": float(cn.median()),
            })

    # 10. Overlapping segments
    n_overlaps = 0
    for _, grp in df.groupby("chrom"):
        prev_end = -1
        for _, row in grp.sort_values("start").iterrows():
            if row["start"] < prev_end:
                n_overlaps += 1
            prev_end = max(prev_end, row["end"])
    if n_overlaps:
        issues.append(f"overlapping segments: {n_overlaps} instances")
        details["overlapping_segments"] = n_overlaps

    # 11. Absent autosomes
    absent = sorted({f"chr{i}" for i in range(1, 23)} - set(df["chrom"].unique()))
    if absent:
        issues.append(f"entire chromosomes absent: {absent}")
        details["absent_chroms"] = absent

    report["issues"]      = issues
    report["issue_count"] = sum(
        v for k, v in details.items()
        if isinstance(v, int) and k not in
           ("seg_length_min", "seg_length_max", "seg_length_median")
    )
    report["has_errors"]  = any(i.startswith("FATAL") for i in issues)
    report["details"]     = details
    return report


def validate_all_seg_files(
        seg_files: dict[str, Path],
        label:     str = "",
        out_path:  Path | None = None,
) -> pd.DataFrame:
    """
    Run validate_seg_file on every sample, streaming rows to *out_path* every
    100 samples to keep peak RAM low. Returns the full summary DataFrame.
    """
    rows:           list[dict] = []
    n_clean = n_warn = n_error = 0
    header_written = False
    total          = len(seg_files)

    for idx, (sample, seg_path) in enumerate(seg_files.items(), 1):
        r        = validate_seg_file(seg_path, sample)
        n_issues = len(r["issues"])
        if r["has_errors"]:
            status  = "ERROR"; n_error += 1
        elif n_issues:
            status  = "WARN";  n_warn  += 1
        else:
            status  = "OK";    n_clean += 1

        row: dict = {
            "sample":      r["sample"],
            "resolution":  label,
            "seg_file":    r["seg_file"],
            "n_rows":      r["n_rows"],
            "status":      status,
            "n_issues":    n_issues,
            "issue_count": r["issue_count"],
            "has_errors":  r["has_errors"],
            "issues_text": " | ".join(r["issues"]),
        }
        for k, v in r["details"].items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)

        if out_path is not None and (idx % 100 == 0 or idx == total):
            chunk = pd.DataFrame(rows)
            chunk.to_csv(out_path, sep="\t", index=False,
                         mode="a" if header_written else "w",
                         header=not header_written)
            header_written = True
            rows = []
            gc.collect()

    if rows:
        chunk = pd.DataFrame(rows)
        if out_path is not None:
            chunk.to_csv(out_path, sep="\t", index=False,
                         mode="a" if header_written else "w",
                         header=not header_written)
            rows = []

    print(f"      {label}: {total} files checked — "
          f"{n_clean} OK, {n_warn} warnings, {n_error} errors.")

    if out_path is not None and out_path.exists():
        return pd.read_csv(out_path, sep="\t", low_memory=False)
    return pd.DataFrame(rows) if rows else chunk
