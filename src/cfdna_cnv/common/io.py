"""
File I/O utilities shared by both ichorCNA and cBioPortal workflows.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .constants import AMP_MIN_CN, DIPLOID_CN

# ---------------------------------------------------------------------------
# Column rename map for .cna.seg files
# ---------------------------------------------------------------------------
_SEG_RENAME: dict[str, str] = {
    "chrom": "chrom", "chr": "chrom", "chromosome": "chrom",
    "start": "start", "loc.start": "start", "start.pos": "start",
    "end":   "end",   "loc.end":   "end",   "end.pos":   "end",
    # sample ID (ichorCNA .seg.txt uses "ID")
    "id":    "sample",
    # integer copy number
    "copy.number": "copy.number", "copy_number": "copy.number",
    "cn": "copy.number", "modal_cn": "copy.number",
    "corrected_copy_number": "copy.number",
    "corrected.copy.number": "copy.number",
    # call / event
    "call": "call", "event": "call",
    "corrected_call": "call",
    "corrected.call": "call",
    # log2 ratio variants
    "seg.mean":           "logr",
    "seg.median.logr":    "logr",   # ichorCNA .seg.txt
    "seg_median_logr":    "logr",
    "median":             "logr",
    "logr":               "logr",
    "log2_ratio":         "logr",
    "logr_copy_number":   "logr",
    "logr.copy.number":   "logr",
    # other ichorCNA columns kept as-is
    "num.mark":         "num.mark",
    "subclone.status":  "subclone.status",
}

_COORD_COLS_LOWER: set[str] = {
    "chr", "chrom", "chromosome", "start", "end", "loc.start", "loc.end"
}

# Log2 columns tried (in priority order) for the broad-calling fallback
LOG2_COL_CANDIDATES: list[str] = [
    "logr",            # canonical name after _SEG_RENAME
    "seg.median.logr", # ichorCNA .seg.txt (pre-rename fallback)
    "log2_ratio", "log2ratio", "seg.mean",
    "median", "logr_copy_ratio", "copy_ratio",
    "corrected_copy_number",
]


# ---------------------------------------------------------------------------
# Sample-name helpers
# ---------------------------------------------------------------------------

# Trailing noise tokens added by sequencing pipelines, stripped iteratively.
# Handles: Illumina FASTQ suffixes (_S4, _L002, _R1, _001),
#           sequencing kit tags (_PS, _Avenio, _WGS),
#           and combinations thereof.
_SAMPLE_NOISE_RE = re.compile(
    r"[-_](?:PS|Avenio|WGS)"    # kit / prep tags
    r"|[-_]S[0-9]+"              # sample-sheet index (_S4, _S13)
    r"|[-_]L[0-9]+"              # lane (_L002)
    r"|[-_]R[0-9]+"              # read number (_R1, _R2)
    r"|[-_][0-9]{3,}$",          # trailing zero-padded number (_001)
    re.IGNORECASE,
)


def normalize_sample_name(name: str) -> str:
    """
    Normalize a sample name to its canonical patient_timepoint form.

    Steps:
      1. Replace hyphens with underscores.
      2. Strip a leading 'PS-' or 'PS_' prefix if present.
      3. Iteratively remove trailing sequencing-pipeline noise tokens
         (Illumina FASTQ suffixes, kit tags, lane/read numbers).

    Examples:
      B46-5-PS_S4_R1_001              → B46_5
      B195_1_Avenio_WGS_S13_L002_R1_001 → B195_1
      B220_5_PS_Avenio_S20_L002_R1_001   → B220_5
      B163_6_R1                          → B163_6
      C130_3                             → C130_3  (unchanged)
    """
    name = name.replace("-", "_")
    name = re.sub(r"^PS[_]", "", name)   # leading PS_ prefix
    # Iteratively strip trailing noise until stable
    prev = None
    while prev != name:
        prev = name
        name = _SAMPLE_NOISE_RE.sub("", name).rstrip("_")
    return name


def parse_patient_timepoint(sample_name: str) -> tuple[str, int | None]:
    """
    Split sample name into (patient_id, timepoint).
    Pattern: [C/B/P/E/L]xxx_x  — last underscore separates patient from timepoint.
    """
    norm = normalize_sample_name(sample_name)
    idx  = norm.rfind("_")
    if idx == -1:
        return norm, None
    patient, timepoint = norm[:idx], norm[idx + 1:]
    return (patient, int(timepoint)) if timepoint.isdigit() else (norm, None)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_sample_files(directory: Path, extension: str) -> dict[str, Path]:
    """
    Recursively walk *directory* with os.scandir (memory-efficient) and
    collect all files ending with *extension*.

    The sample name is taken from the immediate parent subdirectory name
    (ichorCNA layout: <root>/<sample>/<sample>.params.txt).
    Falls back to the file stem for flat layouts.

    Returns {normalized_sample_name: Path}.
    """
    result: dict[str, Path] = {}

    def _walk(dirpath: str):
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        yield from _walk(entry.path)
                    elif entry.is_file() and entry.name.endswith(extension):
                        yield Path(entry.path)
        except PermissionError:
            pass

    for p in _walk(str(directory)):
        raw_name = p.parent.name if p.parent != directory else p.name[: -len(extension)]
        norm = normalize_sample_name(raw_name)
        if norm not in result:
            result[norm] = p
        else:
            print(f"  [WARN] Duplicate after normalization: {p} vs {result[norm]} — keeping first.")

    return result


# ---------------------------------------------------------------------------
# .params.txt parser
# ---------------------------------------------------------------------------

def parse_params(params_path: Path) -> float | None:
    """
    Read an ichorCNA .params.txt file and return the tumor fraction (0–1).
    Tries multiple known key names across ichorCNA versions.
    """
    tf_keys = {"tumorfraction", "n_est", "tfrac", "tf"}
    try:
        with open(params_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    key = parts[0].strip().lower().replace(" ", "")
                    if key in tf_keys or ("tumor" in key and "frac" in key):
                        try:
                            return float(parts[1])
                        except ValueError:
                            pass
    except Exception as e:
        print(f"  [WARN] Could not parse {params_path}: {e}")
    return None


def parse_wig_total_reads(wig_path: Path) -> int | None:
    """
    Sum all read counts in an ichorCNA fixedStep WIG file.

    The WIG file produced by HMMcopy readCounter has the format:
        fixedStep chrom=chr1 start=1 step=1000000 span=1000000
        26283
        66885
        ...

    Returns total read count across all chromosomes, or None on failure.
    """
    total = 0
    found_any = False
    try:
        with open(wig_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("track") or line.startswith("browser"):
                    continue
                if line.startswith("fixedStep") or line.startswith("variableStep"):
                    continue
                # variableStep has two columns (pos value); fixedStep has one
                parts = line.split()
                if len(parts) == 1:
                    try:
                        total += int(float(parts[0]))
                        found_any = True
                    except ValueError:
                        pass
                elif len(parts) == 2:
                    try:
                        total += int(float(parts[1]))
                        found_any = True
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  [WARN] Could not parse WIG {wig_path}: {e}")
        return None
    return total if found_any else None


def collect_tumor_fractions(
        params_files: dict[str, Path],
        wig_files:    dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Return DataFrame with columns:
        sample, patient, timepoint, tumor_fraction, total_reads, params_file

    Parameters
    ----------
    params_files : {normalized_sample_name: Path} to .params.txt files.
    wig_files    : optional {normalized_sample_name: Path} to .wig files.
                   When provided, total reads are summed from the WIG and
                   stored in the 'total_reads' column.

    Duplicate resolution
    --------------------
    When two samples normalize to the same name (e.g. run with different
    params), the one with higher total_reads is kept. If reads are
    unavailable, the one with higher tumor_fraction is kept.
    """
    rows = []
    for sample, path in params_files.items():
        tf = parse_params(path)
        patient, tp = parse_patient_timepoint(sample)
        reads: int | None = None
        if wig_files:
            wig_path = wig_files.get(sample)
            if wig_path:
                reads = parse_wig_total_reads(wig_path)
        rows.append({
            "sample":         sample,
            "patient":        patient,
            "timepoint":      tp,
            "tumor_fraction": tf,
            "total_reads":    reads,
            "params_file":    str(path),
        })

    df = pd.DataFrame(rows)

    # Deduplicate: keep the best sample per (patient, timepoint) pair.
    # Priority: higher total_reads first, then higher tumor_fraction.
    if df.duplicated(subset=["patient", "timepoint"], keep=False).any():
        n_before = len(df)
        df = df.sort_values(
            ["patient", "timepoint", "total_reads", "tumor_fraction"],
            ascending=[True, True, False, False],
            na_position="last",
        ).drop_duplicates(subset=["patient", "timepoint"], keep="first")
        n_after = len(df)
        if n_before > n_after:
            print(f"  [INFO] Deduplicated {n_before - n_after} samples with "
                  f"identical patient/timepoint (kept highest coverage).")

    return df.sort_values(["patient", "timepoint", "sample"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# .cna.seg parser
# ---------------------------------------------------------------------------

def _strip_sample_prefix(columns: list[str]) -> list[str]:
    """
    Detect and strip a sample-name prefix that ichorCNA prepends to data columns.
    E.g. "B163_6_R1.copy.number" → "copy.number".
    Coordinate columns (chr, start, end, …) are never modified.
    """
    # Find the prefix by looking for the first non-coord column whose bare
    # suffix (after the first dot) matches a known data column name.
    prefix: str | None = None
    known_suffixes = {k.replace(".", "_") for k in _SEG_RENAME}
    known_bare = {"copy.number", "event", "logr", "corrected_call",
                  "corrected_copy_number", "logr_copy_number", "subclone.status"}
    for col in columns:
        if col.lower() in _COORD_COLS_LOWER:
            continue
        if "." in col:
            pfx, rest = col.split(".", 1)
            if (rest.lower().replace(".", "_") in known_suffixes
                    or rest.lower() in known_bare):
                prefix = pfx + "."
                break

    if prefix is None:
        return columns
    return [c[len(prefix):] if c.startswith(prefix) else c for c in columns]


def parse_seg_file(seg_path: Path, sample: str) -> pd.DataFrame | None:
    """
    Read a .cna.seg file and return a tidy DataFrame with canonical column names.
    Handles ichorCNA's sample-name-prefixed columns automatically.
    """
    try:
        df = pd.read_csv(seg_path, sep="\t", comment="#", low_memory=False,
                         na_values=["NA", "NaN", "nan", "N/A", "."],
                         keep_default_na=True)
    except Exception as e:
        print(f"  [WARN] Could not read {seg_path}: {e}")
        return None

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Strip ichorCNA sample-name prefix (e.g. "B163_6_R1.copy.number" → "copy.number")
    df.columns = _strip_sample_prefix(list(df.columns))

    # Apply canonical rename map (case-insensitive)
    rename_map = {c: _SEG_RENAME[c.lower()]
                  for c in df.columns if c.lower() in _SEG_RENAME}
    df = df.rename(columns=rename_map)

    # Deduplicate columns by position (not name) to avoid pandas returning
    # a DataFrame instead of a Series when columns share a name
    seen: set[str] = set()
    keep_idx: list[int] = []
    for i, col in enumerate(df.columns):
        if col not in seen:
            seen.add(col)
            keep_idx.append(i)
    df = df.iloc[:, keep_idx]

    # Require chrom column
    if "chrom" not in df.columns:
        print(f"  [WARN] No 'chrom' column in {seg_path}. "
              f"Columns found: {list(df.columns)}")
        return None

    # Drop repeated header rows
    df = df[~df["chrom"].astype(str).str.lower().isin(
        ["chrom", "chr", "chromosome"])]

    # Standardize chromosome format
    df["chrom"] = (df["chrom"].astype(str).str.strip()
                   .apply(lambda c: c if c.startswith("chr") else f"chr{c}"))

    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("copy.number", "logr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If the file had an "ID" column it was already renamed to "sample"
    # by _SEG_RENAME.  Only overwrite if the column is missing or blank.
    if "sample" not in df.columns or df["sample"].isna().all():
        df["sample"] = sample
    df["seg_length"] = df["end"] - df["start"]
    return df


# ---------------------------------------------------------------------------
# Alteration calling
# ---------------------------------------------------------------------------

def call_alteration(row: "pd.Series") -> str | None:
    """
    Classify a segment as AMP, DEL, or NEUTRAL.
    Priority: call column → copy.number → log2 ratio.
    """
    # 1. Explicit call string
    if "call" in row.index and pd.notna(row["call"]):
        call = str(row["call"]).upper()
        if call in ("AMP", "HLAMP", "DUP", "GAIN", "+"):
            return "AMP"
        if call in ("DEL", "HOMDEL", "LOSS", "HLOSS", "-"):
            return "DEL"
        if call in ("NEUT", "NEUTRAL", "0", "DIPLOID"):
            return "NEUTRAL"

    # 2. Integer copy number
    if "copy.number" in row.index and pd.notna(row["copy.number"]):
        cn = float(row["copy.number"])
        if cn >= AMP_MIN_CN:
            return "AMP"
        if cn < DIPLOID_CN:
            return "DEL"
        return "NEUTRAL"

    # 3. Log2 ratio fallback
    for col in LOG2_COL_CANDIDATES:
        if col in row.index and pd.notna(row[col]):
            try:
                l2 = float(row[col])
                if l2 > 0.2:
                    return "AMP"
                if l2 < -0.2:
                    return "DEL"
                return "NEUTRAL"
            except (ValueError, TypeError):
                continue
    return None