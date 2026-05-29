"""
COSMIC Cancer Gene Census loader and focal-amplification annotator.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_cosmic_cgc(cgc_path: Path) -> pd.DataFrame:
    """
    Load a COSMIC Cancer Gene Census file (TSV or CSV, auto-detected).

    Returns a DataFrame with columns:
        gene_name, chrom, gene_start, gene_end,
        role, role_raw, tier, hallmark, chr_band, tumour_types, tissue_type

    Genome Location is expected in hg38 format: N:start-end (no 'chr' prefix).
    """
    # Auto-detect separator
    with open(cgc_path) as fh:
        first_line = fh.readline()
    sep = "\t" if first_line.count("\t") >= 4 else ","

    try:
        cgc = pd.read_csv(cgc_path, sep=sep, quotechar='"',
                          on_bad_lines="warn", low_memory=False)
    except Exception as e:
        raise ValueError(f"Could not read COSMIC CGC file '{cgc_path}': {e}") from e

    cgc.columns = [c.strip() for c in cgc.columns]
    col_lower   = {c.lower(): c for c in cgc.columns}

    def _col(*keys: str) -> str | None:
        for k in keys:
            if k in col_lower:
                return col_lower[k]
        return None

    sym_col      = _col("gene symbol", "gene_symbol", "symbol")
    loc_col      = _col("genome location", "genome_location", "location")
    role_col     = _col("role in cancer", "role_in_cancer", "role")
    tier_col     = _col("tier")
    hallmark_col = _col("hallmark")
    band_col     = _col("chr band", "chr_band", "cytoband")
    tumour_col   = _col("tumour types(somatic)", "tumour_types_somatic",
                        "tumour types", "tumor types")
    tissue_col   = _col("tissue type", "tissue_type")

    for label, col in [("Gene Symbol", sym_col), ("Genome Location", loc_col)]:
        if col is None:
            raise ValueError(
                f"COSMIC CGC missing required column '{label}'. "
                f"Found: {list(cgc.columns)}"
            )

    def _clean(val) -> str:
        if pd.isna(val):
            return ""
        return str(val).strip().strip("\"' ")

    def _normalize_role(raw: str) -> str:
        lower = raw.lower()
        parts = []
        if "oncogene" in lower:
            parts.append("oncogene")
        if "tsg" in lower or "tumour suppressor" in lower:
            parts.append("TSG")
        if "fusion" in lower:
            parts.append("fusion")
        return "/".join(parts) if parts else ("unknown" if raw else "")

    rows = []
    n_skipped = 0
    for _, r in cgc.iterrows():
        gene = _clean(r[sym_col])
        loc  = _clean(r[loc_col]).replace(",", "")
        m    = re.match(r"^(?:chr)?([0-9XYMTxymt]+):([0-9]+)-([0-9]+)$", loc)
        if not m:
            n_skipped += 1
            continue
        raw_role = _clean(r[role_col]) if role_col else ""
        rows.append({
            "gene_name":    gene,
            "chrom":        f"chr{m.group(1)}",
            "gene_start":   int(m.group(2)),
            "gene_end":     int(m.group(3)),
            "role_raw":     raw_role,
            "role":         _normalize_role(raw_role),
            "tier":         _clean(r[tier_col])     if tier_col     else "",
            "hallmark":     _clean(r[hallmark_col]) if hallmark_col else "",
            "chr_band":     _clean(r[band_col])     if band_col     else "",
            "tumour_types": _clean(r[tumour_col])   if tumour_col   else "",
            "tissue_type":  _clean(r[tissue_col])   if tissue_col   else "",
        })

    if not rows:
        raise ValueError(
            f"No valid coordinates parsed from '{cgc_path}'. "
            "Check Genome Location column format (N:start-end)."
        )

    result  = pd.DataFrame(rows)
    n_tier1 = (result["tier"] == "1").sum()
    print(f"      Loaded {len(result)} CGC genes from {cgc_path.name} "
          f"(Tier 1: {n_tier1}, skipped {n_skipped} rows).")
    return result


# Annotation columns and their CGC source fields
_ANN_COLS: list[tuple[str, str]] = [
    ("cosmic_genes",        "gene_name"),
    ("cosmic_roles",        "role"),
    ("cosmic_roles_raw",    "role_raw"),
    ("cosmic_tiers",        "tier"),
    ("cosmic_hallmarks",    "hallmark"),
    ("cosmic_chr_bands",    "chr_band"),
    ("cosmic_tumour_types", "tumour_types"),
    ("cosmic_tissue_types", "tissue_type"),
]


def annotate_with_cgc(focal_df: pd.DataFrame,
                       cgc_df:   pd.DataFrame | None) -> pd.DataFrame:
    """
    Annotate focal amplification events with overlapping COSMIC CGC genes.
    Adds columns: cosmic_genes, cosmic_roles, cosmic_roles_raw, cosmic_tiers,
                  cosmic_hallmarks, cosmic_chr_bands, cosmic_tumour_types,
                  cosmic_tissue_types.
    If cgc_df is None all columns are set to empty string.
    """
    focal_df = focal_df.copy()
    if cgc_df is None or cgc_df.empty:
        for out_col, _ in _ANN_COLS:
            focal_df[out_col] = ""
        return focal_df

    # Build per-chromosome lookup
    cgc_by_chrom: dict[str, list] = {}
    for _, gr in cgc_df.iterrows():
        entry = (int(gr["gene_start"]), int(gr["gene_end"])) + tuple(
            str(gr.get(src_col, "")) for _, src_col in _ANN_COLS
        )
        cgc_by_chrom.setdefault(gr["chrom"], []).append(entry)

    out_lists: list[list[str]] = [[] for _ in _ANN_COLS]

    for _, row in focal_df.iterrows():
        hits: list[list[str]] = [[] for _ in _ANN_COLS]
        for entry in cgc_by_chrom.get(row["chrom"], []):
            g_start, g_end = entry[0], entry[1]
            if row["start"] <= g_end and row["end"] >= g_start:
                for i, val in enumerate(entry[2:]):
                    hits[i].append(val)
        for i, hit_list in enumerate(hits):
            out_lists[i].append(",".join(hit_list))

    for i, (out_col, _) in enumerate(_ANN_COLS):
        focal_df[out_col] = out_lists[i]
    return focal_df
