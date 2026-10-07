"""
Streaming broad CNA frequency calculation for ichorCNA 1 Mbp output.
Processes one sample at a time to keep memory use constant.

Segments are mapped to fixed-size genomic bins during streaming rather than
stored by exact coordinates. This avoids the explosion of unique (start, end)
keys that occurs because different samples have different segment boundaries
even when covering the same genomic region.
"""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from ..common.constants import CHROM_LENGTHS, CHROM_ORDER
from ..common.io import parse_seg_file, call_alteration, LOG2_COL_CANDIDATES


# ---------------------------------------------------------------------------
# Low-TF variant: CN-call-only frequency calculation (no log2R filter)
# ---------------------------------------------------------------------------

def stream_broad_cna_frequencies_cn_only(
        seg_files:       dict[str, Path],
        lowtf_samples:   list[str],
        sample_entity:   dict[str, str] | None = None,
        bin_size:        int = 1_000_000,
        tf_df:           "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Compute per-bin AMP/DEL frequencies using ichorCNA copy-number calls only.

    Unlike stream_broad_cna_frequencies, no log2-ratio magnitude filter is
    applied. A bin is counted as AMP (or DEL) for a sample if ichorCNA's
    integer copy-number call (or explicit call string) for that segment is
    non-neutral. This is appropriate for low-TF samples (3–10%) where the
    log2R signal is small by definition and would suppress real events.

    When tf_df is provided, one representative sample per patient is used
    (highest TF in the supplied window), so frequencies are at patient level.

    Parameters
    ----------
    seg_files      : {normalized_sample_name: Path} for the 1 Mbp run.
    lowtf_samples  : list of sample names in the TF window.
    sample_entity  : optional {sample_name: entity_label} for per-entity counts.
    bin_size       : genomic bin size in bp (default 1 Mbp).
    tf_df          : tumor fraction table (columns: sample, patient,
                     tumor_fraction). When provided, one sample per patient
                     is selected (highest TF).

    Returns
    -------
    DataFrame with columns:
        entity, chrom, bin_start, bin_end, n_amp, n_del,
        n_total, freq_amp, freq_del
    """
    lowtf_set = set(lowtf_samples)

    if tf_df is not None and not tf_df.empty:
        _tf = tf_df[tf_df["sample"].isin(lowtf_set)].copy()
        _best = (
            _tf.sort_values("tumor_fraction", ascending=False)
               .drop_duplicates(subset=["patient"], keep="first")
        )
        representative_samples = set(_best["sample"].tolist())
        n_patients  = len(representative_samples)
        n_discarded = len(lowtf_set) - n_patients
        print(f"        Patient-level dedup: {n_patients} representative samples "
              f"(discarded {n_discarded} lower-TF timepoints).")
    else:
        representative_samples = lowtf_set
        n_patients  = len(representative_samples)

    n_total         = n_patients
    seg_files_lowtf = {s: p for s, p in seg_files.items()
                       if s in representative_samples}
    n               = len(seg_files_lowtf)

    if n == 0:
        print("        [WARN] No low-TF samples found in seg_files dict.")
        return pd.DataFrame()

    # Probe first sample
    _probe_sample, _probe_path = next(iter(seg_files_lowtf.items()))
    _probe_df = parse_seg_file(_probe_path, _probe_sample)
    if _probe_df is not None and not _probe_df.empty:
        has_call = "call" in _probe_df.columns
        has_cn   = "copy.number" in _probe_df.columns
        has_log2 = next((c for c in LOG2_COL_CANDIDATES
                         if c in _probe_df.columns), None)
        print(f"        Seg columns — call: {has_call}, "
              f"copy.number: {has_cn}, log2 col: {has_log2 or 'none'} "
              f"[CN-only mode, log2R filter disabled]")
        _probe_df["_alt"] = _probe_df.apply(call_alteration, axis=1)
        n_called = _probe_df["_alt"].notna().sum()
        print(f"        First-sample probe: {n_called}/{len(_probe_df)} "
              f"segments successfully called.")
        if n_called == 0:
            print("        [WARN] Zero segments called — check that seg files "
                  "contain a 'call' or 'copy.number' column.")
    del _probe_df
    gc.collect()

    use_entities = sample_entity is not None
    entities: set[str] = set()
    if use_entities:
        entities = {sample_entity.get(s, "Other") for s in seg_files_lowtf}

    amp_counts: dict[tuple, int] = {}
    del_counts: dict[tuple, int] = {}
    entity_amp: dict[str, dict[tuple, int]] = {e: {} for e in entities}
    entity_del: dict[str, dict[tuple, int]] = {e: {} for e in entities}

    entity_totals: dict[str, int] = {}
    if use_entities:
        for s in seg_files_lowtf:
            ent = sample_entity.get(s, "Other")
            entity_totals[ent] = entity_totals.get(ent, 0) + 1

    for i, (sample, seg_path) in enumerate(seg_files_lowtf.items(), 1):
        if i % 100 == 0 or i == n:
            print(f"        broad CNA (CN-only): {i}/{n} samples ...", flush=True)

        df = parse_seg_file(seg_path, sample)
        if df is None or df.empty:
            continue

        df["alteration"] = df.apply(call_alteration, axis=1)

        ent = sample_entity.get(sample, "Other") if use_entities else None

        sample_amp_bins: set[tuple] = set()
        sample_del_bins: set[tuple] = set()

        for _, row in df.iterrows():
            alt = row["alteration"]
            if alt is None or alt == "NEUTRAL":
                continue
            # No log2R magnitude filter — trust ichorCNA's CN call directly
            chrom = row["chrom"]
            if chrom not in CHROM_LENGTHS:
                continue
            bins = _seg_to_bins(chrom, int(row["start"]), int(row["end"]), bin_size)
            for bin_key in bins:
                if alt == "AMP":
                    if bin_key not in sample_amp_bins:
                        sample_amp_bins.add(bin_key)
                        amp_counts[bin_key] = amp_counts.get(bin_key, 0) + 1
                        if ent is not None:
                            entity_amp[ent][bin_key] = \
                                entity_amp[ent].get(bin_key, 0) + 1
                else:
                    if bin_key not in sample_del_bins:
                        sample_del_bins.add(bin_key)
                        del_counts[bin_key] = del_counts.get(bin_key, 0) + 1
                        if ent is not None:
                            entity_del[ent][bin_key] = \
                                entity_del[ent].get(bin_key, 0) + 1

        del df
        gc.collect()

    all_keys = set(amp_counts) | set(del_counts)
    if not all_keys:
        return pd.DataFrame()

    def _sorted_keys(keys):
        chrom_rank = {c: i for i, c in enumerate(CHROM_ORDER)}
        return sorted(keys, key=lambda k: (chrom_rank.get(k[0], 99), k[1]))

    rows = []

    for key in _sorted_keys(all_keys):
        chrom, bin_start = key
        bin_end = min(bin_start + bin_size, CHROM_LENGTHS.get(chrom, bin_start + bin_size))
        n_amp = amp_counts.get(key, 0)
        n_del = del_counts.get(key, 0)
        rows.append({
            "entity":    "All",
            "chrom":     chrom,
            "bin_start": bin_start,
            "bin_end":   bin_end,
            "n_amp":     n_amp,
            "n_del":     n_del,
            "n_total":   n_total,
            "freq_amp":  n_amp / n_total,
            "freq_del":  n_del / n_total,
        })

    for ent in sorted(entities):
        n_ent  = entity_totals.get(ent, 1)
        e_keys = set(entity_amp.get(ent, {})) | set(entity_del.get(ent, {}))
        for key in _sorted_keys(e_keys):
            chrom, bin_start = key
            bin_end = min(bin_start + bin_size,
                          CHROM_LENGTHS.get(chrom, bin_start + bin_size))
            n_amp = entity_amp[ent].get(key, 0)
            n_del = entity_del[ent].get(key, 0)
            rows.append({
                "entity":    ent,
                "chrom":     chrom,
                "bin_start": bin_start,
                "bin_end":   bin_end,
                "n_amp":     n_amp,
                "n_del":     n_del,
                "n_total":   n_ent,
                "freq_amp":  n_amp / n_ent,
                "freq_del":  n_del / n_ent,
            })

    return pd.DataFrame(rows)


def _seg_to_bins(chrom: str, start: int, end: int,
                 bin_size: int) -> list[tuple[str, int]]:
    """
    Return all bin keys (chrom, bin_start) that a segment overlaps.
    bin_start is always a multiple of bin_size.
    """
    chrom_len  = CHROM_LENGTHS.get(chrom, 0)
    first_bin  = (start // bin_size) * bin_size
    last_bin   = (min(end, chrom_len) // bin_size) * bin_size
    bins = []
    b = first_bin
    while b <= last_bin:
        bins.append((chrom, b))
        b += bin_size
    return bins


def stream_broad_cna_frequencies(
        seg_files:       dict[str, Path],
        high_tf_samples: list[str],
        sample_entity:   dict[str, str] | None = None,
        bin_size:        int = 1_000_000,
        tf_df:           "pd.DataFrame | None" = None,
        min_abs_log2r:   float = 0.2,
) -> pd.DataFrame:
    """
    Compute per-bin AMP/DEL frequencies across all high-TF samples, streaming
    one sample at a time.

    Each segment is mapped to all fixed-size bins it overlaps. A bin is
    counted as AMP (or DEL) for a sample if any of that sample's segments
    covering the bin carry that call AND have |log2R| > min_abs_log2r
    (default 0.2). This prevents very shallow gains/losses (often noise or
    tumour-fraction artefacts) from inflating frequencies.

    When tf_df is provided, only the highest-TF sample per patient is used,
    so frequencies are computed at the patient level rather than sample level.
    This prevents patients with many timepoints from being over-represented.

    Parameters
    ----------
    seg_files       : {normalized_sample_name: Path} for the 1 Mbp run.
    high_tf_samples : list of sample names that passed the TF threshold.
    sample_entity   : optional {sample_name: entity_label} mapping for
                      per-entity counts alongside the global "All" counts.
    bin_size        : genomic bin size in bp (default 1 Mbp).
    tf_df           : tumor fraction table (columns: sample, patient,
                      tumor_fraction). When provided, one representative
                      sample per patient is selected (highest TF).
    min_abs_log2r   : minimum absolute log2 ratio for a segment to be
                      counted (default 0.2). Segments with |log2R| below
                      this threshold are treated as NEUTRAL regardless of
                      their call label.

    Returns
    -------
    DataFrame with columns:
        entity, chrom, bin_start, bin_end, n_amp, n_del,
        n_total, freq_amp, freq_del
    """
    high_tf_set      = set(high_tf_samples)

    # Select one representative sample per patient (highest TF) when tf_df
    # is provided, so frequencies are at the patient level.
    if tf_df is not None and not tf_df.empty:
        _tf = tf_df[tf_df["sample"].isin(high_tf_set)].copy()
        # For each patient keep only the sample with the highest TF
        _best = (
            _tf.sort_values("tumor_fraction", ascending=False)
               .drop_duplicates(subset=["patient"], keep="first")
        )
        representative_samples = set(_best["sample"].tolist())
        n_patients  = len(representative_samples)
        n_discarded = len(high_tf_set) - n_patients
        print(f"        Patient-level dedup: {n_patients} representative samples "
              f"(discarded {n_discarded} lower-TF timepoints).")
    else:
        representative_samples = high_tf_set
        n_patients  = len(representative_samples)

    n_total          = n_patients
    seg_files_hightf = {s: p for s, p in seg_files.items()
                        if s in representative_samples}
    n                = len(seg_files_hightf)

    if n == 0:
        print("        [WARN] No high-TF samples found in seg_files dict.")
        return pd.DataFrame()

    # Probe first sample to report which columns drive alteration calling
    _probe_sample, _probe_path = next(iter(seg_files_hightf.items()))
    _probe_df = parse_seg_file(_probe_path, _probe_sample)
    if _probe_df is not None and not _probe_df.empty:
        has_call = "call" in _probe_df.columns
        has_cn   = "copy.number" in _probe_df.columns
        has_log2 = next((c for c in LOG2_COL_CANDIDATES
                         if c in _probe_df.columns), None)
        print(f"        Seg columns — call: {has_call}, "
              f"copy.number: {has_cn}, log2 col: {has_log2 or 'none'}")
        print(f"        All columns: {list(_probe_df.columns)}")
        _probe_df["_alt"] = _probe_df.apply(call_alteration, axis=1)
        n_called = _probe_df["_alt"].notna().sum()
        print(f"        First-sample probe: {n_called}/{len(_probe_df)} "
              f"segments successfully called.")
        if n_called == 0:
            print("        [WARN] Zero segments called — check that seg files "
                  "contain a 'call', 'copy.number', or log2-ratio column.")
    del _probe_df
    gc.collect()

    use_entities = sample_entity is not None
    entities: set[str] = set()
    if use_entities:
        entities = {sample_entity.get(s, "Other") for s in seg_files_hightf}

    # Bin-level counters: {(chrom, bin_start): count}
    amp_counts: dict[tuple, int] = {}
    del_counts: dict[tuple, int] = {}

    entity_amp: dict[str, dict[tuple, int]] = {e: {} for e in entities}
    entity_del: dict[str, dict[tuple, int]] = {e: {} for e in entities}

    entity_totals: dict[str, int] = {}
    if use_entities:
        for s in seg_files_hightf:
            ent = sample_entity.get(s, "Other")
            entity_totals[ent] = entity_totals.get(ent, 0) + 1

    for i, (sample, seg_path) in enumerate(seg_files_hightf.items(), 1):
        if i % 100 == 0 or i == n:
            print(f"        broad CNA: {i}/{n} samples ...", flush=True)

        df = parse_seg_file(seg_path, sample)
        if df is None or df.empty:
            continue

        df["alteration"] = df.apply(call_alteration, axis=1)

        # Apply |log2R| > min_abs_log2r filter.
        # Retrieve the log2 ratio column for this sample (same candidates
        # used by call_alteration as fallback).
        log2_col = next((c for c in LOG2_COL_CANDIDATES if c in df.columns), None)

        ent = sample_entity.get(sample, "Other") if use_entities else None

        # Track which bins this sample has already been counted in to avoid
        # double-counting when multiple segments of the same call overlap a bin
        sample_amp_bins: set[tuple] = set()
        sample_del_bins: set[tuple] = set()

        for _, row in df.iterrows():
            alt = row["alteration"]
            if alt is None or alt == "NEUTRAL":
                continue

            # Skip segments whose log2R magnitude is below the threshold
            if log2_col is not None and pd.notna(row.get(log2_col)):
                if abs(float(row[log2_col])) < min_abs_log2r:
                    continue
            chrom = row["chrom"]
            if chrom not in CHROM_LENGTHS:
                continue
            bins = _seg_to_bins(chrom, int(row["start"]), int(row["end"]), bin_size)
            for bin_key in bins:
                if alt == "AMP":
                    if bin_key not in sample_amp_bins:
                        sample_amp_bins.add(bin_key)
                        amp_counts[bin_key] = amp_counts.get(bin_key, 0) + 1
                        if ent is not None:
                            entity_amp[ent][bin_key] = \
                                entity_amp[ent].get(bin_key, 0) + 1
                else:
                    if bin_key not in sample_del_bins:
                        sample_del_bins.add(bin_key)
                        del_counts[bin_key] = del_counts.get(bin_key, 0) + 1
                        if ent is not None:
                            entity_del[ent][bin_key] = \
                                entity_del[ent].get(bin_key, 0) + 1

        del df
        gc.collect()

    all_keys = set(amp_counts) | set(del_counts)
    if not all_keys:
        return pd.DataFrame()

    # Build output in chromosome order
    def _sorted_keys(keys):
        chrom_rank = {c: i for i, c in enumerate(CHROM_ORDER)}
        return sorted(keys, key=lambda k: (chrom_rank.get(k[0], 99), k[1]))

    rows = []

    # Global "All" rows
    for key in _sorted_keys(all_keys):
        chrom, bin_start = key
        bin_end = min(bin_start + bin_size, CHROM_LENGTHS.get(chrom, bin_start + bin_size))
        n_amp = amp_counts.get(key, 0)
        n_del = del_counts.get(key, 0)
        rows.append({
            "entity":    "All",
            "chrom":     chrom,
            "bin_start": bin_start,
            "bin_end":   bin_end,
            "n_amp":     n_amp,
            "n_del":     n_del,
            "n_total":   n_total,
            "freq_amp":  n_amp / n_total,
            "freq_del":  n_del / n_total,
        })

    # Per-entity rows
    for ent in sorted(entities):
        n_ent  = entity_totals.get(ent, 1)
        e_keys = set(entity_amp.get(ent, {})) | set(entity_del.get(ent, {}))
        for key in _sorted_keys(e_keys):
            chrom, bin_start = key
            bin_end = min(bin_start + bin_size,
                          CHROM_LENGTHS.get(chrom, bin_start + bin_size))
            n_amp = entity_amp[ent].get(key, 0)
            n_del = entity_del[ent].get(key, 0)
            rows.append({
                "entity":    ent,
                "chrom":     chrom,
                "bin_start": bin_start,
                "bin_end":   bin_end,
                "n_amp":     n_amp,
                "n_del":     n_del,
                "n_total":   n_ent,
                "freq_amp":  n_amp / n_ent,
                "freq_del":  n_del / n_ent,
            })

    return pd.DataFrame(rows)