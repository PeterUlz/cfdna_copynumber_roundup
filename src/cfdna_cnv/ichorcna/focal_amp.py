"""
Focal amplification detection for ichorCNA 50 kbp output.
Uses streaming to keep peak memory at ~1 sample's worth of segments.
"""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.constants import FOCAL_MIN_BP, FOCAL_MAX_BP, AMP_MIN_CN
from ..common.io import parse_seg_file, call_alteration
from ..common.cosmic import annotate_with_cgc


# ---------------------------------------------------------------------------
# Per-sample focal amp detection
# ---------------------------------------------------------------------------

def _get_log2_col(df: pd.DataFrame) -> str | None:
    """Return the first log2-ratio column found in df."""
    candidates = ["logr", "log2_ratio", "log2ratio", "seg.mean",
                  "median", "logr_copy_ratio"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "log" in cl and ("ratio" in cl or cl.endswith("r")):
            return c
    return None


def _is_amp_call(call_str: str) -> bool:
    """Return True only for high-confidence amplification calls (not GAIN/CN=3)."""
    c = str(call_str).upper().strip()
    return c in ("AMP", "HLAMP")


def detect_focal_amplifications(
        seg_df:       pd.DataFrame,
        segs_1mb:     pd.DataFrame | None = None,
        min_bp:       int   = FOCAL_MIN_BP,
        max_bp:       int   = FOCAL_MAX_BP,
        min_cn:       int   = 4,
        min_bins:     int   = 3,
        log2_delta:   float = 0.2,
        flank_bins:   int   = 2,
        noise_factor: float = 2.0,
        max_focal_per_sample: int = 5,
) -> pd.DataFrame:
    """
    Detect focal amplifications in a single sample's 50 kbp segment DataFrame.

    Filters applied in order:
      1. Segment length between min_bp and max_bp.
      2. copy.number >= min_cn  (default 4 — excludes GAIN/CN=3).
         The call column must be AMP or HLAMP (not GAIN).
      3. num.mark >= min_bins  (default 3, i.e. ≥ 3 × 50 kbp = 150 kbp).
      4. Log2 ratio elevation >= log2_delta above the median of flanking
         1 Mbp bins (= flank_bins × 50 kbp = 20 × 50 kbp = 1 Mbp).
         One-sided flanking is used when a segment is near a chromosome end
         so that end-of-arm segments are not automatically discarded.
      5. Samples with more than max_focal_per_sample events are discarded
         entirely (likely noisy or aneuploid with widespread instability).

    Parameters
    ----------
    seg_df     : segmented output for one sample (parse_seg_file result).
    segs_1mb   : 1 Mbp segments for the same sample (background calculation).
    min_bp     : minimum segment length in bp.
    max_bp     : maximum segment length in bp.
    min_cn     : minimum integer copy number (default 4).
    min_bins   : minimum number of 50 kbp bins in the segment (default 3).
    log2_delta : minimum log2 elevation above local background.
    flank_bins : number of 1 Mbp windows on each side for background.
    noise_factor : segment logR must be >= noise_factor × sample_noise
                   (default 2.0). Sample noise = median absolute difference
                   between consecutive 50 kbp bins on the same chromosome.
    max_focal_per_sample : discard sample if more events pass all filters.
    """
    df = seg_df.copy()

    # ── Filter 1 + 2: size and CN/call ──────────────────────────────────────
    # Require CN >= min_cn AND call must be AMP or HLAMP (not GAIN)
    has_call = "call" in df.columns
    has_cn   = "copy.number" in df.columns

    def _passes_cn_call(row) -> bool:
        # Call check: must be AMP or HLAMP if call column is available
        if has_call and pd.notna(row["call"]):
            if not _is_amp_call(row["call"]):
                return False
        # CN check
        if has_cn and pd.notna(row["copy.number"]):
            return float(row["copy.number"]) >= min_cn
        # Fallback: logr > 1.0 (≈ CN=4 at TF=1)
        logr_col = _get_log2_col(df)
        if logr_col and pd.notna(row.get(logr_col)):
            return float(row[logr_col]) >= 1.0
        return False

    candidates = df[
        (df["seg_length"] >= min_bp) &
        (df["seg_length"] <= max_bp)
    ].copy()
    candidates = candidates[candidates.apply(_passes_cn_call, axis=1)]

    if candidates.empty:
        return candidates.reset_index(drop=True)

    # ── Filter 3: minimum bin count ──────────────────────────────────────────
    if "num.mark" in candidates.columns:
        before = len(candidates)
        candidates = candidates[
            pd.to_numeric(candidates["num.mark"], errors="coerce") >= min_bins
        ]
        removed = before - len(candidates)
        if removed:
            print(f"        Bin-count filter removed {removed} candidates "
                  f"(num.mark < {min_bins}).")
    else:
        # Approximate from segment length / 50 kbp
        before = len(candidates)
        candidates = candidates[
            (candidates["seg_length"] / 50_000) >= min_bins
        ]
        removed = before - len(candidates)
        if removed:
            print(f"        Bin-count filter (approx) removed {removed} "
                  f"candidates.")

    if candidates.empty:
        return candidates.reset_index(drop=True)

    # ── Filters 4 + 4b: background elevation and neighbour noise ────────────
    # Background is computed from flanking 50 kbp bins (same resolution as
    # the candidates), using a length-weighted median of all bins within
    # flank_bins * 1 Mbp on either side that don't overlap the segment.
    # One-sided flanking is used at chromosome ends.
    #
    # Filter 4b: segment logR must be >= noise_factor * sample_noise, where
    # sample_noise = genome-wide median absolute difference between
    # consecutive bins on the same chromosome.  This ensures the focal amp
    # is genuinely elevated above the technical noise floor of the sample.
    log2_col_50k = _get_log2_col(df)

    # Compute sample noise once (used in filter 4b below)
    sample_noise: float | None = None
    if log2_col_50k is not None:
        diffs = []
        for chrom, grp in df.groupby("chrom"):
            vals  = pd.to_numeric(grp[log2_col_50k], errors="coerce").dropna().values
            if len(vals) >= 2:
                diffs.append(np.abs(np.diff(vals)))
        if diffs:
            sample_noise = float(np.median(np.concatenate(diffs)))
            print(f"        Sample neighbour noise: {sample_noise:.4f}")

    if log2_col_50k is None:
        print(f"        [INFO] No log2 column in 50 kbp data — background filter skipped.")
        candidates["bg_log2"] = np.nan
        candidates["delta_log2"] = np.nan
    else:
        flank_bp = flank_bins * 1_000_000

        # Build per-chrom index of all 50 kbp bins: (start, end, logr)
        bins_by_chrom: dict[str, list] = {}
        for _, r in df.iterrows():
            lv = r.get(log2_col_50k, np.nan)
            if pd.isna(lv):
                continue
            bins_by_chrom.setdefault(r["chrom"], []).append(
                (int(r["start"]), int(r["end"]), float(lv))
            )
        for k in bins_by_chrom:
            bins_by_chrom[k].sort()

        def _weighted_median(vals: list, wts: list) -> float:
            """Weighted median via sorted cumulative weight."""
            pairs = sorted(zip(vals, wts), key=lambda x: x[0])
            total = sum(w for _, w in pairs)
            cumsum = 0.0
            for v, w in pairs:
                cumsum += w
                if cumsum >= total / 2:
                    return float(v)
            return float(pairs[-1][0])

        def _background(chrom: str, seg_start: int, seg_end: int) -> float | None:
            """
            Length-weighted median logR of 50 kbp bins within flank_bp of
            the focal segment that do not overlap it.
            Bins on both sides are pooled; one-sided if at chromosome end.
            """
            upstream:   list[tuple] = []   # (logr, bin_length)
            downstream: list[tuple] = []

            for s, e, lv in bins_by_chrom.get(chrom, []):
                overlap = min(e, seg_end) - max(s, seg_start)
                if overlap > 0:
                    continue  # overlaps focal segment — skip
                bin_len = e - s
                if e <= seg_start and seg_start - e <= flank_bp:
                    upstream.append((lv, bin_len))
                elif s >= seg_end and s - seg_end <= flank_bp:
                    downstream.append((lv, bin_len))

            combined = upstream + downstream
            if not combined:
                return None
            vals, wts = zip(*combined)
            return _weighted_median(list(vals), list(wts))

        keep, bg_out, delta_out = [], [], []
        for _, row in candidates.iterrows():
            seg_l2 = row[log2_col_50k]
            if pd.isna(seg_l2):
                keep.append(True)
                bg_out.append(np.nan)
                delta_out.append(np.nan)
                continue
            bg = _background(row["chrom"], int(row["start"]), int(row["end"]))
            if bg is None:
                # No flanking bins — apply noise filter only
                if sample_noise is not None and sample_noise > 0:
                    passes_noise = float(seg_l2) >= noise_factor * sample_noise
                else:
                    passes_noise = True
                keep.append(passes_noise)
                bg_out.append(np.nan)
                delta_out.append(np.nan)
            else:
                delta = float(seg_l2) - bg
                # Filter 4: absolute AND relative elevation above background
                passes = (float(seg_l2) >= log2_delta) and (delta >= log2_delta)
                # Filter 4b: segment logR >= noise_factor * sample_noise
                if passes and sample_noise is not None and sample_noise > 0:
                    passes = float(seg_l2) >= noise_factor * sample_noise
                keep.append(passes)
                bg_out.append(round(bg, 4))
                delta_out.append(round(delta, 4))

        n_removed = sum(not k for k in keep)
        if n_removed:
            print(f"        Elevation/noise filter removed {n_removed} candidates.")

        candidates = candidates[keep].copy()
        candidates["bg_log2"]    = [v for v, k in zip(bg_out,   keep) if k]
        candidates["delta_log2"] = [v for v, k in zip(delta_out, keep) if k]

    # ── Filter 5: per-sample event count cap ────────────────────────────────
    n_events = len(candidates)
    if n_events > max_focal_per_sample:
        print(f"        Sample has {n_events} focal amps > max "
              f"{max_focal_per_sample} — discarding all events for this sample.")
        return pd.DataFrame()

    return candidates.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streaming over the full cohort
# ---------------------------------------------------------------------------

def stream_focal_amplifications(
        seg_files_50k:        dict[str, Path],
        high_tf_samples:      list[str],
        tmp_dir:              Path,
        min_bp:               int   = FOCAL_MIN_BP,
        max_bp:               int   = FOCAL_MAX_BP,
        min_cn:               int   = 4,
        min_bins:             int   = 3,
        log2_delta:           float = 0.2,
        flank_bins:           int   = 2,
        noise_factor:         float = 2.0,
        max_focal_per_sample: int   = 5,
        cgc_df:               "pd.DataFrame | None" = None,
        # kept for backwards compatibility — no longer used
        seg_files_1mb:        "dict[str, Path] | None" = None,
) -> Path:
    """
    Detect focal amplifications sample by sample, appending events to a temp
    TSV. Peak RAM stays at ~1 sample's worth of segments.
    Returns the path to the temp TSV.

    Background is computed from flanking 50 kbp bins (same resolution as
    candidates) — seg_files_1mb is no longer needed and is ignored.
    """
    high_tf_set      = set(high_tf_samples)
    seg_files_50k_ht = {s: p for s, p in seg_files_50k.items() if s in high_tf_set}
    n                = len(seg_files_50k_ht)
    out_path         = tmp_dir / "focal_amp_events.tsv"
    header_written   = False
    n_total_events   = 0

    for i, (sample, seg_path_50k) in enumerate(seg_files_50k_ht.items(), 1):
        if i % 100 == 0 or i == n:
            print(f"        focal amps: {i}/{n} samples …", flush=True)

        df_50k = parse_seg_file(seg_path_50k, sample)
        if df_50k is None or df_50k.empty:
            continue

        focal = detect_focal_amplifications(
            df_50k,
            segs_1mb             = None,
            min_bp               = min_bp,
            max_bp               = max_bp,
            min_cn               = min_cn,
            min_bins             = min_bins,
            log2_delta           = log2_delta,
            flank_bins           = flank_bins,
            noise_factor         = noise_factor,
            max_focal_per_sample = max_focal_per_sample,
        )
        if focal.empty:
            del df_50k
            gc.collect()
            continue

        focal = annotate_with_cgc(focal, cgc_df)
        n_total_events += len(focal)

        focal.to_csv(
            out_path, sep="\t", index=False,
            mode="a" if header_written else "w",
            header=not header_written,
        )
        header_written = True
        del df_50k, focal
        gc.collect()

    print(f"        Total focal amp events: {n_total_events}")
    return out_path

# ---------------------------------------------------------------------------
# Frequency aggregation — coverage-based, 50 kbp bins, patient-level
# ---------------------------------------------------------------------------

_COSMIC_COLS = (
    "cosmic_genes", "cosmic_roles", "cosmic_roles_raw", "cosmic_tiers",
    "cosmic_hallmarks", "cosmic_chr_bands", "cosmic_tumour_types",
    "cosmic_tissue_types",
)

_BIN_SIZE = 50_000   # matches the ichorCNA 50 kbp resolution


def aggregate_focal_frequencies(
        focal_tmp_path:  Path,
        high_tf_samples: list[str],
        tf_df:           "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Compute per-50-kbp-bin focal amplification frequencies at the patient level,
    then merge consecutive bins with identical patient sets into regions.

    Post-processing
    ---------------
    Adjacent 50 kbp bins on the same chromosome with the same patient set are
    merged into a single region row.  This collapses flat plateaus in the
    frequency track into one row per contiguous focal amp region, making the
    table much more readable and directly showing each MCR as one entry.

    Per-entity frequencies
    ----------------------
    When tf_df contains an 'entity' column (or one can be derived from patient
    IDs via entity_from_sample), per-entity n_patients and frequency columns
    are added: n_patients_<Entity> and freq_<Entity>.

    Parameters
    ----------
    focal_tmp_path  : path to the streaming focal amp TSV.
    high_tf_samples : full list of high-TF samples (denominator).
    tf_df           : tumor fraction table (sample, patient, tumor_fraction,
                      optionally entity).

    Returns
    -------
    DataFrame with one row per merged focal amp region, sorted by chromosome
    then start position. Columns: chrom, bin_start, bin_end, n_bins,
    n_patients, n_total_patients, frequency, [n_patients_X, freq_X per entity],
    mean_cn, max_cn, patients, cosmic_genes, …
    """
    if not focal_tmp_path.exists() or focal_tmp_path.stat().st_size == 0:
        return pd.DataFrame()

    from ..common.constants import CHROM_ORDER
    from ..ichorcna.plots import entity_from_sample

    # Build patient/entity mappings
    high_tf_set = set(high_tf_samples)
    if tf_df is not None and not tf_df.empty:
        _tf = tf_df[tf_df["sample"].isin(high_tf_set)].copy()
        rep_samples: set[str] = set(
            _tf.sort_values("tumor_fraction", ascending=False)
               .drop_duplicates(subset=["patient"], keep="first")
               ["sample"]
        )
        sample_to_patient: dict[str, str] = dict(zip(_tf["sample"], _tf["patient"]))
        n_total_patients = _tf["patient"].nunique()

        # Build patient → entity and entity → patient set
        if "entity" in _tf.columns:
            patient_entity: dict[str, str] = dict(zip(_tf["patient"], _tf["entity"]))
        else:
            patient_entity = {p: entity_from_sample(str(p))
                              for p in _tf["patient"].unique()}
        entity_all_patients: dict[str, set] = {}
        for pat, ent in patient_entity.items():
            entity_all_patients.setdefault(ent, set()).add(pat)
    else:
        rep_samples         = high_tf_set
        sample_to_patient   = {s: s for s in high_tf_set}
        n_total_patients    = len(high_tf_set)
        patient_entity      = {}
        entity_all_patients = {}

    all_entities = sorted(entity_all_patients.keys())

    # Accumulate per-bin data
    bin_patients: dict[tuple, set]  = {}
    bin_cn_vals:  dict[tuple, list] = {}
    bin_cosmic:   dict[tuple, dict] = {}

    for chunk in pd.read_csv(focal_tmp_path, sep="\t",
                             chunksize=50_000, low_memory=False):
        for _, row in chunk.iterrows():
            sample = str(row["sample"])
            if sample not in rep_samples:
                continue
            patient   = sample_to_patient.get(sample, sample)
            chrom     = str(row["chrom"])
            seg_start = int(row["start"])
            seg_end   = int(row["end"])
            first_bin = (seg_start // _BIN_SIZE) * _BIN_SIZE
            last_bin  = ((seg_end - 1) // _BIN_SIZE) * _BIN_SIZE
            cn = row.get("copy.number", float("nan"))
            try:
                cn = float(cn)
            except (ValueError, TypeError):
                cn = float("nan")
            b = first_bin
            while b <= last_bin:
                key = (chrom, b)
                bin_patients.setdefault(key, set()).add(patient)
                if not pd.isna(cn):
                    bin_cn_vals.setdefault(key, []).append(cn)
                if key not in bin_cosmic:
                    bin_cosmic[key] = {c: set() for c in _COSMIC_COLS}
                for c in _COSMIC_COLS:
                    val = str(row.get(c, ""))
                    for part in val.split(","):
                        part = part.strip()
                        if part and part != "nan":
                            bin_cosmic[key][c].add(part)
                b += _BIN_SIZE

    if not bin_patients:
        return pd.DataFrame()

    # Sort bins in genomic order
    chrom_rank = {c: i for i, c in enumerate(CHROM_ORDER)}
    sorted_keys = sorted(bin_patients,
                         key=lambda k: (chrom_rank.get(k[0], 99), k[1]))

    # ── Merge consecutive bins with identical patient sets ───────────────────
    merged_rows = []
    cur_chrom     = None
    cur_start     = None
    cur_end       = None
    cur_patients  = None
    cur_cn_vals   = []
    cur_cosmic    = {c: set() for c in _COSMIC_COLS}
    cur_n_bins    = 0

    def _flush(chrom, start, end, patients, cn_vals, cosmic, n_bins):
        n_pat   = len(patients)
        cn_list = cn_vals
        row_out = {
            "chrom":            chrom,
            "bin_start":        start,
            "bin_end":          end,
            "n_bins":           n_bins,
            "n_patients":       n_pat,
            "n_total_patients": n_total_patients,
            "frequency":        n_pat / n_total_patients if n_total_patients else 0,
            "mean_cn":          float(np.mean(cn_list)) if cn_list else float("nan"),
            "max_cn":           float(np.max(cn_list))  if cn_list else float("nan"),
            "patients":         ",".join(sorted(patients)),
        }
        # Per-entity frequencies
        for ent in all_entities:
            ent_total = len(entity_all_patients.get(ent, set()))
            ent_n     = sum(1 for p in patients
                            if patient_entity.get(p) == ent)
            row_out[f"n_patients_{ent}"] = ent_n
            row_out[f"freq_{ent}"]       = (ent_n / ent_total
                                             if ent_total > 0 else 0.0)
        for c in _COSMIC_COLS:
            row_out[c] = ",".join(sorted(cosmic.get(c, set())))
        merged_rows.append(row_out)

    for key in sorted_keys:
        chrom, b_start = key
        b_end      = b_start + _BIN_SIZE
        patients   = bin_patients[key]
        cn_vals    = bin_cn_vals.get(key, [])
        cosmic     = bin_cosmic.get(key, {c: set() for c in _COSMIC_COLS})

        if (cur_chrom == chrom and
                cur_end == b_start and       # adjacent bin
                cur_patients == patients):   # identical patient set
            # Extend current region
            cur_end     = b_end
            cur_cn_vals.extend(cn_vals)
            for c in _COSMIC_COLS:
                cur_cosmic[c].update(cosmic.get(c, set()))
            cur_n_bins += 1
        else:
            # Flush previous region and start new one
            if cur_chrom is not None:
                _flush(cur_chrom, cur_start, cur_end,
                       cur_patients, cur_cn_vals, cur_cosmic, cur_n_bins)
            cur_chrom    = chrom
            cur_start    = b_start
            cur_end      = b_end
            cur_patients = patients
            cur_cn_vals  = list(cn_vals)
            cur_cosmic   = {c: set(cosmic.get(c, set())) for c in _COSMIC_COLS}
            cur_n_bins   = 1

    if cur_chrom is not None:
        _flush(cur_chrom, cur_start, cur_end,
               cur_patients, cur_cn_vals, cur_cosmic, cur_n_bins)

    return (pd.DataFrame(merged_rows)
            .sort_values(["chrom", "bin_start"],
                         key=lambda col: col.map(chrom_rank)
                         if col.name == "chrom" else col)
            .reset_index(drop=True))