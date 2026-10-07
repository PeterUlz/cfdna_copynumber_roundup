"""
ichorCNA workflow: all 7 steps orchestrated in one place.
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from ..common.cosmic import load_cosmic_cgc
from ..common.io import (find_sample_files, collect_tumor_fractions,
                          parse_wig_total_reads, filter_seg_files_by_tf_df)
from ..locate.qc import find_incomplete_runs, validate_all_seg_files
from ..locate.longitudinal import emergent_focal_amps
from ..locate.focal_amp_plots import plot_all_focal_amp_diagnostics
from ..locate.longitudinal_plots import (
    plot_all_pairwise_longitudinal,
    plot_longitudinal_combined,
    plot_all_focal_amp_trajectories,
    compute_scaled_tf_estimates,
    plot_focal_amp_cohort_stability,
    plot_focal_amp_stability_focused,
)
from ..locate.plots import (
    plot_broad_cna_frequencies,
    plot_broad_cna_frequencies_lowtf,
    plot_focal_amp_frequencies as plot_focal_amp_ichorcna,
    plot_emergent_focal_amps,
)
from ..locate.broad_cna import (stream_broad_cna_frequencies,
                                   stream_broad_cna_frequencies_cn_only)
from ..locate.focal_amp import (
    stream_focal_amplifications,
    aggregate_focal_frequencies,
)
from ..locate.plots import entity_from_sample


def _safe_read(p: Path, **kwargs) -> pd.DataFrame | None:
    """Read a TSV only if it exists and has at least one data row."""
    if not p.exists():
        return None
    with open(p) as fh:
        lines = [ln for ln in fh if ln.strip()]
    if len(lines) < 2:
        return None
    try:
        return pd.read_csv(p, sep="\t", low_memory=False, **kwargs)
    except Exception:
        return None


def run(args: argparse.Namespace) -> None:
    dir_1mb = Path(args.dir_1mb)
    dir_50k = Path(args.dir_50kb)
    outdir  = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    for d, label in [(dir_1mb, "1 Mbp"), (dir_50k, "50 kbp")]:
        if not d.is_dir():
            print(f"[ERROR] {label} directory not found: {d}")
            sys.exit(1)

    # Load COSMIC CGC
    cgc_df = None
    if args.cosmic_cgc:
        cgc_path = Path(args.cosmic_cgc)
        if not cgc_path.is_file():
            print(f"[ERROR] COSMIC CGC file not found: {cgc_path}")
            sys.exit(1)
        try:
            cgc_df = load_cosmic_cgc(cgc_path)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    else:
        print("[INFO] No --cosmic-cgc provided; gene annotation skipped.")

    print("=" * 70)
    print("ichorCNA Analysis Pipeline")
    print("=" * 70)

    # ── 0. Incomplete runs ───────────────────────────────────────────────────
    print(f"\n[0/7] Checking for incomplete runs …")
    incomplete_rows = []
    for d, label in [(dir_1mb, "1Mbp"), (dir_50k, "50kbp")]:
        inc = find_incomplete_runs(d)
        if not inc.empty:
            inc.insert(0, "resolution", label)
            incomplete_rows.append(inc)
            print(f"      {label}: {len(inc)} incomplete sample(s) found.")
        else:
            print(f"      {label}: all sample directories look complete.")
    if incomplete_rows:
        inc_df  = pd.concat(incomplete_rows, ignore_index=True)
        inc_out = outdir / "incomplete_runs.tsv"
        inc_df.to_csv(inc_out, sep="\t", index=False)
        print(f"      Written: {inc_out}")
        for _, row in inc_df.iterrows():
            print(f"        [{row['resolution']}]  {row['sample']:30s}  {row['status']}")

    # ── 1. Tumor fractions ───────────────────────────────────────────────────
    print(f"\n[1/7] Collecting tumor fractions from 1 Mbp run ({dir_1mb}) …")
    params_1mb = find_sample_files(dir_1mb, ".params.txt", keep_duplicates=True)
    print(f"      Found {len(params_1mb)} samples via .params.txt files.")

    # Find WIG files for read-count QC and duplicate resolution
    wig_files = find_sample_files(dir_1mb, ".wig", keep_duplicates=True)
    if wig_files:
        print(f"      Found {len(wig_files)} WIG files for read-count QC.")
    else:
        print("      No WIG files found — total_reads will be NA.")

    tf_df = collect_tumor_fractions(params_1mb, wig_files=wig_files or None)
    tf_out = outdir / "tf_summary.tsv"
    tf_df.to_csv(tf_out, sep="\t", index=False)
    print(f"      Written: {tf_out}")

    # ── 2. TF filter ─────────────────────────────────────────────────────────
    threshold  = args.tf_threshold
    min_reads  = args.min_reads
    print(f"\n[2/7] Filtering at TF >= {threshold*100:.1f}%"
          f"  and  reads >= {min_reads:,} …")

    has_reads = "total_reads" in tf_df.columns and tf_df["total_reads"].notna().any()

    # Apply read-depth filter first
    if has_reads and min_reads > 0:
        # Only exclude samples where reads are known AND below threshold.
        # Samples with NA reads (no WIG file found) are kept.
        low_reads_mask  = (tf_df["total_reads"].notna() &
                           (tf_df["total_reads"] < min_reads))
        n_low_reads     = int(low_reads_mask.sum())
        if n_low_reads:
            print(f"      Samples EXCLUDED (< {min_reads:,} reads): {n_low_reads}")
        tf_df_filtered = tf_df[~low_reads_mask]
    else:
        tf_df_filtered = tf_df

    high_tf     = tf_df_filtered[tf_df_filtered["tumor_fraction"] >= threshold]
    low_tf      = tf_df_filtered[tf_df_filtered["tumor_fraction"] <  threshold]
    high_tf_samples = high_tf["sample"].tolist()
    print(f"      Samples PASSING: {len(high_tf_samples)}")
    print(f"      Samples EXCLUDED (low TF): {len(low_tf)}")
    if not high_tf_samples:
        print("[WARN] No samples pass the TF threshold — nothing to analyze.")
        sys.exit(0)

    # ── 3. Collect seg files ─────────────────────────────────────────────────
    print(f"\n[3/7] Collecting segment files …")
    # Search for seg files in priority order: .seg.txt (ichorCNA segmented
    # output), then .cna.seg, then bare .seg as a last fallback.
    def _find_seg(directory: Path, prefer_segmented: bool = True) -> dict:
        """
        Find segment files in priority order.
        prefer_segmented=True  → .seg.txt first (segmented, for broad CNA)
        prefer_segmented=False → .cna.seg first (bin-level, for longitudinal)
        """
        if prefer_segmented:
            extensions = (".seg.txt", ".cna.seg", ".seg")
        else:
            extensions = (".cna.seg", ".seg.txt", ".seg")
        for ext in extensions:
            found = find_sample_files(directory, ext, keep_duplicates=True)
            if found:
                print(f"        Using extension '{ext}' ({len(found)} files found).")
                return found
        return {}

    seg_1mb_files = _find_seg(dir_1mb, prefer_segmented=True)
    print(f"      Found {len(seg_1mb_files)} segment files (1 Mbp).")

    # For 50 kbp: find BOTH the segmented (.seg.txt) and bin-level (.cna.seg)
    # files — they are used for different purposes.
    seg_50k_files = _find_seg(dir_50k, prefer_segmented=True)
    print(f"      Found {len(seg_50k_files)} segment files (50 kbp, segmented).")

    seg_50k_bins = _find_seg(dir_50k, prefer_segmented=False)
    print(f"      Found {len(seg_50k_bins)} segment files (50 kbp, bin-level .cna.seg).")

    # Filter seg files to only samples that survived coverage deduplication
    seg_1mb_files = filter_seg_files_by_tf_df(seg_1mb_files, tf_df)
    seg_50k_files = filter_seg_files_by_tf_df(seg_50k_files, tf_df)
    seg_50k_bins  = filter_seg_files_by_tf_df(seg_50k_bins,  tf_df)

    # ── Entity exclusion ─────────────────────────────────────────────────────
    excluded_entities = getattr(args, "exclude_entities", [])
    if excluded_entities:
        tf_df["_entity"] = tf_df["patient"].apply(
            lambda p: entity_from_sample(str(p))
        )
        mask_excl = tf_df["_entity"].isin(excluded_entities)
        n_excl    = int(mask_excl.sum())
        if n_excl:
            excl_counts = tf_df[mask_excl]["_entity"].value_counts().to_dict()
            print(f"      Excluding entities: {excluded_entities} "
                  f"→ removing {n_excl} samples {excl_counts}")
            tf_df         = tf_df[~mask_excl].drop(columns=["_entity"])
            high_tf_samples = [s for s in high_tf_samples
                               if s in set(tf_df["sample"])]
            seg_1mb_files = filter_seg_files_by_tf_df(seg_1mb_files, tf_df)
            seg_50k_files = filter_seg_files_by_tf_df(seg_50k_files, tf_df)
            seg_50k_bins  = filter_seg_files_by_tf_df(seg_50k_bins,  tf_df)
        else:
            tf_df = tf_df.drop(columns=["_entity"])
            print(f"      --exclude-entities {excluded_entities}: "
                  f"no matching samples found.")

    seg_1mb_hightf = {s: p for s, p in seg_1mb_files.items()
                      if s in set(high_tf_samples)}
    seg_50k_hightf = {s: p for s, p in seg_50k_files.items()
                      if s in set(high_tf_samples)}
    seg_50k_bins_hightf = {s: p for s, p in seg_50k_bins.items()
                           if s in set(high_tf_samples)}

    # ── 4. Validation ────────────────────────────────────────────────────────
    print(f"\n[4/7] Validating segment files (high-TF samples only) …")
    val_frames: list[pd.DataFrame] = []
    val_out    = outdir / "seg_validation.tsv"
    val_df     = pd.DataFrame()

    if args.skip_validation:
        print("      Skipped (--skip-validation).")
    else:
        if seg_1mb_hightf:
            val_1mb = validate_all_seg_files(
                seg_1mb_hightf, label="1Mbp", out_path=val_out)
            val_frames.append(val_1mb)
        if seg_50k_hightf:
            val_50k = validate_all_seg_files(
                seg_50k_hightf, label="50kbp", out_path=val_out)
            val_frames.append(val_50k)
        if val_frames:
            val_df = (pd.read_csv(val_out, sep="\t", low_memory=False)
                      if val_out.exists()
                      else pd.concat(val_frames, ignore_index=True))
            print(f"      Written: {val_out}")
            flagged = val_df[val_df["status"] != "OK"]
            if not flagged.empty:
                print(f"      Flagged samples ({len(flagged)}):")
                for _, row in flagged.iterrows():
                    print(f"        [{row['resolution']}] {row['sample']:30s}  "
                          f"{row['status']:5s}  {row['issues_text'][:80]}")
            else:
                print("      All segment files passed validation.")

    # ── 5. Broad CNA frequencies ─────────────────────────────────────────────
    print(f"\n[5/7] Computing broad CNA frequencies (1 Mbp, streaming) …")
    print(f"      Processing {len(seg_1mb_hightf)} high-TF samples …")
    if seg_1mb_hightf:
        # Build sample→entity map from tf_df so broad CNA counts are per entity
        from ..locate.plots import ENTITY_MAP
        _sample_entity = {
            row["sample"]: entity_from_sample(str(row["patient"]))
            for _, row in tf_df.iterrows()
        }
        broad_freq = stream_broad_cna_frequencies(
            seg_1mb_hightf, high_tf_samples,
            sample_entity  = _sample_entity,
            tf_df          = tf_df,   # deduplicate to one sample per patient
            min_abs_log2r  = getattr(args, "log2_delta", 0.2),
        )
        if not broad_freq.empty:
            broad_out = outdir / "broad_cna_frequencies.tsv"
            broad_freq.to_csv(broad_out, sep="\t", index=False)
            print(f"      Written: {broad_out}  ({len(broad_freq)} loci)")
            plot_broad_cna_frequencies(
                broad_tsv           = broad_out,
                tf_tsv              = tf_out,
                output_path         = outdir / "broad_cna_frequencies_plot",
                bin_size            = getattr(args, "plot_bin_size", 5_000_000),
                exact_entity_counts = True,
            )
        else:
            print("      [WARN] No altered segments found in 1 Mbp run.")
    else:
        print("      [WARN] No 1 Mbp segment files for high-TF samples.")
    gc.collect()

    # ── 5b. Broad CNA frequencies — low-TF window (3 %–10 %) ────────────────
    _lowtf_low  = getattr(args, "lowtf_min", 0.03)
    _lowtf_high = getattr(args, "lowtf_max", 0.10)
    print(f"\n[5b] Broad CNA frequencies — low-TF window "
          f"({_lowtf_low*100:.0f}%–{_lowtf_high*100:.0f}%) …")

    # Identify samples in the window (reads filter already applied to tf_df_filtered)
    _lowtf_df = tf_df_filtered[
        (tf_df_filtered["tumor_fraction"] >= _lowtf_low) &
        (tf_df_filtered["tumor_fraction"] <  _lowtf_high)
    ]
    _lowtf_samples = _lowtf_df["sample"].tolist()
    print(f"      {len(_lowtf_samples)} samples in TF window "
          f"({_lowtf_low*100:.0f}%–{_lowtf_high*100:.0f}%).")

    _seg_1mb_lowtf = {s: p for s, p in seg_1mb_files.items()
                      if s in set(_lowtf_samples)}
    if _seg_1mb_lowtf:
        _sample_entity_lowtf = {
            row["sample"]: entity_from_sample(str(row["patient"]))
            for _, row in tf_df_filtered.iterrows()
        }
        _broad_lowtf = stream_broad_cna_frequencies_cn_only(
            _seg_1mb_lowtf, _lowtf_samples,
            sample_entity = _sample_entity_lowtf,
            tf_df         = _lowtf_df,
        )
        if not _broad_lowtf.empty:
            _broad_lowtf_out = outdir / "broad_cna_frequencies_lowtf.tsv"
            _broad_lowtf.to_csv(_broad_lowtf_out, sep="\t", index=False)
            print(f"      Written: {_broad_lowtf_out}  ({len(_broad_lowtf)} loci)")
            plot_broad_cna_frequencies_lowtf(
                broad_tsv   = _broad_lowtf_out,
                tf_tsv      = tf_out,
                output_path = outdir / "broad_cna_frequencies_lowtf_plot",
                bin_size    = getattr(args, "plot_bin_size", 5_000_000),
                tf_low      = _lowtf_low,
                tf_high     = _lowtf_high,
            )
        else:
            print("      [WARN] No altered segments found in low-TF window.")
    else:
        print("      [WARN] No 1 Mbp segment files found for low-TF window.")
    gc.collect()

    # ── 6. Focal amplifications ──────────────────────────────────────────────
    print(f"\n[6/7] Detecting focal amplifications (50 kbp, streaming) …")
    print(f"      Processing {len(seg_50k_hightf)} high-TF samples …")
    print(f"      Criteria: {args.focal_min_bp//1000}–{args.focal_max_bp//1000} kbp, "
          f"CN >= {args.amp_min_cn}, log2 delta >= {args.log2_delta}")

    if not seg_50k_hightf:
        print("      [WARN] No 50 kbp segment files for high-TF samples.")
    else:
        tmp_dir  = Path(tempfile.mkdtemp(prefix="locate_tmp_"))
        focal_tmp = stream_focal_amplifications(
            seg_files_50k        = seg_50k_hightf,
            high_tf_samples      = high_tf_samples,
            tmp_dir              = tmp_dir,
            min_bp               = args.focal_min_bp,
            max_bp               = args.focal_max_bp,
            min_cn               = args.amp_min_cn,
            min_bins             = args.focal_min_bins,
            log2_delta           = args.log2_delta,
            flank_bins           = args.focal_flank_bins,
            noise_factor         = args.focal_noise_factor,
            max_focal_per_sample = args.max_focal_per_sample,
            cgc_df               = cgc_df,
        )
        gc.collect()

        focal_out = outdir / "focal_amplifications.tsv"
        if focal_tmp.exists() and focal_tmp.stat().st_size > 0:
            shutil.copy2(focal_tmp, focal_out)
            n_events = sum(1 for _ in open(focal_out)) - 1
            print(f"      Written: {focal_out}  ({n_events} focal amp events)")
            if cgc_df is not None:
                _ann = pd.read_csv(focal_out, sep="\t",
                                   usecols=["cosmic_genes", "cosmic_tiers",
                                            "cosmic_hallmarks"],
                                   low_memory=False)
                n_gene     = (_ann["cosmic_genes"].fillna("") != "").sum()
                n_tier1    = _ann["cosmic_tiers"].fillna("").str.contains("1").sum()
                n_hallmark = (_ann["cosmic_hallmarks"].fillna("")
                                  .str.lower().str.contains("yes")).sum()
                print(f"      COSMIC CGC: {n_gene} events overlap a CGC gene "
                      f"({n_tier1} Tier 1, {n_hallmark} Hallmark).")
                del _ann; gc.collect()
        else:
            print("      [WARN] No focal amplifications detected.")

        freq_focal = aggregate_focal_frequencies(
            focal_tmp, high_tf_samples, tf_df=tf_df
        )
        freq_out   = outdir / "focal_amp_frequencies.tsv"
        freq_focal.to_csv(freq_out, sep="\t", index=False)
        print(f"      Written: {freq_out}  ({len(freq_focal)} unique focal loci)")
        if not freq_focal.empty:
            plot_focal_amp_ichorcna(
                focal_freq_tsv = freq_out,
                tf_tsv         = tf_out,
                output_path    = outdir / "focal_amp_frequencies_plot",
                min_freq       = getattr(args, "focal_freq_min_freq", 0.0),
            )

        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()

        # Per-sample diagnostic plots (includes discarded samples)
        if getattr(args, "focal_amp_diagnostics", False):
            diag_dir = outdir / "focal_amp_diagnostics"
            print(f"\n      Generating focal amp diagnostic plots "
                  f"→ {diag_dir}")
            print(f"      Candidate detection uses segmented files "
                  f"({len(seg_50k_hightf)} samples)")
            print(f"      Background scatter uses bin-level files "
                  f"({len(seg_50k_bins)} samples)")
            # Detection uses segmented .seg.txt (seg_50k_hightf) for candidates;
            # the plot function uses bin-level .cna.seg (seg_50k_bins) for scatter.
            # We pass seg_50k_bins as seg_files_50k so _load_bins gets the
            # dense bin data, but run detect_focal_amplifications_diagnostic
            # with the segmented files inside plot_all_focal_amp_diagnostics.
            # The function signature accepts both via seg_files_50k (for scatter
            # and candidate detection) — since segmented files ARE valid for
            # candidate detection we use seg_50k_hightf there and pass
            # seg_50k_bins separately for the logR scatter.
            n_diag = plot_all_focal_amp_diagnostics(
                seg_files_50k        = seg_50k_hightf,
                seg_files_50k_bins   = seg_50k_bins,
                high_tf_samples      = high_tf_samples,
                tf_df                = tf_df,
                output_dir           = diag_dir,
                min_bp               = args.focal_min_bp,
                max_bp               = args.focal_max_bp,
                min_cn               = args.amp_min_cn,
                min_bins             = args.focal_min_bins,
                log2_delta           = args.log2_delta,
                flank_bins           = args.focal_flank_bins,
                noise_factor         = args.focal_noise_factor,
                max_focal_per_sample = args.max_focal_per_sample,
            )
            print(f"      Generated {n_diag} diagnostic plot(s) in {diag_dir}")

    # ── 7. Longitudinal comparison ───────────────────────────────────────────
    print(f"\n[7/7] Longitudinal CN comparison …")
    focal_out = outdir / "focal_amplifications.tsv"
    focal_for_longit = pd.DataFrame()
    if focal_out.exists() and focal_out.stat().st_size > 0:
        _header = pd.read_csv(focal_out, sep="\t", nrows=0).columns.tolist()
        _cn_col = next(
            (c for c in _header
             if c.lower().replace(".", "_").replace(" ", "_")
                in ("copy_number", "cn", "modal_cn", "total_cn")),
            None,
        )
        _want = {"sample", "chrom", "start", "end"}
        if _cn_col:
            _want.add(_cn_col)
        for _opt in ("cosmic_genes", "cosmic_hallmarks", "oncogenes"):
            if _opt in _header:
                _want.add(_opt)
        _keep = [c for c in _header if c in _want]
        if "sample" in _header:
            focal_for_longit = pd.read_csv(
                focal_out, sep="\t", usecols=_keep, low_memory=False)
            if _cn_col and _cn_col != "copy.number":
                focal_for_longit = focal_for_longit.rename(
                    columns={_cn_col: "copy.number"})

    longit = pd.DataFrame()
    try:
        from ..locate.longitudinal import longitudinal_comparison
        longit = longitudinal_comparison(focal_for_longit, tf_df)
    except ImportError:
        pass
    longit_out = outdir / "longitudinal_comparison.tsv"
    longit.to_csv(longit_out, sep="\t", index=False)
    patients_tracked = longit["patient"].nunique() if not longit.empty else 0
    print(f"      Written: {longit_out}  ({patients_tracked} patients tracked)")

    # Emergent focal amps (present in follow-up but absent at baseline)
    print(f"      Identifying emergent focal amplifications …")
    emergent = emergent_focal_amps(focal_for_longit, tf_df)
    emergent_out = outdir / "emergent_focal_amps.tsv"
    emergent.to_csv(emergent_out, sep="\t", index=False)
    n_emergent_patients = emergent["patient"].nunique() if not emergent.empty else 0
    n_emergent_events   = len(emergent)
    print(f"      Written: {emergent_out}  "
          f"({n_emergent_events} events in {n_emergent_patients} patients)")
    if not emergent.empty:
        plot_emergent_focal_amps(
            emergent_tsv = emergent_out,
            output_path  = outdir / "emergent_focal_amps_plot",
        )

    # Pairwise longitudinal CN plots (one per consecutive timepoint pair)
    if getattr(args, "longitudinal_plots", False):
        longit_plot_dir = Path(
            args.longitudinal_plot_dir
            if getattr(args, "longitudinal_plot_dir", None)
            else str(outdir / "longitudinal_plots")
        )
        print(f"      Generating pairwise longitudinal CN plots ...")
        print(f"      Output directory: {longit_plot_dir}")

        # Build a tf_df restricted to samples that actually have a 50 kbp
        # seg file — avoids patients being silently skipped because one of
        # their timepoints is missing from the 50 kbp run.
        # Also include ALL timepoints (not just high-TF) so that a patient
        # whose earlier sample fell below the TF threshold is still shown;
        # the plot will simply have that sample's logR less corrected.
        _seg50_samples  = set(seg_50k_bins.keys())  # full 50 kbp bin-level index
        _tf_for_longit  = tf_df[tf_df["sample"].isin(_seg50_samples)].copy()

        print(f"      Samples in tf_df           : {len(tf_df)}")
        print(f"      Samples with 50 kbp seg    : {len(_tf_for_longit)}")
        _multi = sum(
            1 for _, g in _tf_for_longit.groupby("patient")
            if g["timepoint"].nunique() >= 2
        )
        print(f"      Patients with >= 2 timepoints and seg files: {_multi}")

        # Use 50 kbp .cna.seg files for bin-level (unsegmented) log2 ratios
        _max_noise = getattr(args, "max_noise", 0.15) or None
        _min_signal = getattr(args, "min_signal", None)
        _emergent_tsv = outdir / "emergent_cna_regions.tsv"
        n_generated = plot_all_pairwise_longitudinal(
            tf_df                = _tf_for_longit,
            seg_files            = seg_50k_bins,
            seg_files_segs       = seg_50k_files,
            output_dir           = longit_plot_dir,
            min_tf               = 0.10,
            focal_tsv            = outdir / "focal_amplifications.tsv",
            max_noise            = _max_noise,
            min_signal           = _min_signal,
            min_altered_fraction = getattr(args, "min_altered_fraction", 0.05),
            plot_raw             = getattr(args, "plot_raw", False),
            emergent_min_bins    = getattr(args, "emergent_min_bins", 3),
            emergent_min_delta   = getattr(args, "emergent_min_delta", 0.5),
            emergent_tsv         = _emergent_tsv,
            clean_plots          = not getattr(args, "no_clean_plots", False),
        )
        if _emergent_tsv.exists() and _emergent_tsv.stat().st_size > 0:
            n_emerg = sum(1 for _ in open(_emergent_tsv)) - 1
            print(f"      Emergent CNA regions: {n_emerg} → {_emergent_tsv}")
        print(f"      Generated {n_generated} pairwise plot(s).")

        # Note: differential plot is now included as the bottom panel
        # of the combined plot above — no separate differential plots needed.

        # Focal amplification trajectory plots (one pair of plots per
        # patient with >= 2 timepoints and at least one focal amp event)
        print(f"      Generating focal amplification trajectory plots ...")
        focal_traj_dir = longit_plot_dir / "focal_amp_trajectories"
        n_traj = plot_all_focal_amp_trajectories(
            focal_tsv      = outdir / "focal_amplifications.tsv",
            tf_df          = _tf_for_longit,
            seg_files_bins = seg_50k_bins,
            output_dir     = focal_traj_dir,
            min_tf         = 0.10,
            min_timepoints = 2,
            seg_files_segs = seg_50k_files,
            min_reads      = 1_000_000,
            max_scale      = 3.0,
        )
        print(f"      Generated {n_traj} focal amp trajectory plot pair(s) "
              f"→ {focal_traj_dir}")

        # Cohort-level stability overview
        print(f"\n      Generating cohort focal amp stability plot …")
        n_stab = plot_focal_amp_cohort_stability(
            focal_tsv      = outdir / "focal_amplifications.tsv",
            tf_df          = _tf_for_longit,
            seg_files_bins = seg_50k_bins,
            output_dir     = longit_plot_dir / "focal_amp_trajectories",
            seg_files_segs = seg_50k_files,
            min_tf         = 0.10,
            min_reads      = 1_000_000,
            max_scale      = 3.0,
            min_timepoints = 2,
        )
        print(f"      Stability plot: {n_stab} samples included.")

        plot_focal_amp_stability_focused(
            focal_tsv      = outdir / "focal_amplifications.tsv",
            tf_df          = _tf_for_longit,
            seg_files_bins = seg_50k_bins,
            output_dir     = longit_plot_dir / "focal_amp_trajectories",
            seg_files_segs = seg_50k_files,
            min_tf         = 0.10,
            min_reads      = 1_000_000,
            max_scale      = 3.0,
            min_timepoints = 2,
            delta_threshold = 0.5,
        )

        # ── TF estimates from empirical scaling ───────────────────────────
        print(f"\n      Computing TF estimates from empirical scale factors …")
        compute_scaled_tf_estimates(
            tf_df          = _tf_for_longit,
            seg_files      = seg_50k_bins,
            seg_files_segs = seg_50k_files,
            output_dir     = longit_plot_dir / "tf_estimates",
            min_tf         = getattr(args, "tf_threshold", 0.03),
        )

    # ── Summary ──────────────────────────────────────────────────────────────

    # Entity map: first letter of patient ID → entity name
    ENTITY_MAP: dict[str, str] = {
        "B": "Breast",
        "C": "Colorectal",
        "P": "Prostate",
        "L": "Lung",
        "E": "Ovarian",
    }

    def _entity_stats(df: pd.DataFrame) -> dict[str, dict]:
        """
        Given a DataFrame with 'sample' and 'patient' columns, return
        per-entity counts of samples and patients keyed by entity name.
        First letter of the patient ID determines the entity.
        """
        stats: dict[str, dict] = {}
        df = df.copy()
        df["entity"] = df["patient"].str[0].str.upper().map(ENTITY_MAP).fillna("Other")
        for entity, grp in df.groupby("entity"):
            stats[entity] = {
                "n_samples":  grp["sample"].nunique(),
                "n_patients": grp["patient"].nunique(),
            }
        return stats

    all_stats    = _entity_stats(tf_df)
    hightf_stats = _entity_stats(high_tf[["sample", "patient"]])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Overall counts
    n_patients_total  = tf_df["patient"].nunique()
    n_patients_hightf = high_tf["patient"].nunique()
    print(f"  Total samples                      : {len(tf_df)}")
    print(f"  Total patients                     : {n_patients_total}")
    print(f"  Samples with TF >= {threshold*100:.0f}%            : {len(high_tf_samples)}")
    print(f"  Patients with TF >= {threshold*100:.0f}%           : {n_patients_hightf}")

    # Per-entity breakdown (all samples)
    # Add read depth stats to the summary
    has_reads = "total_reads" in tf_df.columns and tf_df["total_reads"].notna().any()
    if has_reads:
        median_reads = tf_df["total_reads"].median()
        print(f"  Median total reads (all samples)   : {median_reads:,.0f}")
        low_reads = tf_df[tf_df["total_reads"] < median_reads * 0.5]
        if not low_reads.empty:
            print(f"  Samples with < 50%% median reads   : {len(low_reads)} "
                  f"(potential QC flags)")

    print(f"\n  {'Entity':<14} {'Samples (all)':>13} {'Patients (all)':>14} "
          f"{'Samples (TF>={threshold*100:.0f}%)':>16} {'Patients (TF>={threshold*100:.0f}%)':>17}")
    print("  " + "-" * 76)
    all_entities = sorted(set(all_stats) | set(hightf_stats))
    for entity in all_entities:
        a = all_stats.get(entity,    {"n_samples": 0, "n_patients": 0})
        h = hightf_stats.get(entity, {"n_samples": 0, "n_patients": 0})
        print(f"  {entity:<14} {a['n_samples']:>13} {a['n_patients']:>14} "
              f"{h['n_samples']:>16} {h['n_patients']:>17}")
    print("  " + "-" * 76)
    print(f"  {'TOTAL':<14} {len(tf_df):>13} {n_patients_total:>14} "
          f"{len(high_tf_samples):>16} {n_patients_hightf:>17}")

    # CNA results
    print()
    _broad = _safe_read(outdir / "broad_cna_frequencies.tsv",
                        usecols=["n_amp", "n_del"])
    if _broad is not None:
        n_alt = int((_broad["n_amp"] + _broad["n_del"]).gt(0).sum())
        print(f"  Broad altered loci (1 Mbp)         : {n_alt}")

    _focal_df = _safe_read(outdir / "focal_amplifications.tsv")
    print(f"  Focal amp events (50 kbp)          : "
          f"{len(_focal_df) if _focal_df is not None else 0}")

    _freq = _safe_read(outdir / "focal_amp_frequencies.tsv")
    print(f"  Unique focal amp loci              : "
          f"{len(_freq) if _freq is not None else 0}")
    if _freq is not None:
        for gc_col in ("cosmic_genes", "oncogenes"):
            if gc_col in _freq.columns:
                n_gene = int((_freq[gc_col].fillna("") != "").sum())
                print(f"  Focal loci with CGC gene overlap   : {n_gene}")
                break

    print(f"  Longitudinal patients              : {patients_tracked}")

    if args.skip_validation:
        print(f"  Seg files validated                : skipped")
    elif not val_df.empty:
        n_warn  = int((val_df["status"] == "WARN").sum())
        n_error = int((val_df["status"] == "ERROR").sum())
        print(f"  Seg files validated                : {len(val_df)}  "
              f"({n_warn} warnings, {n_error} errors)")

    if incomplete_rows:
        n_inc = sum(len(i) for i in incomplete_rows)
        print(f"  Incomplete runs detected           : "
              f"{n_inc} (see incomplete_runs.tsv)")

    print(f"\nAll results written to: {outdir.resolve()}")
    print("=" * 70)
