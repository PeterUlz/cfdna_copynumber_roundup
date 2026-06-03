"""
cBioPortal CNA frequency workflow orchestration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..common.cosmic import load_cosmic_cgc, annotate_with_cgc
from ..cbio.loader import load_entity_files, make_bins
from ..cbio.liftover import liftover_entity_files, check_chain_file, check_liftover_binary
from ..cbio.frequencies import (
    compute_frequencies,
    detect_focal_amps_cbio,
)
from ..cbio.plots import plot_cna_frequencies
from ..ichorcna.focal_amp import aggregate_focal_frequencies
from ..ichorcna.plots import plot_focal_amp_frequencies as plot_focal_amp_ichorcna


def _parse_paths(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in raw.split(";") if p.strip()]


def run(args: argparse.Namespace) -> None:
    # ── Build entity → list[Path] map ────────────────────────────────────────
    entity_files: dict[str, list[Path]] = {}
    for attr in ("breast", "prostate", "colorectal"):
        val = getattr(args, attr, None)
        if val:
            entity_files[attr] = _parse_paths(val)
    for pair in getattr(args, "seg", []):
        if "=" not in pair:
            print(f"[ERROR] --seg entries must be NAME=FILE, got: {pair}")
            sys.exit(1)
        name, fpath = pair.split("=", 1)
        entity_files[name.strip()] = _parse_paths(fpath)

    if not entity_files:
        print("[ERROR] No input files. Use --breast / --prostate / "
              "--colorectal / --seg.")
        sys.exit(1)

    for entity, paths in entity_files.items():
        for path in paths:
            if not path.is_file():
                print(f"[ERROR] File not found for '{entity}': {path}")
                sys.exit(1)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load COSMIC CGC ───────────────────────────────────────────────────────
    cgc_df = None
    if args.cosmic_cgc:
        cgc_path = Path(args.cosmic_cgc)
        if not cgc_path.is_file():
            print(f"[ERROR] COSMIC CGC file not found: {cgc_path}")
            sys.exit(1)
        try:
            cgc_df = load_cosmic_cgc(cgc_path)
        except ValueError as e:
            print(f"[ERROR] {e}"); sys.exit(1)
    else:
        print("[INFO] No --cosmic-cgc provided; gene annotation skipped.")

    print("=" * 65)
    print("cBioPortal CNA Frequency Analysis")
    print("=" * 65)

    # ── 1. Load segments ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading segment files …")
    seg_by_entity = load_entity_files(entity_files, args.min_seg_size)

    # ── 1b. LiftOver hg19 → hg38 (optional) ─────────────────────────────────
    if getattr(args, "liftover", False):
        if not args.chain_file:
            print("[ERROR] --chain-file is required when --liftover is set.")
            print("        Download: wget https://hgdownload.gi.ucsc.edu/"
                  "goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz")
            sys.exit(1)
        try:
            check_chain_file(args.chain_file)
            check_liftover_binary(args.liftover_bin)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        print(f"\n[1b] Lifting coordinates hg19 → hg38 "
              f"(min-match={args.liftover_min_match}) …")
        seg_by_entity = liftover_entity_files(
            seg_by_entity,
            chain_file   = args.chain_file,
            liftover_bin = args.liftover_bin,
            min_match    = args.liftover_min_match,
        )

    # ── 2. Broad CNA frequencies ──────────────────────────────────────────────
    print(f"\n[2/4] Computing broad CNA frequencies "
          f"(bin={args.bin_size//1_000_000} Mbp, "
          f"gain>{args.gain_threshold}, loss<{args.loss_threshold}) …")
    bins_df          = make_bins(args.bin_size)
    freq_by_entity:  dict[str, pd.DataFrame] = {}
    all_freq_frames: list[pd.DataFrame]      = []

    for entity, seg_df in seg_by_entity.items():
        print(f"      Processing {entity} …")
        freq = compute_frequencies(
            seg_df, bins_df,
            gain_threshold=args.gain_threshold,
            loss_threshold=args.loss_threshold,
        )
        freq_by_entity[entity] = freq
        out_tsv = outdir / f"{entity}_cna_frequencies.tsv"
        freq.to_csv(out_tsv, sep="\t", index=False)
        print(f"      Written: {out_tsv}")
        all_freq_frames.append(freq)

    combined = pd.concat(all_freq_frames, ignore_index=True)
    comb_out = outdir / "combined_cna_frequencies.tsv"
    combined.to_csv(comb_out, sep="\t", index=False)
    print(f"      Written: {comb_out}")

    # ── 3. Broad CNA plot ─────────────────────────────────────────────────────
    print(f"\n[3/4] Generating genome-wide CNA frequency plot …")
    plot_cna_frequencies(freq_by_entity, outdir / "cna_frequency_plot")

    # ── 4. Focal amplifications ───────────────────────────────────────────────
    print(f"\n[4/4] Detecting focal amplifications …")
    print(f"      (size {args.focal_min_bp//1000}–{args.focal_max_bp//1000} kbp, "
          f"seg.mean >= {args.focal_log2_min}, "
          f"delta >= {args.focal_log2_delta} vs "
          f"{args.focal_flank_bp//1_000_000} Mbp background)")

    import tempfile, shutil
    tmp_dir  = Path(tempfile.mkdtemp(prefix='cbio_focal_'))
    tmp_path = tmp_dir / 'focal_amp_events.tsv'
    header_written = False

    # Build a minimal tf_df compatible with aggregate_focal_frequencies
    tf_rows = []
    for entity, seg_df in seg_by_entity.items():
        for pat in seg_df['patient'].unique():
            tf_rows.append({'sample': pat, 'patient': pat,
                            'tumor_fraction': float('nan'), 'entity': entity})
    cbio_tf_df = pd.DataFrame(tf_rows).drop_duplicates(subset=['patient'])
    all_patients = cbio_tf_df['patient'].tolist()

    for entity, seg_df in seg_by_entity.items():
        n_patients = seg_df['patient'].nunique()
        print(f"      [{entity}] {n_patients} patients …")

        focal = detect_focal_amps_cbio(
            seg_df,
            focal_min_bp = args.focal_min_bp,
            focal_max_bp = args.focal_max_bp,
            log2_min     = args.focal_log2_min,
            log2_delta   = args.focal_log2_delta,
            flank_bp     = args.focal_flank_bp,
        )
        print(f"        Found {len(focal)} focal amp segments in "
              f"{focal['patient'].nunique() if not focal.empty else 0} patients.")

        if focal.empty:
            continue

        focal = annotate_with_cgc(focal, cgc_df)

        # Write per-entity TSV for reference
        focal.to_csv(outdir / f'{entity}_focal_amplifications.tsv',
                     sep='\t', index=False)

        # Map to ichorCNA streaming TSV schema for shared aggregation
        focal_out = focal.copy()
        focal_out['sample'] = focal_out['patient']
        if 'seg_mean' in focal_out.columns:
            focal_out = focal_out.rename(columns={'seg_mean': 'logr'})
        if 'copy_number' in focal_out.columns:
            focal_out = focal_out.rename(columns={'copy_number': 'copy.number'})
        focal_out.to_csv(
            tmp_path, sep='\t', index=False,
            mode='a' if header_written else 'w',
            header=not header_written,
        )
        header_written = True

    # Aggregate using shared ichorCNA function (bin coverage + merging)
    freq_focal = aggregate_focal_frequencies(
        tmp_path, all_patients, tf_df=cbio_tf_df
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not freq_focal.empty:
        freq_out = outdir / 'focal_amp_frequencies.tsv'
        freq_focal.to_csv(freq_out, sep='\t', index=False)
        print(f"      Written: {freq_out}  ({len(freq_focal)} regions)")

        # Build a tf_tsv for the plot function
        tf_tsv = outdir / 'cbio_tf_summary.tsv'
        cbio_tf_df.to_csv(tf_tsv, sep='\t', index=False)
        plot_focal_amp_ichorcna(
            focal_freq_tsv = freq_out,
            tf_tsv         = tf_tsv,
            output_path    = outdir / 'focal_amp_frequencies_plot',
            min_freq       = getattr(args, 'focal_min_freq', 0.0),
        )
    else:
        print('      No focal amplifications detected.')


    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for entity, freq in freq_by_entity.items():
        from ..common.constants import ENTITY_PALETTE
        n_pts    = int(freq["n_total"].iloc[0])
        n_gained = int((freq["freq_gain"] > 0).sum())
        n_lost   = int((freq["freq_loss"] > 0).sum())
        label    = ENTITY_PALETTE.get(entity, {}).get("label", entity)
        if not freq_focal.empty:
            ent_col = f"freq_{entity}"
            if ent_col in freq_focal.columns:
                n_loci = int((freq_focal[ent_col] > 0).sum())
            else:
                n_loci = len(freq_focal)
        else:
            n_loci = 0
        print(f"  {label:25s}: {n_pts:4d} patients | "
              f"{n_gained:4d} bins with gain | "
              f"{n_lost:4d} bins with loss | "
              f"{n_loci:4d} focal amp regions")
    print(f"\nResults written to: {outdir.resolve()}")
    print("=" * 65)