"""
cfdna-cnv  —  cfDNA copy-number analysis toolkit

Full pipeline:
    ichorcna   Analyze ichorCNA output (broad CNA + focal amps + longitudinal)
    cbio       Analyze cBioPortal segmentation files (broad CNA + focal amps)

Focused commands (ichorCNA data):
    focal-amps       Detect and plot focal amplifications across a cohort
    compare          Pairwise longitudinal CN comparison for two timepoints
    trajectories     CN trajectories of focal amps across patient timepoints
    signatures       Copy-number signature analysis (sigminer)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_cosmic_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cosmic-cgc", default=None, metavar="FILE",
        help="COSMIC Cancer Gene Census CSV/TSV for focal amp annotation "
             "(download from https://cancer.sanger.ac.uk/census).",
    )


def _add_common_ichorcna_args(p: argparse.ArgumentParser) -> None:
    """Args shared by all ichorCNA-based subcommands."""
    p.add_argument("--config", default=None, metavar="FILE",
                   help="YAML config file. CLI arguments override config values.")
    p.add_argument("--write-config", default=None, metavar="FILE",
                   help="Write a default config YAML to FILE and exit.")
    p.add_argument("--tf-threshold", type=float, default=0.03,
                   help="Minimum tumor fraction (default 3%%).")
    p.add_argument("--min-reads", type=int, default=1_000_000,
                   help="Minimum total reads from WIG file (default 1,000,000). "
                        "Set to 0 to disable.")
    p.add_argument("--exclude-entities", nargs="+", default=[], metavar="ENTITY",
                   help="Entity names to exclude (Breast, Colorectal, Prostate, "
                        "Lung, Ovarian, Other).")
    p.add_argument("--rscript", default="Rscript", metavar="BIN",
                   help="Path to Rscript binary (default: Rscript).")
    _add_cosmic_arg(p)


def _add_focal_amp_args(p: argparse.ArgumentParser) -> None:
    """Focal amp detection args shared by several subcommands."""
    p.add_argument("--amp-min-cn",          type=int,   default=4,
                   help="Minimum CN for focal amp (default 4).")
    p.add_argument("--focal-min-bp",        type=int,   default=50_000,
                   help="Minimum focal amp segment size in bp (default 50,000).")
    p.add_argument("--focal-max-bp",        type=int,   default=500_000,
                   help="Maximum focal amp segment size in bp (default 500,000).")
    p.add_argument("--focal-min-bins",      type=int,   default=3,
                   help="Minimum 50 kbp bins in a segment (default 3).")
    p.add_argument("--focal-flank-bins",    type=int,   default=2,
                   help="Background window: this many 1 Mbp bins each side "
                        "(default 2).")
    p.add_argument("--max-focal-per-sample",type=int,   default=5,
                   help="Discard samples with more focal amps than this "
                        "(default 5).")
    p.add_argument("--focal-noise-factor",  type=float, default=2.0,
                   help="Segment logR must be >= this × sample noise "
                        "(default 2.0).")
    p.add_argument("--focal-freq-min-freq", type=float, default=0.0,
                   help="Minimum frequency for focal amp frequency plot "
                        "(default 0.0 = show all).")
    p.add_argument("--log2-delta",          type=float, default=0.2,
                   help="Min log2 elevation above local background (default 0.2).")
    p.add_argument("--focal-amp-diagnostics", action="store_true",
                   help="Generate per-sample diagnostic plots.")


# ---------------------------------------------------------------------------
# ichorcna — full pipeline (unchanged)
# ---------------------------------------------------------------------------

def _build_ichorcna_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ichorcna",
        help="Full ichorCNA analysis pipeline (broad CNA + focal amps + "
             "longitudinal + signatures).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_ichorcna_args(p)
    p.add_argument("--dir-1mb",  default=None,
                   help="Directory containing ichorCNA 1 Mbp run output.")
    p.add_argument("--dir-50kb", default=None,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--output", default="ichorcna_results",
                   help="Output directory.")
    _add_focal_amp_args(p)
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip segment file validation.")
    p.add_argument("--plot-bin-size", type=int, default=5_000_000,
                   help="Bin size in bp for the broad CNA frequency plot.")
    p.add_argument("--longitudinal-plots", action="store_true",
                   help="Generate longitudinal pairwise CN comparison plots.")
    p.add_argument("--longitudinal-plot-dir", default=None, metavar="DIR",
                   help="Output directory for longitudinal plots.")
    p.add_argument("--max-noise", type=float, default=0.15,
                   help="Max neighbour noise for longitudinal plots.")
    p.add_argument("--min-signal", type=float, default=None,
                   help="Min SNR for longitudinal plots.")
    p.add_argument("--min-altered-fraction", type=float, default=0.05,
                   help="Min fraction of altered bins for longitudinal plots.")
    p.add_argument("--plot-raw", action="store_true",
                   help="Also output unsmoothed longitudinal plots.")
    p.add_argument("--emergent-min-bins", type=int, default=3,
                   help="Min consecutive bins to call emergent CNA (default 3).")
    p.add_argument("--emergent-min-delta", type=float, default=0.5,
                   help="Min |delta log2R| for emergent CNA (default 0.5).")
    p.add_argument("--no-clean-plots", action="store_true",
                   help="Disable clean publication-style longitudinal plots.")
    p.add_argument("--sigminer", action="store_true",
                   help="Run sigminer CN signature analysis.")
    p.add_argument("--sigminer-ref", nargs="+", default=["CNS_TCGA"], metavar="REF",
                   help="Reference signature sets (default: CNS_TCGA).")
    p.add_argument("--sigminer-min-tf", type=float, default=0.15,
                   help="Min TF for sigminer (default 15%%).")
    p.add_argument("--sigminer-genome", default="hg38",
                   choices=["hg38", "hg19"],
                   help="Reference genome for sigminer.")
    p.add_argument("--sigminer-cores", type=int, default=1,
                   help="Parallel cores for sigminer.")
    p.add_argument("--sigminer-n-sig", type=int, default=6,
                   help="Number of de novo CN signatures to extract.")
    p.add_argument("--sigminer-no-run", action="store_true",
                   help="Prepare R script only, do not execute.")
    p.add_argument("--liquidcna", action="store_true",
                   help="Run liquidCNA subclonal evolution analysis.")
    p.add_argument("--liquidcna-functions", default=None, metavar="FILE",
                   help="Path to mixture_estimation_functions.R.")
    p.add_argument("--liquidcna-min-tf", type=float, default=0.20,
                   help="Min TF for liquidCNA (default 20%%).")
    p.add_argument("--liquidcna-min-timepoints", type=int, default=2,
                   help="Min qualifying timepoints for liquidCNA (default 2).")
    p.add_argument("--liquidcna-no-run", action="store_true",
                   help="Prepare inputs only, do not execute Rscript.")
    p.set_defaults(func=_run_ichorcna)


# ---------------------------------------------------------------------------
# focal-amps — detect and plot focal amplifications across a cohort
# ---------------------------------------------------------------------------

def _build_focal_amps_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "focal-amps",
        help="Detect focal amplifications and compute cohort-level frequencies "
             "from ichorCNA output. Runs broad CNA + focal amp detection only "
             "(no longitudinal, no signatures).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  cfdna-cnv focal-amps --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --cosmic-cgc Census.tsv --output focal_results

  # With a config file:
  cfdna-cnv focal-amps --config analysis.yaml
""",
    )
    _add_common_ichorcna_args(p)
    p.add_argument("--dir-1mb",  default=None,
                   help="Directory containing ichorCNA 1 Mbp run output.")
    p.add_argument("--dir-50kb", default=None,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--output", default="focal_amp_results",
                   help="Output directory (default: focal_amp_results).")
    _add_focal_amp_args(p)
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip segment file validation.")
    p.add_argument("--plot-bin-size", type=int, default=5_000_000,
                   help="Bin size in bp for the broad CNA frequency plot.")
    p.set_defaults(func=_run_focal_amps)


def _run_focal_amps(args: argparse.Namespace) -> None:
    """Delegate to ichorcna_cmd.run with longitudinal and signature flags off."""
    args.longitudinal_plots    = False
    args.longitudinal_plot_dir = None
    args.max_noise             = 0.15
    args.min_signal            = None
    args.min_altered_fraction  = 0.05
    args.plot_raw              = False
    args.emergent_min_bins     = 3
    args.emergent_min_delta    = 0.5
    args.no_clean_plots        = True
    args.sigminer              = False
    args.liquidcna             = False
    from .commands.ichorcna_cmd import run
    run(args)


# ---------------------------------------------------------------------------
# compare — pairwise longitudinal CN comparison
# ---------------------------------------------------------------------------

def _build_compare_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compare",
        help="Pairwise longitudinal CN comparison between two ichorCNA "
             "timepoints. Generates overlaid CN profiles, differential plots, "
             "and emergent CNA detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Two specific samples in the same 50kb run directory:
  cfdna-cnv compare --dir-50kb /runs/50kb \\
      --sample-a P12_TP1 --sample-b P12_TP2 --output compare_P12

  # All consecutive longitudinal pairs in the cohort:
  cfdna-cnv compare --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --all-pairs --output longitudinal_results
""",
    )
    _add_common_ichorcna_args(p)
    p.add_argument("--dir-1mb",  default=None,
                   help="Directory containing ichorCNA 1 Mbp run output.")
    p.add_argument("--dir-50kb", default=None,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--output", default="compare_results",
                   help="Output directory (default: compare_results).")
    p.add_argument("--max-noise", type=float, default=0.15,
                   help="Max neighbour noise; noisier pairs are skipped.")
    p.add_argument("--min-signal", type=float, default=None,
                   help="Min SNR (p95/noise); null = no filter.")
    p.add_argument("--min-altered-fraction", type=float, default=0.05,
                   help="Min fraction of altered bins for a pair to be plotted.")
    p.add_argument("--plot-raw", action="store_true",
                   help="Also output unsmoothed plots.")
    p.add_argument("--emergent-min-bins", type=int, default=3,
                   help="Min consecutive bins to call emergent CNA.")
    p.add_argument("--emergent-min-delta", type=float, default=0.5,
                   help="Min |delta log2R| for emergent CNA.")
    p.add_argument("--no-clean-plots", action="store_true",
                   help="Disable clean publication-style longitudinal plots.")
    _add_focal_amp_args(p)
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip segment file validation.")
    p.add_argument("--plot-bin-size", type=int, default=5_000_000,
                   help="Bin size for broad CNA frequency plot.")
    p.set_defaults(func=_run_compare)


def _run_compare(args: argparse.Namespace) -> None:
    """Delegate to ichorcna_cmd.run with longitudinal plots enabled."""
    args.longitudinal_plots    = True
    args.longitudinal_plot_dir = getattr(args, "longitudinal_plot_dir", None)
    args.sigminer              = False
    args.liquidcna             = False
    from .commands.ichorcna_cmd import run
    run(args)


# ---------------------------------------------------------------------------
# trajectories — focal amp CN trajectories across patient timepoints
# ---------------------------------------------------------------------------

def _build_compare_samples_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compare-samples",
        help="Direct pairwise CN comparison between two explicitly named "
             "ichorCNA samples. No TF filtering or cohort logic — just load "
             "the two samples and plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Minimal: compare two samples, auto-detect TF from params files
  cfdna-cnv compare-samples \\
      --dir-50kb /runs/50kb \\
      --sample-a P12_TP1 --sample-b P12_TP2 \\
      --output P12_comparison

  # With explicit TFs and optional 1Mb focal amp annotation:
  cfdna-cnv compare-samples \\
      --dir-50kb /runs/50kb --dir-1mb /runs/1mb \\
      --sample-a P12_TP1 --tf-a 0.42 \\
      --sample-b P12_TP2 --tf-b 0.21 \\
      --patient P12 --label "Pre vs Post treatment" \\
      --focal-tsv /results/focal_amplifications.tsv \\
      --output P12_comparison

  # Compare multiple pairs in one call:
  cfdna-cnv compare-samples \\
      --dir-50kb /runs/50kb \\
      --pairs "P12_TP1:P12_TP2" "B34_TP1:B34_TP3" \\
      --output batch_comparisons
""",
    )
    p.add_argument("--dir-50kb", required=True,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--dir-1mb", default=None,
                   help="Directory containing ichorCNA 1 Mbp run output "
                        "(optional; used for TF lookup if --tf-a/b not given).")
    p.add_argument("--output", default="compare_samples_results",
                   help="Output directory (default: compare_samples_results).")

    # ── Single-pair mode ────────────────────────────────────────────────
    pair = p.add_argument_group("single pair")
    pair.add_argument("--sample-a", default=None, metavar="NAME",
                      help="Name of the earlier sample (as it appears in the "
                           "ichorCNA output file names).")
    pair.add_argument("--sample-b", default=None, metavar="NAME",
                      help="Name of the later sample.")
    pair.add_argument("--tf-a", type=float, default=None,
                      help="Tumor fraction of sample A (0–1). If omitted, "
                           "read from the .params.txt file.")
    pair.add_argument("--tf-b", type=float, default=None,
                      help="Tumor fraction of sample B (0–1). If omitted, "
                           "read from the .params.txt file.")
    pair.add_argument("--patient", default=None,
                      help="Patient ID for plot title and focal amp lookup. "
                           "Inferred from sample name if omitted.")
    pair.add_argument("--label", default=None,
                      help="Free-text subtitle for the plot "
                           "(e.g. 'Pre vs Post treatment').")
    pair.add_argument("--timepoint-a", type=int, default=None,
                      help="Timepoint index for sample A (e.g. 1).")
    pair.add_argument("--timepoint-b", type=int, default=None,
                      help="Timepoint index for sample B (e.g. 2).")

    # ── Batch mode ──────────────────────────────────────────────────────
    batch = p.add_argument_group("batch mode (alternative to single pair)")
    batch.add_argument("--pairs", nargs="+", default=[], metavar="A:B",
                       help="One or more sample pairs as 'SAMPLE_A:SAMPLE_B'. "
                            "TFs are auto-detected from params files.")

    # ── Shared options ──────────────────────────────────────────────────
    p.add_argument("--focal-tsv", default=None, metavar="FILE",
                   help="Path to focal_amplifications.tsv for focal amp "
                        "shading (optional).")
    p.add_argument("--max-noise", type=float, default=0.15,
                   help="Max neighbour noise; pairs above this are skipped.")
    p.add_argument("--min-signal", type=float, default=None,
                   help="Min SNR; pairs below this are skipped (default: off).")
    p.add_argument("--min-altered-fraction", type=float, default=0.05,
                   help="Min fraction of altered bins.")
    p.add_argument("--emergent-min-bins", type=int, default=3,
                   help="Min consecutive bins to call emergent CNA.")
    p.add_argument("--emergent-min-delta", type=float, default=0.5,
                   help="Min |delta log2R| for emergent CNA.")
    p.add_argument("--no-clean-plots", action="store_true",
                   help="Skip clean publication-style plots.")
    p.add_argument("--plot-raw", action="store_true",
                   help="Also output unsmoothed plots.")
    p.add_argument("--cosmic-cgc", default=None, metavar="FILE",
                   help="COSMIC Cancer Gene Census file for annotation.")
    p.set_defaults(func=_run_compare_samples)


def _run_compare_samples(args: argparse.Namespace) -> None:
    """
    Directly invoke plot_longitudinal_combined for one or more explicit
    sample pairs, without running the full ichorCNA pipeline.
    """
    import re
    from pathlib import Path as _Path

    from .common.io import find_sample_files, normalize_sample_name
    from .common.cosmic import load_cosmic_cgc, annotate_with_cgc
    from .ichorcna.longitudinal_plots import plot_longitudinal_combined

    outdir = _Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    dir_50kb = _Path(args.dir_50kb)
    dir_1mb  = _Path(args.dir_1mb) if args.dir_1mb else None

    # ── Discover all 50 kbp bin files and seg-mean files ─────────────────
    seg_bins = find_sample_files(dir_50kb, ".cna.seg")
    seg_segs = find_sample_files(dir_50kb, ".seg.txt")
    print(f"  Found {len(seg_bins)} 50 kbp bin files and "
          f"{len(seg_segs)} seg-mean files in {dir_50kb}")

    # ── Discover TF from params files if dir_1mb provided ────────────────
    tf_lookup: dict[str, float] = {}
    if dir_1mb is not None:
        params_files = find_sample_files(dir_1mb, ".params.txt")
        for samp, ppath in params_files.items():
            try:
                import re as _re
                txt = ppath.read_text()
                m = _re.search(r"tumor_fraction\s*=\s*([0-9.]+)", txt)
                if m:
                    tf_lookup[samp] = float(m.group(1))
            except Exception:
                pass
        print(f"  Loaded TF for {len(tf_lookup)} samples from params files.")

    # ── Build list of (sample_a, sample_b, tf_a, tf_b, label) tuples ─────
    pairs: list[tuple[str, str, float | None, float | None, str | None]] = []

    if args.pairs:
        for pair_str in args.pairs:
            if ":" not in pair_str:
                print(f"  [WARN] Skipping malformed pair '{pair_str}' "
                      f"(expected format 'SAMPLE_A:SAMPLE_B')")
                continue
            sa, sb = pair_str.split(":", 1)
            pairs.append((sa.strip(), sb.strip(), None, None, None))
    elif args.sample_a and args.sample_b:
        pairs.append((args.sample_a, args.sample_b,
                      args.tf_a, args.tf_b, args.label))
    else:
        print("  [ERROR] Specify either --sample-a/--sample-b or --pairs.")
        sys.exit(1)

    # ── Load COSMIC if provided ───────────────────────────────────────────
    cosmic_df = None
    if getattr(args, "cosmic_cgc", None):
        try:
            cosmic_df = load_cosmic_cgc(_Path(args.cosmic_cgc))
            print(f"  Loaded {len(cosmic_df)} COSMIC CGC genes.")
        except Exception as e:
            print(f"  [WARN] Could not load COSMIC CGC: {e}")

    # ── Focal amp TSV ─────────────────────────────────────────────────────
    focal_tsv = _Path(args.focal_tsv) if getattr(args, "focal_tsv", None) else None

    emergent_tsv = outdir / "emergent_cna_regions.tsv"
    n_ok = 0

    for sa, sb, tf_a_arg, tf_b_arg, pair_label in pairs:

        # Resolve files
        seg_bin_a = seg_bins.get(sa) or seg_bins.get(normalize_sample_name(sa))
        seg_bin_b = seg_bins.get(sb) or seg_bins.get(normalize_sample_name(sb))
        seg_seg_a = seg_segs.get(sa) or seg_segs.get(normalize_sample_name(sa))
        seg_seg_b = seg_segs.get(sb) or seg_segs.get(normalize_sample_name(sb))

        if seg_bin_a is None or seg_bin_b is None:
            missing = ([sa] if seg_bin_a is None else []) + \
                      ([sb] if seg_bin_b is None else [])
            print(f"  [SKIP] {sa} vs {sb}: could not find 50 kbp bin files "
                  f"for: {', '.join(missing)}")
            available = sorted(seg_bins.keys())[:10]
            print(f"         Available sample names (first 10): "
                  f"{', '.join(available)}")
            continue

        # Resolve TFs
        tf_a = tf_a_arg or tf_lookup.get(sa) or tf_lookup.get(
            normalize_sample_name(sa))
        tf_b = tf_b_arg or tf_lookup.get(sb) or tf_lookup.get(
            normalize_sample_name(sb))
        if tf_a is None or tf_b is None:
            missing_tf = ([sa] if tf_a is None else []) + \
                         ([sb] if tf_b is None else [])
            print(f"  [SKIP] {sa} vs {sb}: tumor fraction unknown for "
                  f"{', '.join(missing_tf)}. "
                  f"Provide --tf-a/--tf-b or --dir-1mb for auto-detection.")
            continue

        # Patient and timepoints
        patient = getattr(args, "patient", None) or sa.split("_")[0]
        tp_a    = getattr(args, "timepoint_a", None) or 1
        tp_b    = getattr(args, "timepoint_b", None) or 2

        # Output stem: one file per pair
        stem = f"{sa}_vs_{sb}"
        out_path = outdir / stem

        print(f"\n  Comparing: {sa} (TF={tf_a:.1%})  vs  {sb} (TF={tf_b:.1%})")
        if pair_label:
            print(f"             {pair_label}")

        # Focal amp regions for this patient
        focal_regions: list[dict] = []
        if focal_tsv is not None and focal_tsv.exists():
            try:
                import pandas as pd
                fdf = pd.read_csv(focal_tsv, sep="\t", low_memory=False)
                pat_fdf = fdf[fdf["sample"].isin([sa, sb])]
                if not pat_fdf.empty:
                    focal_regions = [
                        {
                            "chrom":       str(r["chrom"]),
                            "start":       int(r["start"]),
                            "end":         int(r["end"]),
                            "cosmic_genes": str(r.get("cosmic_genes", "")),
                        }
                        for _, r in pat_fdf.iterrows()
                    ]
            except Exception as e:
                print(f"  [WARN] Could not read focal TSV: {e}")

        try:
            plot_longitudinal_combined(
                sample_a             = sa,
                sample_b             = sb,
                seg_path_a           = seg_bin_a,
                seg_path_b           = seg_bin_b,
                seg_path_a_segs      = seg_seg_a,
                seg_path_b_segs      = seg_seg_b,
                tf_a                 = float(tf_a),
                tf_b                 = float(tf_b),
                timepoint_a          = tp_a,
                timepoint_b          = tp_b,
                patient              = patient,
                entity               = "",
                output_path          = out_path,
                focal_regions        = focal_regions,
                max_noise            = args.max_noise,
                min_signal           = args.min_signal,
                min_altered_fraction = args.min_altered_fraction,
                plot_raw             = args.plot_raw,
                emergent_min_bins    = args.emergent_min_bins,
                emergent_min_delta   = args.emergent_min_delta,
                emergent_tsv         = emergent_tsv,
                clean_plots          = not args.no_clean_plots,
            )
            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] {sa} vs {sb}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Done: {n_ok}/{len(pairs)} pair(s) plotted → {outdir}")


def _build_trajectories_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trajectories",
        help="Track copy-number trajectories of focal amplifications across "
             "multiple timepoints per patient. Produces per-patient line plots "
             "(log2 and CN space), genome maps, and overlay profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  cfdna-cnv trajectories --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --cosmic-cgc Census.tsv --output trajectory_results

  # Relax TF filter:
  cfdna-cnv trajectories --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --traj-min-tf 0.07 --traj-min-reads 500000
""",
    )
    _add_common_ichorcna_args(p)
    p.add_argument("--dir-1mb",  default=None,
                   help="Directory containing ichorCNA 1 Mbp run output.")
    p.add_argument("--dir-50kb", default=None,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--output", default="trajectory_results",
                   help="Output directory (default: trajectory_results).")
    _add_focal_amp_args(p)
    p.add_argument("--traj-min-tf", type=float, default=0.10,
                   help="Min TF for trajectory timepoints (default 10%%). "
                        "Note: separate from --tf-threshold which controls "
                        "focal amp detection.")
    p.add_argument("--traj-min-reads", type=int, default=1_000_000,
                   help="Min reads for trajectory timepoints (default 1,000,000).")
    p.add_argument("--traj-max-scale", type=float, default=3.0,
                   help="Max empirical scale factor; larger values excluded "
                        "(default 3.0).")
    p.add_argument("--traj-min-timepoints", type=int, default=2,
                   help="Min qualifying timepoints per patient (default 2).")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip segment file validation.")
    p.add_argument("--plot-bin-size", type=int, default=5_000_000,
                   help="Bin size for broad CNA frequency plot.")
    p.set_defaults(func=_run_trajectories)


def _run_trajectories(args: argparse.Namespace) -> None:
    """Delegate to ichorcna_cmd.run with longitudinal + trajectory flags set."""
    # Map trajectory-specific args to the internal names expected by ichorcna_cmd
    args.longitudinal_plots    = True
    args.longitudinal_plot_dir = None
    args.max_noise             = 0.15
    args.min_signal            = None
    args.min_altered_fraction  = 0.05
    args.plot_raw              = False
    args.emergent_min_bins     = 3
    args.emergent_min_delta    = 0.5
    args.no_clean_plots        = False
    args.sigminer              = False
    args.liquidcna             = False
    # traj-specific args get picked up directly by plot_all_focal_amp_trajectories
    # via the ichorcna_cmd run function
    from .commands.ichorcna_cmd import run
    run(args)


# ---------------------------------------------------------------------------
# signatures — CN signature extraction via sigminer
# ---------------------------------------------------------------------------

def _build_validate_tf_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "validate-tf",
        help="Validate tumor fraction estimates against a known in-silico "
             "dilution series. Compares ichorCNA TF, scaling-based TF, and "
             "expected (known) TF across replicates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Metadata TSV format (tab-separated):
    sample          sample name matching ichorCNA output file names
    expected_tf     known dilution fraction (0–1), e.g. 0.50, 0.25, 0.125
    series          identifier grouping dilutions from the same original sample
    replicate       (optional) replicate number; auto-assigned if omitted
    label           (optional) free-text label for the plot

Example:
    sample              expected_tf  series    replicate
    P12_dil50_r1        0.50         P12       1
    P12_dil50_r2        0.50         P12       2
    P12_dil25_r1        0.25         P12       1
    P12_undil_r1        0.82         P12       1

    The highest expected_tf sample in each series is treated as the
    reference for the empirical scaling step.

Examples:
  cfdna-cnv validate-tf \\
      --metadata dilution_metadata.tsv \\
      --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --output tf_validation

  # With a lower informative-bin threshold:
  cfdna-cnv validate-tf \\
      --metadata dilution_metadata.tsv \\
      --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --min-abs 0.10 --output tf_validation
""",
    )
    p.add_argument("--metadata", required=True, metavar="TSV",
                   help="Dilution metadata TSV (sample, expected_tf, series, "
                        "optional: replicate, label).")
    p.add_argument("--dir-1mb", required=True,
                   help="ichorCNA 1 Mbp run directory (for .params.txt TF).")
    p.add_argument("--dir-50kb", required=True,
                   help="ichorCNA 50 kbp run directory (for scaling-based TF).")
    p.add_argument("--output", default="tf_validation",
                   help="Output directory (default: tf_validation).")
    p.add_argument("--min-abs", type=float, default=0.05,
                   help="Min |log2R| for bins included in WLS fit (default 0.05).")
    p.set_defaults(func=_run_validate_tf)


def _run_validate_tf(args: argparse.Namespace) -> None:
    from pathlib import Path as _Path
    from .validate_tf import run_tf_validation

    run_tf_validation(
        meta_tsv   = _Path(args.metadata),
        dir_1mb    = _Path(args.dir_1mb),
        dir_50kb   = _Path(args.dir_50kb),
        output_dir = _Path(args.output),
        min_abs    = args.min_abs,
    )


def _build_gene_trajectories_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "gene-trajectories",
        help="Gene- or region-centric CN trajectories: find all patients "
             "with a given gene amplified (or overlapping a genomic region) "
             "and plot CN trajectories across their timepoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # By gene name:
  cfdna-cnv gene-trajectories \\
      --dir-50kb /runs/50kb --dir-1mb /runs/1mb \\
      --gene AR --entity Prostate \\
      --focal-tsv results/focal_amplifications.tsv \\
      --output gene_trajectories

  # By explicit region (chrX:66500000-67500000):
  cfdna-cnv gene-trajectories \\
      --dir-50kb /runs/50kb --dir-1mb /runs/1mb \\
      --region chrX:66500000-67500000 \\
      --focal-tsv results/focal_amplifications.tsv \\
      --output gene_trajectories

  # Region with custom label and entity filter:
  cfdna-cnv gene-trajectories \\
      --dir-50kb /runs/50kb --dir-1mb /runs/1mb \\
      --region chrX:66500000-67500000 --region-label AR_locus \\
      --entity Prostate \\
      --focal-tsv results/focal_amplifications.tsv \\
      --output gene_trajectories

  # Multiple genes:
  cfdna-cnv gene-trajectories \\
      --dir-50kb /runs/50kb --dir-1mb /runs/1mb \\
      --gene EGFR MYC CCND1 \\
      --focal-tsv results/focal_amplifications.tsv \\
      --output gene_trajectories
""",
    )
    _add_common_ichorcna_args(p)
    p.add_argument("--dir-50kb", default=None,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--dir-1mb",  default=None,
                   help="Directory containing ichorCNA 1 Mbp run output "
                        "(used for TF lookup).")
    p.add_argument("--focal-tsv", default=None, metavar="FILE",
                   help="Path to focal_amplifications.tsv. Required when "
                        "using --gene; optional with --region.")
    p.add_argument("--tf-tsv", default=None, metavar="FILE",
                   help="Path to tf_summary.tsv (for TF and timepoint lookup). "
                        "If omitted, requires --dir-1mb.")

    # ── What to plot ─────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--gene", nargs="+", metavar="GENE",
                     help="Gene name(s) to plot (e.g. AR EGFR MYC). "
                          "Case-insensitive, partial match against "
                          "focal_amplifications.tsv cosmic_genes column.")
    grp.add_argument("--region", metavar="CHR:START-END",
                     help="Explicit genomic region, e.g. chrX:66500000-67500000. "
                          "All patients with a focal amp overlapping this region "
                          "are included; CN is measured in this window for ALL "
                          "patients (not patient-specific coords).")

    p.add_argument("--region-label", default=None, metavar="LABEL",
                   help="Label to use in plot titles and filenames when "
                        "--region is specified (default: the region string).")
    p.add_argument("--entity", default=None, metavar="ENTITY",
                   help="Restrict to patients from this entity "
                        "(e.g. Prostate, Breast, Colorectal). "
                        "Case-insensitive. Default: all entities.")
    p.add_argument("--output", default="gene_trajectories",
                   help="Output directory (default: gene_trajectories).")
    p.add_argument("--min-tf", type=float, default=0.10,
                   help="Min tumor fraction per timepoint (default 10%%).")
    p.add_argument("--traj-min-reads", type=int, default=1_000_000,
                   help="Min reads per timepoint (default 1,000,000).")
    p.add_argument("--traj-max-scale", type=float, default=3.0,
                   help="Max empirical scale factor (default 3.0).")
    p.add_argument("--min-timepoints", type=int, default=2,
                   help="Min qualifying timepoints per patient (default 2).")
    p.set_defaults(func=_run_gene_trajectories)


def _run_gene_trajectories(args: argparse.Namespace) -> None:
    import re as _re
    from pathlib import Path as _Path
    import pandas as pd

    from .common.io import find_sample_files, collect_tumor_fractions
    from .ichorcna.longitudinal_plots import plot_gene_focal_trajectories

    outdir = _Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load tf_df ────────────────────────────────────────────────────────
    tf_tsv = getattr(args, "tf_tsv", None)
    if tf_tsv and _Path(tf_tsv).exists():
        tf_df = pd.read_csv(_Path(tf_tsv), sep="\t")
        print(f"  Loaded tf_summary from {tf_tsv} ({len(tf_df)} rows)")
    elif getattr(args, "dir_1mb", None):
        dir_1mb      = _Path(args.dir_1mb)
        params_files = find_sample_files(dir_1mb, ".params.txt")
        wig_files    = find_sample_files(dir_1mb, ".wig")
        tf_df        = collect_tumor_fractions(params_files, wig_files)
        print(f"  Collected TF for {len(tf_df)} samples from {dir_1mb}")
    else:
        print("  [ERROR] Provide either --tf-tsv or --dir-1mb to load TF data.")
        sys.exit(1)

    # ── Discover 50 kbp seg files ─────────────────────────────────────────
    if not getattr(args, "dir_50kb", None):
        print("  [ERROR] --dir-50kb is required.")
        sys.exit(1)
    dir_50kb       = _Path(args.dir_50kb)
    seg_files_bins = find_sample_files(dir_50kb, ".cna.seg")
    seg_files_segs = find_sample_files(dir_50kb, ".seg.txt")
    print(f"  Found {len(seg_files_bins)} 50 kbp bin files.")

    focal_tsv = _Path(args.focal_tsv) if getattr(args, "focal_tsv", None) else None

    common_kw = dict(
        tf_df          = tf_df,
        seg_files_bins = seg_files_bins,
        output_dir     = outdir,
        seg_files_segs = seg_files_segs,
        entity_filter  = args.entity,
        min_tf         = args.min_tf,
        min_reads      = getattr(args, "traj_min_reads", 1_000_000),
        max_scale      = getattr(args, "traj_max_scale", 3.0),
        min_timepoints = args.min_timepoints,
    )

    # ── Gene mode ─────────────────────────────────────────────────────────
    if args.gene:
        if focal_tsv is None:
            print("  [ERROR] --focal-tsv is required when using --gene.")
            sys.exit(1)
        for gene in args.gene:
            print(f"\n  === Gene: {gene} ===")
            n = plot_gene_focal_trajectories(
                gene      = gene,
                focal_tsv = focal_tsv,
                **common_kw,
            )
            print(f"  {gene}: {n} patient(s) plotted → {outdir}")

    # ── Region mode ───────────────────────────────────────────────────────
    elif args.region:
        region_str = args.region.strip()
        # Parse chrX:66500000-67500000 or chrX:66.5Mb-67.5Mb
        m = _re.match(
            r"^(chr[0-9XYM]+):([0-9]+(?:\.[0-9]+)?(?:Mb|Kb)?)"
            r"-([0-9]+(?:\.[0-9]+)?(?:Mb|Kb)?)$",
            region_str, _re.IGNORECASE,
        )
        if not m:
            print(f"  [ERROR] Could not parse --region '{region_str}'. "
                  f"Expected format: chrX:66500000-67500000 or chrX:66.5Mb-67.5Mb")
            sys.exit(1)

        def _parse_pos(s: str) -> int:
            s = s.strip()
            if s.lower().endswith("mb"):
                return int(float(s[:-2]) * 1_000_000)
            if s.lower().endswith("kb"):
                return int(float(s[:-2]) * 1_000)
            return int(s)

        chrom = m.group(1)
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        start = _parse_pos(m.group(2))
        end   = _parse_pos(m.group(3))
        label = args.region_label or region_str.replace(":", "_").replace("-", "_")

        print(f"\n  === Region: {chrom}:{start:,}-{end:,} (label: {label}) ===")
        n = plot_gene_focal_trajectories(
            gene           = label,
            focal_tsv      = focal_tsv,
            fixed_region   = {"chrom": chrom, "start": start, "end": end,
                               "cosmic_genes": label},
            **common_kw,
        )
        print(f"  Region {label}: {n} patient(s) plotted → {outdir}")


def _build_signatures_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "signatures",
        help="Extract copy-number signatures from ichorCNA output using "
             "sigminer (R). Performs de novo NMF extraction and maps to "
             "COSMIC CN reference signatures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Requires R packages: r-sigminer r-ggplot2 r-ggplotify r-dplyr (add to pixi.toml)

Examples:
  cfdna-cnv signatures --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --output signature_results

  # Extract 8 signatures and compare to multiple references:
  cfdna-cnv signatures --dir-1mb /runs/1mb --dir-50kb /runs/50kb \\
      --sig-n-sig 8 --sig-ref CNS_TCGA CNS_PCAWG176
""",
    )
    _add_common_ichorcna_args(p)
    p.add_argument("--dir-1mb",  default=None,
                   help="Directory containing ichorCNA 1 Mbp run output.")
    p.add_argument("--dir-50kb", default=None,
                   help="Directory containing ichorCNA 50 kbp run output.")
    p.add_argument("--output", default="signature_results",
                   help="Output directory (default: signature_results).")
    p.add_argument("--sig-min-tf", type=float, default=0.15,
                   help="Min TF to include a sample (default 15%%).")
    p.add_argument("--sig-ref", nargs="+", default=["CNS_TCGA"], metavar="REF",
                   help="Reference signature sets. Options: CNS_TCGA, "
                        "CNS_USARC, CNS_TCGA176, CNS_PCAWG176 "
                        "(default: CNS_TCGA).")
    p.add_argument("--sig-n-sig", type=int, default=6,
                   help="Number of de novo signatures to extract (default 6).")
    p.add_argument("--sig-genome", default="hg38",
                   choices=["hg38", "hg19"],
                   help="Reference genome (default hg38).")
    p.add_argument("--sig-cores", type=int, default=1,
                   help="Parallel cores for NMF (default 1).")
    p.add_argument("--sig-no-run", action="store_true",
                   help="Write R script only, do not execute Rscript.")
    _add_focal_amp_args(p)
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip segment file validation.")
    p.add_argument("--plot-bin-size", type=int, default=5_000_000,
                   help="Bin size for broad CNA frequency plot.")
    p.set_defaults(func=_run_signatures)


def _run_signatures(args: argparse.Namespace) -> None:
    """Delegate to ichorcna_cmd.run with sigminer enabled."""
    # Map sig-* args → internal sigminer_* names
    args.sigminer         = True
    args.sigminer_ref     = args.sig_ref
    args.sigminer_min_tf  = args.sig_min_tf
    args.sigminer_genome  = args.sig_genome
    args.sigminer_cores   = args.sig_cores
    args.sigminer_n_sig   = args.sig_n_sig
    args.sigminer_no_run  = args.sig_no_run
    args.longitudinal_plots   = False
    args.longitudinal_plot_dir = None
    args.max_noise            = 0.15
    args.min_signal           = None
    args.min_altered_fraction = 0.05
    args.plot_raw             = False
    args.emergent_min_bins    = 3
    args.emergent_min_delta   = 0.5
    args.no_clean_plots       = True
    args.liquidcna            = False
    from .commands.ichorcna_cmd import run
    run(args)


# ---------------------------------------------------------------------------
# cbio — cBioPortal analysis (unchanged)
# ---------------------------------------------------------------------------

def _build_cbio_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "cbio",
        help="Analyze cBioPortal segmentation files (broad CNA + focal amps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, metavar="FILE",
                   help="YAML config file. CLI arguments override config values.")
    p.add_argument("--write-config", default=None, metavar="FILE",
                   help="Write a default config YAML to FILE and exit.")
    p.add_argument("--breast",     default=None,
                   help="Breast cancer .seg file(s), semicolon-separated.")
    p.add_argument("--prostate",   default=None,
                   help="Prostate cancer .seg file(s), semicolon-separated.")
    p.add_argument("--colorectal", default=None,
                   help="Colorectal cancer .seg file(s), semicolon-separated.")
    p.add_argument("--seg", nargs="+", metavar="NAME=FILE", default=[],
                   help="Additional entities as NAME=FILE.")
    p.add_argument("--output", default="cbio_cna_results",
                   help="Output directory.")
    p.add_argument("--bin-size",          type=int,   default=5_000_000)
    p.add_argument("--gain-threshold",    type=float, default=0.2)
    p.add_argument("--loss-threshold",    type=float, default=-0.2)
    p.add_argument("--min-seg-size",      type=int,   default=0)
    p.add_argument("--focal-min-bp",      type=int,   default=50_000)
    p.add_argument("--focal-max-bp",      type=int,   default=500_000)
    p.add_argument("--focal-log2-min",    type=float, default=0.5)
    p.add_argument("--focal-log2-delta",  type=float, default=0.2)
    p.add_argument("--focal-flank-bp",    type=int,   default=10_000_000)
    p.add_argument("--focal-min-freq",    type=float, default=0.01)
    p.add_argument("--liftover",          action="store_true")
    p.add_argument("--chain-file",        default=None, metavar="FILE")
    p.add_argument("--liftover-bin",      default="liftOver", metavar="PATH")
    p.add_argument("--liftover-min-match",type=float, default=0.95, metavar="FRAC")
    _add_cosmic_arg(p)
    p.set_defaults(func=_run_cbio)


def _run_ichorcna(args: argparse.Namespace) -> None:
    from .commands.ichorcna_cmd import run
    run(args)


def _run_cbio(args: argparse.Namespace) -> None:
    from .commands.cbio_cmd import run
    run(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cfdna-cnv",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")
    sub.required = True
    _build_ichorcna_parser(sub)
    _build_cbio_parser(sub)
    _build_focal_amps_parser(sub)
    _build_compare_parser(sub)
    _build_compare_samples_parser(sub)
    _build_trajectories_parser(sub)
    _build_gene_trajectories_parser(sub)
    _build_signatures_parser(sub)
    _build_validate_tf_parser(sub)

    args = parser.parse_args()

    # ── --write-config: dump default and exit ────────────────────────────
    if getattr(args, "write_config", None):
        from .config import write_default_config
        cmd = args.command if args.command in ("ichorcna", "cbio") else "ichorcna"
        write_default_config(Path(args.write_config), cmd)
        sys.exit(0)

    # ── --config: load YAML and apply values not set on the CLI ─────────
    if getattr(args, "config", None):
        from .config import load_config, apply_config_to_namespace
        cfg        = load_config(Path(args.config))
        sub_parser = sub._name_parser_map[args.command]
        args       = apply_config_to_namespace(args, cfg, sub_parser)

    # ── Validate required inputs after config merge ──────────────────────
    _NEEDS_1MB  = {"ichorcna", "focal-amps", "compare",
                   "trajectories", "signatures"}
    _NEEDS_50KB = {"ichorcna", "focal-amps", "compare",
                   "trajectories", "signatures", "compare-samples",
                   "gene-trajectories"}
    missing = []
    if args.command in _NEEDS_1MB  and not getattr(args, "dir_1mb",  None):
        missing.append("--dir-1mb")
    if args.command in _NEEDS_50KB and not getattr(args, "dir_50kb", None):
        missing.append("--dir-50kb")
    if missing:
        parser.error(
            f"The following arguments are required (or set in --config): "
            f"{', '.join(missing)}"
        )

    args.func(args)


if __name__ == "__main__":
    main()