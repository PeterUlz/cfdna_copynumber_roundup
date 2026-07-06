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
    _build_trajectories_parser(sub)
    _build_signatures_parser(sub)

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
                   "trajectories", "signatures"}
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