"""
cfdna-cnv  —  cfDNA copy-number analysis toolkit

Subcommands:
    ichorcna   Analyze ichorCNA output (broad CNA + focal amps + longitudinal)
    cbio       Analyze cBioPortal segmentation files (broad CNA + focal amps)
"""
from __future__ import annotations

import argparse
import sys


def _add_cosmic_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cosmic-cgc", default=None, metavar="FILE",
        help="COSMIC Cancer Gene Census CSV/TSV for focal amp annotation "
             "(download from https://cancer.sanger.ac.uk/census).",
    )


# ---------------------------------------------------------------------------
# ichorcna sub-command
# ---------------------------------------------------------------------------

def _build_ichorcna_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ichorcna",
        help="Analyze ichorCNA output directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--dir-1mb",  required=True,
                   help="Directory containing ichorCNA 1 Mbp run output.")
    p.add_argument("--dir-50kb", required=True,
                   help="Directory containing ichorCNA 50 kbp run output.")
    # Output
    p.add_argument("--output", default="ichorcna_results",
                   help="Output directory.")
    # Tumor fraction
    p.add_argument("--tf-threshold", type=float, default=0.03,
                   help="Minimum tumor fraction (default 3%%).")
    p.add_argument("--exclude-entities", nargs="+", default=[],
                   metavar="ENTITY",
                   help="Entity names to exclude from all analysis "
                        "(e.g. --exclude-entities Lung Ovarian). "
                        "Entity names are case-sensitive and must match "
                        "those derived by entity_from_sample "
                        "(Breast, Colorectal, Prostate, Lung, Ovarian, Other).")
    p.add_argument("--min-reads", type=int, default=1_000_000,
                   help="Minimum total reads from WIG file to include a "
                        "sample (default 1,000,000). Samples below this "
                        "threshold are excluded before TF filtering. "
                        "Set to 0 to disable.")
    # Broad CNA
    p.add_argument("--amp-min-cn", type=int, default=4,
                   help="Minimum copy number to call focal amplification "
                        "(default 4; GAIN/CN=3 is excluded).")
    # Focal amp
    p.add_argument("--focal-min-bp", type=int, default=50_000,
                   help="Minimum focal amp segment size in bp.")
    p.add_argument("--focal-max-bp", type=int, default=500_000,
                   help="Maximum focal amp segment size in bp.")
    p.add_argument("--focal-min-bins", type=int, default=3,
                   help="Minimum 50 kbp bins in a segment (default 3 = 150 kbp).")
    p.add_argument("--focal-flank-bins", type=int, default=2,
                   help="Number of 1 Mbp bins per side for local background "
                        "(default 2 = 2 Mbp window on each side).")
    p.add_argument("--max-focal-per-sample", type=int, default=5,
                   help="Discard samples with more focal amps than this (default 5).")
    p.add_argument("--focal-noise-factor", type=float, default=2.0,
                   help="Focal amp segment logR must be >= this multiple of "
                        "the sample neighbour noise (default 2.0).")
    p.add_argument("--focal-freq-min-freq", type=float, default=0.0,
                   help="Minimum frequency to show in the focal amp "
                        "frequency plot (default 0.0 = show all). "
                        "Set e.g. to 0.01 to restrict to events in "
                        "at least 1%% of patients.")
    p.add_argument("--focal-amp-diagnostics", action="store_true",
                   help="Generate per-sample diagnostic plots for all samples "
                        "with focal amp candidates (including discarded samples). "
                        "Plots written to <output>/focal_amp_diagnostics/.")
    p.add_argument("--log2-delta", type=float, default=0.2,
                   help="Min log2 elevation above local background.")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip segment file validation (step 4/7).")
    # Plotting
    p.add_argument("--plot-bin-size", type=int, default=5_000_000,
                   help="Bin size in bp for the broad CNA frequency plot "
                        "(default 5 Mbp). Smaller = higher resolution.")
    p.add_argument("--longitudinal-plots", action="store_true",
                   help="Generate one genome-wide pairwise CN comparison plot "
                        "per consecutive timepoint pair per patient. "
                        "Can produce many files for large cohorts.")
    p.add_argument("--longitudinal-plot-dir", default=None, metavar="DIR",
                   help="Output directory for longitudinal pairwise plots "
                        "(default: <output>/longitudinal_plots/).")
    p.add_argument("--max-noise", type=float, default=0.15,
                   help="Maximum per-sample neighbour noise (median absolute "
                        "consecutive bin difference) allowed for longitudinal "
                        "plots. Pairs where either sample exceeds this "
                        "threshold are skipped (default 0.15). Set to 0 to "
                        "disable filtering.")
    p.add_argument("--min-signal", type=float, default=None,
                   help="Minimum SNR (p95/noise) for longitudinal plots. "
                        "Pairs below this threshold are skipped "
                        "(default: no filter). Suggested: 1.5.")
    p.add_argument("--min-altered-fraction", type=float, default=0.05,
                   help="Minimum fraction of 50 kbp bins (after 10-bin "
                        "smoothing) with |log2R| > 0.2 required for both "
                        "samples in a longitudinal pair (default 0.05 = 5%%). "
                        "Pairs where either sample falls below this are "
                        "considered flat and skipped.")
    p.add_argument("--plot-raw", action="store_true",
                   help="Also output an unsmoothed longitudinal plot in "
                        "addition to the 5-bin and 10-bin smoothed versions "
                        "(default: off).")
    p.add_argument("--emergent-min-bins", type=int, default=3,
                   help="Minimum consecutive bins with |delta| > threshold "
                        "to call an emergent CNA region (default 3).")
    p.add_argument("--emergent-min-delta", type=float, default=0.2,
                   help="Minimum |delta log2R| per bin to count toward "
                        "an emergent CNA region (default 0.2).")
    # Annotation
    _add_cosmic_arg(p)
    p.set_defaults(func=_run_ichorcna)


def _run_ichorcna(args: argparse.Namespace) -> None:
    from .commands.ichorcna_cmd import run
    run(args)


# ---------------------------------------------------------------------------
# cbio sub-command
# ---------------------------------------------------------------------------

def _build_cbio_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "cbio",
        help="Analyze cBioPortal segmentation files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input — built-in entities (semicolon-separated for multiple files)
    p.add_argument("--breast",     default=None,
                   help="Breast cancer .seg file(s), semicolon-separated.")
    p.add_argument("--prostate",   default=None,
                   help="Prostate cancer .seg file(s), semicolon-separated.")
    p.add_argument("--colorectal", default=None,
                   help="Colorectal cancer .seg file(s), semicolon-separated.")
    p.add_argument("--seg", nargs="+", metavar="NAME=FILE", default=[],
                   help="Additional entities as NAME=FILE (semicolons for "
                        "multiple files per entity).")
    # Output
    p.add_argument("--output", default="cbio_cna_results",
                   help="Output directory.")
    # Broad CNA
    p.add_argument("--bin-size",        type=int,   default=5_000_000,
                   help="Genomic bin size in bp.")
    p.add_argument("--gain-threshold",  type=float, default=0.2,
                   help="seg.mean threshold for GAIN call.")
    p.add_argument("--loss-threshold",  type=float, default=-0.2,
                   help="seg.mean threshold for LOSS call.")
    p.add_argument("--min-seg-size",    type=int,   default=0,
                   help="Minimum segment size in bp (0 = no filter).")
    # Focal amp
    p.add_argument("--focal-min-bp",    type=int,   default=50_000,
                   help="Minimum focal amp segment size in bp.")
    p.add_argument("--focal-max-bp",    type=int,   default=500_000,
                   help="Maximum focal amp segment size in bp.")
    p.add_argument("--focal-log2-min",  type=float, default=0.5,
                   help="Minimum absolute seg.mean for focal amp.")
    p.add_argument("--focal-log2-delta",type=float, default=0.2,
                   help="Min elevation above local background.")
    p.add_argument("--focal-flank-bp",  type=int,   default=10_000_000,
                   help="Flanking window (bp) for local background.")
    p.add_argument("--focal-min-freq",  type=float, default=0.01,
                   help="Minimum frequency to include in focal amp plot.")
    # LiftOver (hg19 → hg38)
    p.add_argument("--liftover", action="store_true",
                   help="Lift coordinates from hg19 to hg38 before analysis.")
    p.add_argument("--chain-file", default=None, metavar="FILE",
                   help="Path to hg19ToHg38.over.chain.gz. Required when "
                        "--liftover is set. Download from: "
                        "https://hgdownload.gi.ucsc.edu/goldenPath/hg19/"
                        "liftOver/hg19ToHg38.over.chain.gz")
    p.add_argument("--liftover-bin", default="liftOver", metavar="PATH",
                   help="Path to the UCSC liftOver binary (default: liftOver "
                        "on PATH). Download from: https://hgdownload.soe.ucsc"
                        ".edu/admin/exe/linux.x86_64/liftOver")
    p.add_argument("--liftover-min-match", type=float, default=0.95,
                   metavar="FRAC",
                   help="Minimum fraction of bases that must remap (default "
                        "0.95). Lower to 0.1 to keep more segments.")
    # Annotation
    _add_cosmic_arg(p)
    p.set_defaults(func=_run_cbio)


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()