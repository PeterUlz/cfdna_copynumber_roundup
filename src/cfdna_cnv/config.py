"""
Configuration file support for cfdna-cnv.

Config files are YAML and map directly to CLI argument names.
CLI arguments always override config file values.

Usage:
    cfdna-cnv locate --config analysis.yaml
    cfdna-cnv locate --config analysis.yaml --tf-threshold 0.05  # override one
    cfdna-cnv locate --write-config analysis.yaml                 # dump defaults
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Explicit mapping from flattened YAML key → argparse dest.
# Only needed where the YAML section-key combination doesn't directly match
# the CLI argument name after replacing hyphens with underscores.
_ALIASES: dict[str, str] = {
    "tf-min-reads":                      "min_reads",
    "tf-exclude-entities":               "exclude_entities",
    "focal-max-focal-per-sample":        "max_focal_per_sample",
    "focal-amp-diagnostics":             "focal_amp_diagnostics",
    "longitudinal-plots":                "longitudinal_plots",
    "longitudinal-plot-dir":             "longitudinal_plot_dir",
    "longitudinal-max-noise":            "max_noise",
    "longitudinal-min-signal":           "min_signal",
    "longitudinal-min-altered-fraction": "min_altered_fraction",
    "longitudinal-plot-raw":             "plot_raw",
    "longitudinal-emergent-min-bins":    "emergent_min_bins",
    "longitudinal-emergent-min-delta":   "emergent_min_delta",
    "longitudinal-no-clean-plots":       "no_clean_plots",
    "amp-min-cn":                        "amp_min_cn",
    "log2-delta":                        "log2_delta",
    "plot-bin-size":                     "plot_bin_size",
    "cosmic-cgc":                        "cosmic_cgc",
    "liquidcna-functions":               "liquidcna_functions",
    "liquidcna-min-tf":                  "liquidcna_min_tf",
    "liquidcna-min-timepoints":          "liquidcna_min_timepoints",
    "liquidcna-no-run":                  "liquidcna_no_run",
    "sigminer-ref":                      "sigminer_ref",
    "sigminer-min-tf":                   "sigminer_min_tf",
    "sigminer-genome":                   "sigminer_genome",
    "sigminer-cores":                    "sigminer_cores",
    "sigminer-no-run":                   "sigminer_no_run",
}


def load_config(config_path: Path) -> dict:
    """Load a YAML config file and return as a flat dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for --config support. "
            "Install with: pip install pyyaml"
        )
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Flatten nested sections into a single dict of hyphen-separated keys
    flat: dict = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for key, val in values.items():
                flat[f"{section}-{key}"] = val
        else:
            flat[section] = values
    return flat


def apply_config_to_namespace(
        args:    argparse.Namespace,
        config:  dict,
        parser:  argparse.ArgumentParser,
) -> argparse.Namespace:
    """
    Apply config values to a parsed Namespace, without overriding any
    value the user explicitly passed on the command line.
    """
    defaults = {a.dest: a.default for a in parser._actions}

    for cfg_key, value in config.items():
        # Resolve dest: check alias map first, then simple replacement
        dest = _ALIASES.get(cfg_key, cfg_key.replace("-", "_"))
        if not hasattr(args, dest):
            continue
        # Only apply if the current value is still the argparse default
        if getattr(args, dest) == defaults.get(dest):
            setattr(args, dest, value)

    return args


def write_default_config(path: Path, subcommand: str = "locate") -> None:
    """Write a well-commented default config file for the given subcommand."""
    if subcommand == "locate":
        content = _LOCATE_DEFAULT_CONFIG
    elif subcommand == "cbio":
        content = _CBIO_DEFAULT_CONFIG
    else:
        raise ValueError(f"Unknown subcommand: {subcommand}")
    path.write_text(content)
    print(f"Default config written to: {path}")


# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

_LOCATE_DEFAULT_CONFIG = """\
# cfdna-cnv locate configuration
# =================================
# Generate this file with:  cfdna-cnv locate --write-config analysis.yaml
# Run with:                 cfdna-cnv locate --config analysis.yaml
#
# CLI arguments always override values here, e.g.:
#   cfdna-cnv locate --config analysis.yaml --tf-threshold 0.05

# ── Input / output ──────────────────────────────────────────────────────────
dir-1mb:  /path/to/ichorCNA_1Mbp_output    # required
dir-50kb: /path/to/ichorCNA_50kbp_output   # required
output:   locate_results

# ── Sample filtering ─────────────────────────────────────────────────────────
tf:
  threshold:  0.03        # Minimum tumor fraction (3%)
  min-reads:  1000000     # Minimum total reads from WIG file
  # exclude-entities: [Lung, Ovarian]  # Remove small entity cohorts

# ── Copy-number thresholds ───────────────────────────────────────────────────
amp:
  min-cn: 4               # Min CN for focal amp (excludes GAIN/CN=3)

log2:
  delta: 0.2              # Min log2 elevation above background (abs + relative)

# ── Focal amplification detection ───────────────────────────────────────────
focal:
  min-bp:               50000   # Minimum segment size (50 kbp)
  max-bp:               500000  # Maximum segment size (500 kbp)
  min-bins:             3       # Minimum 50 kbp bins in segment (≥ 150 kbp)
  flank-bins:           2       # Background window: 2 × 1 Mbp on each side
  noise-factor:         2.0     # Segment logR must be ≥ noise_factor × noise
  max-focal-per-sample: 5       # Discard samples with > N focal amps
  freq-min-freq:        0.0     # Min patient frequency for frequency plot
  amp-diagnostics:      false   # Generate per-sample diagnostic plots

# ── Broad CNA plot ───────────────────────────────────────────────────────────
plot:
  bin-size: 5000000       # Genomic bin size for broad CNA frequency plot (5 Mbp)

# ── COSMIC annotation ────────────────────────────────────────────────────────
# cosmic-cgc: /path/to/Census_allThu.tsv

# ── Sigminer CN signatures ───────────────────────────────────────────────────
sigminer: false                   # Set to true to enable
sigminer-ref: [CNS_TCGA]  # Only CNS_TCGA works without minor_cn (LOH)
                               # CNS_TCGA/CNS_TCGA176/CNS_PCAWG176 require LOH
sigminer-min-tf:      0.15        # Min TF for samples (15%)
sigminer-genome:      hg38        # hg38 or hg19
sigminer-cores:       1           # Parallel cores
sigminer-no-run:      false       # Prepare R script but don't execute

# ── LiquidCNA subclonal evolution ────────────────────────────────────────────
liquidcna: false              # Set to true to enable
# liquidcna-functions: /path/to/mixture_estimation_functions.R  # required
liquidcna-min-tf:           0.20  # Min TF to include in liquidCNA (20%)
liquidcna-min-timepoints:   2     # Min qualifying timepoints per patient
liquidcna-no-run:           false # Prepare inputs but don't execute Rscript

# ── Longitudinal plots ───────────────────────────────────────────────────────
longitudinal:
  plots:                true    # Enable longitudinal comparison plots
  # plot-dir: null              # Output dir (default: <output>/longitudinal_plots)
  max-noise:            0.15    # Max neighbour noise; noisier pairs are skipped
  min-signal:           null    # Min SNR (p95/noise); null = disabled
  min-altered-fraction: 0.05    # Min fraction of bins with |logR| > 0.2
  plot-raw:             false   # Also output unsmoothed plots
  emergent-min-bins:    3       # Min consecutive bins to call emergent CNA
  emergent-min-delta:   0.5     # Min |delta log2R| per bin for emergent CNA
  no-clean-plots:       false   # Set true to skip 'clean' figure-ready plots
"""

_CBIO_DEFAULT_CONFIG = """\
# cfdna-cnv cbio configuration
# ============================
# Generate with: cfdna-cnv cbio --write-config cbio.yaml

# ── Input seg files per entity ───────────────────────────────────────────────
# breast:     /path/to/breast.seg
# prostate:   /path/to/prostate.seg
# colorectal: /path/to/colorectal.seg

output: cbio_cna_results

# ── Broad CNA ────────────────────────────────────────────────────────────────
bin:
  size: 5000000           # Genomic bin size (5 Mbp)

gain:
  threshold: 0.2          # log2 gain threshold

loss:
  threshold: -0.2         # log2 loss threshold

min:
  seg-size: 0             # Minimum segment size in bp (0 = no filter)

# ── Focal amplifications ─────────────────────────────────────────────────────
focal:
  min-bp:     50000       # Minimum segment size
  max-bp:     500000      # Maximum segment size
  log2-min:   0.5         # Minimum seg.mean log2 ratio
  log2-delta: 0.3         # Minimum elevation above background
  flank-bp:   10000000    # Background window (10 Mbp)
  min-freq:   0.0         # Minimum frequency for frequency plot

# ── Liftover (hg19 → hg38) ───────────────────────────────────────────────────
# liftover: false
# chain-file:         /path/to/hg19ToHg38.over.chain.gz
# liftover-bin:       /path/to/liftOver
# liftover-min-match: 0.95

# ── COSMIC annotation ────────────────────────────────────────────────────────
# cosmic-cgc: /path/to/Census_allThu.tsv
"""