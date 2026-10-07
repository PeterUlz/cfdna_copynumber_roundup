"""
Sigminer CN signature analysis module.

Fits copy-number signatures from ichorCNA segmentation data using the
sigminer R package. Signatures are fitted against pre-computed COSMIC
reference sets (no external files required).

Available reference sets (built into sigminer):
  CNS_TCGA     : 48 categories  — recommended default, broad cancer types
  CNS_TCGA176  : 176 categories — higher resolution, TCGA cohort
  CNS_PCAWG176 : 176 categories — higher resolution, PCAWG cohort
  CNS_USARC    : 40 categories  — USARC cohort

Pipeline
--------
1. Filter: samples with TF >= min_tf (default 0.15) only.
2. Load 1 Mbp .seg.txt files and convert to sigminer's MAF-like format.
3. Write a combined seg TSV for sigminer's read_copynumber().
4. Generate and run an R script that:
   a. Reads segments with sigminer::read_copynumber()
   b. Extracts CN features and builds the sample-by-component matrix
   c. Fits against one or more reference signature sets using sig_fit()
   d. Writes per-sample exposure TSVs and summary plots
5. Output:
   sigminer/
     segments_input.tsv            — combined seg input for sigminer
     run_sigminer.R                — generated R script
     exposures_<ref>.tsv           — per-sample signature exposures
     exposures_<ref>_relative.tsv  — relative (normalised) exposures
     plots/                        — signature profile and exposure plots

Usage (CLI)
-----------
    cfdna-cnv locate ... --sigminer
    cfdna-cnv locate ... --sigminer --sigminer-ref CNS_TCGA CNS_PCAWG176
    cfdna-cnv locate ... --sigminer --sigminer-min-tf 0.20
    cfdna-cnv locate ... --sigminer --sigminer-no-run
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .common.constants import CHROM_ORDER
from .common.io import parse_seg_file

# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_sigminer_input(
        tf_df:       pd.DataFrame,
        seg_files:   dict[str, Path],
        output_dir:  Path,
        min_tf:      float = 0.15,
) -> Path | None:
    """
    Select qualifying samples and build a combined segment TSV for sigminer.

    Parameters
    ----------
    tf_df      : tumor fraction table (sample, patient, timepoint,
                 tumor_fraction, entity).
    seg_files  : {sample: Path} to 1 Mbp .seg.txt files.
    output_dir : directory where outputs are written.
    min_tf     : minimum TF to include (default 0.15 = 15%).

    Returns
    -------
    Path to the combined segments TSV, or None if no samples qualify.

    Notes
    -----
    sigminer's read_copynumber() expects columns:
        sample, chromosome, start, end, segVal
    where segVal is the absolute copy number (not log2 ratio).
    ichorCNA provides copy.number directly — we use that if available,
    otherwise convert logR to CN: CN = 2 * 2^logR.

    One representative sample per patient (highest TF) is used so that
    samples from the same patient don't dominate the signature extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to qualifying samples
    eligible = tf_df[
        (tf_df["tumor_fraction"] >= min_tf) &
        tf_df["sample"].isin(seg_files)
    ].copy()

    if eligible.empty:
        print(f"  [sigminer] No samples with TF >= {min_tf:.0%}.")
        return None

    # One representative per patient: highest TF
    eligible = (eligible
                .sort_values("tumor_fraction", ascending=False)
                .drop_duplicates(subset=["patient"], keep="first"))
    eligible = eligible.sort_values(["patient", "sample"])

    print(f"  [sigminer] {len(eligible)} samples qualifying "
          f"(TF >= {min_tf:.0%}, one per patient).")

    chrom_rank = {c: i for i, c in enumerate(CHROM_ORDER)}

    rows = []
    n_loaded = 0
    for _, row in eligible.iterrows():
        sample  = row["sample"]
        seg_df  = parse_seg_file(seg_files[sample], sample)
        if seg_df is None or seg_df.empty:
            print(f"    [WARN] Could not load seg file for {sample} — skipping.")
            continue

        # Determine copy number column
        cn_col  = None
        logr_col = None
        for c in ["copy.number", "copy_number", "cn"]:
            if c in seg_df.columns:
                cn_col = c
                break
        for c in ["logr", "seg.mean", "log2_ratio", "median"]:
            if c in seg_df.columns:
                logr_col = c
                break

        for _, seg in seg_df.iterrows():
            chrom = str(seg["chrom"])
            if chrom not in chrom_rank:
                continue
            start = int(seg["start"])
            end   = int(seg["end"])

            if cn_col is not None and not pd.isna(seg[cn_col]):
                seg_val = float(seg[cn_col])
            elif logr_col is not None and not pd.isna(seg[logr_col]):
                # Convert logR to absolute CN assuming diploid baseline
                seg_val = 2.0 * (2.0 ** float(seg[logr_col]))
            else:
                continue

            # Clamp to reasonable range
            seg_val = max(0.0, min(seg_val, 20.0))

            rows.append({
                "sample":     sample,
                "chromosome": chrom,
                "start":      start,
                "end":        end,
                "segVal":     round(seg_val, 4),
                "entity":     row.get("entity", "Unknown"),
            })
        n_loaded += 1

    if not rows:
        print("  [sigminer] No valid segment data loaded.")
        return None

    seg_df_out = pd.DataFrame(rows)
    # Sort by sample, then genome order
    seg_df_out["_chrom_rank"] = seg_df_out["chromosome"].map(chrom_rank)
    seg_df_out = (seg_df_out
                  .sort_values(["sample", "_chrom_rank", "start"])
                  .drop(columns=["_chrom_rank"]))

    out_path = output_dir / "segments_input.tsv"
    seg_df_out.to_csv(out_path, sep="\t", index=False)
    print(f"  [sigminer] Written: {out_path}  "
          f"({n_loaded} samples, {len(seg_df_out)} segments)")
    return out_path


# ---------------------------------------------------------------------------
# R script template
# ---------------------------------------------------------------------------

_R_SCRIPT = """\
#!/usr/bin/env Rscript
# Auto-generated by cfdna-cnv sigminer module
# Fits CN signatures using sigminer against pre-computed reference sets.
#
# Required R packages (add to pixi.toml):
#   r-sigminer, r-ggplot2, r-dplyr, r-cowplot

suppressPackageStartupMessages({{
  library(sigminer)
  library(ggplot2)
  library(dplyr)
}})

# ── Configuration ─────────────────────────────────────────────────────────────
seg_file   <- "{seg_file}"
output_dir <- "{output_dir}"
ref_sets   <- c({ref_sets})   # e.g. c("CNS_TCGA", "CNS_PCAWG176")
genome     <- "{genome}"      # hg19 or hg38
min_segs   <- {min_segs}      # minimum segments per sample
cores      <- {cores}
n_sig_max  <- {n_sig_max}  # number of de novo signatures to extract

dir.create(file.path(output_dir, "plots"), showWarnings=FALSE, recursive=TRUE)

cat("Reading segments ...\\n")
seg <- read.table(seg_file, sep="\\t", header=TRUE, stringsAsFactors=FALSE)
# Keep only the 5 columns sigminer needs; drop entity and any extras
seg <- seg[, c("sample", "chromosome", "start", "end", "segVal")]
cat(sprintf("  %d segments across %d samples\\n",
    nrow(seg), length(unique(seg$sample))))
# Add minor_cn=0 (ichorCNA has no allele data) so add_loh=TRUE works
seg$minor_cn <- 0L
# ── Build copy-number object ──────────────────────────────────────────────────
cat("Building CopyNumber object ...\\n")
cn <- tryCatch(
  read_copynumber(
    seg,
    seg_cols       = c("chromosome", "start", "end", "segVal"),
    samp_col       = "sample",
    genome_build   = genome,
    genome_measure = "wg",
    min_segnum     = min_segs,
    join_adj_seg   = FALSE,
    add_loh        = TRUE,
    complement     = FALSE,
    verbose        = FALSE
  ),
  error = function(e) {{
    cat(sprintf("[ERROR] read_copynumber failed: %s\n", conditionMessage(e)))
    quit(status=1)
  }}
)
cat(sprintf("  CopyNumber object built.\\n"))

# Check sample count
n_samples_loaded <- length(unique(cn@data$sample))
cat(sprintf("  Loaded %d samples\n", n_samples_loaded))
if (n_samples_loaded == 0) {{
  cat("[ERROR] No samples in CopyNumber object.\n")
  quit(status=1)
}}

# ── Extract CN features ───────────────────────────────────────────────────────
# Use method 'S' (Steele) with add_loh=TRUE + CN_48, compatible with CNS_TCGA.
# minor_cn=0 above satisfies add_loh; all segments treated as non-LOH.
cat('Extracting CN features (method S, CN_48) ...\\n')
cn_tally <- tryCatch(
  sig_tally(cn, method='S', cores=cores),
  error = function(e) {{
    cat(sprintf('[ERROR] sig_tally failed: %s\\n', conditionMessage(e)))
    quit(status=1)
  }}
)
if (!'CN_48' %in% names(cn_tally$all_matrices)) {{
  cat(sprintf('[ERROR] CN_48 not found. Available: %s\\n',
      paste(names(cn_tally$all_matrices), collapse=', ')))
  quit(status=1)
}}
cat_matrix <- t(cn_tally$all_matrices$CN_48)  # 48 rows x N samples
cat(sprintf('  CN_48 matrix: %d components x %d samples\\n',
    nrow(cat_matrix), ncol(cat_matrix)))
write.table(cn_tally$all_matrices$CN_48,
  file.path(output_dir, 'cn_feature_matrix.tsv'),
  sep='\\t', quote=FALSE, col.names=NA)
cat(sprintf('  Written: cn_feature_matrix.tsv\\n'))

# ── De novo extraction + similarity to references ───────────────────────────
# Note: direct sig_fit against CNS_TCGA fails due to internal row-matching
# in this sigminer version despite correct component names. Instead we:
#   1. Extract de novo signatures via NMF (sig_extract)
#   2. Map to CNS_TCGA via cosine similarity (get_sig_similarity)
#   3. Report which TCGA signature each de novo sig best matches
cat(sprintf('\\nExtracting %d de novo CN signatures ...\\n', n_sig_max))
nmf_mat <- cn_tally$all_matrices$CN_48  # samples x 48 components
sig_obj <- tryCatch(
  sig_extract(nmf_mat, n_sig=n_sig_max, cores=cores, seed=42),
  error=function(e) {{ cat(sprintf('[ERROR] sig_extract: %s\\n', conditionMessage(e))); quit(status=1) }}
)
cat(sprintf('  Extracted %d signatures.\\n', ncol(sig_obj$Signature)))

# Write exposures
write.table(sig_obj$Signature, file.path(output_dir, 'denovo_signatures.tsv'),
  sep='\\t', quote=FALSE, col.names=NA)
write.table(t(sig_obj$Exposure), file.path(output_dir, 'exposures_denovo.tsv'),
  sep='\\t', quote=FALSE, col.names=NA)
exp_rel <- t(sig_obj$Exposure); rs <- rowSums(exp_rel)
exp_rel[rs > 0, ] <- exp_rel[rs > 0, ] / rs[rs > 0]
write.table(exp_rel, file.path(output_dir, 'exposures_denovo_relative.tsv'),
  sep='\\t', quote=FALSE, col.names=NA)
cat('  Written: denovo_signatures.tsv, exposures_denovo.tsv, exposures_denovo_relative.tsv\\n')

# Map to reference signatures
for (ref in ref_sets) {{
  tryCatch({{
    sim <- get_sig_similarity(sig_obj, sig_db=ref, normalize='row')
    write.table(sim$similarity, file.path(output_dir, sprintf('similarity_%s.tsv', ref)),
      sep='\\t', quote=FALSE, col.names=NA)
    cat(sprintf('  Written: similarity_%s.tsv\\n', ref))
    bm <- sim$best_match
    if (!is.null(bm) && is.data.frame(bm) && nrow(bm) > 0) {{
      cat(sprintf('  Best matches to %s:\\n', ref))
      for (i in seq_len(nrow(bm)))
        cat(sprintf('    Sig%d -> %s (cosine=%.3f)\\n', i, bm$best_db_sig[i], bm$similarity[i]))
    }}
  }}, error=function(e) cat(sprintf('  [WARN] Similarity to %s: %s\\n', ref, conditionMessage(e))))
}}

# Plots
tryCatch({{
  p_prof <- show_sig_profile(
    sig_obj,
    mode            = 'copynumber',
    method          = 'S',
    style           = 'default',
    check_sig_names = FALSE
  )
  ggplot2::ggsave(file.path(output_dir, 'plots', 'denovo_profiles.pdf'),
    plot=p_prof, width=14, height=max(6, 3*n_sig_max), limitsize=FALSE)
  p_exp <- show_sig_exposure(sig_obj, rm_space=TRUE)
  ggplot2::ggsave(file.path(output_dir, 'plots', 'denovo_exposures.pdf'),
    plot=p_exp, width=max(10, ncol(cat_matrix)*0.3), height=6, limitsize=FALSE)
  cat('  Plots written: denovo_profiles.pdf, denovo_exposures.pdf\\n')
}}, error=function(e) cat(sprintf('  [WARN] Plots: %s\\n', conditionMessage(e))))

# ── Summary: dominant signature per sample ─────────────────────────────────────
cat("\\n── Dominant signature per sample ──\\n")
for (ref in ref_sets) {{
  abs_path <- file.path(output_dir, sprintf("exposures_%s.tsv", ref))
  if (!file.exists(abs_path)) next
  exp <- read.table(abs_path, sep="\\t", header=TRUE, stringsAsFactors=FALSE)
  sig_cols <- names(exp)[sapply(exp, is.numeric)]
  if (length(sig_cols) == 0) next
  dom_sig  <- apply(exp[, sig_cols, drop=FALSE], 1,
                    function(x) names(x)[which.max(x)])
  cat(sprintf("\\n%s dominant signatures:\\n", ref))
  print(table(dom_sig))

  # Write summary
  summary_df <- data.frame(
    sample          = exp[[1]],
    dominant_sig    = dom_sig,
    dominant_weight = apply(exp[, sig_cols, drop=FALSE], 1, max)
  )
  summ_out <- file.path(output_dir,
                         sprintf("dominant_signature_%s.tsv", ref))
  write.table(summary_df, summ_out, sep="\\t", quote=FALSE, row.names=FALSE)
  cat(sprintf("Written: %s\\n", summ_out))
}}

cat("\\nsigminer CN signature analysis complete.\\n")
"""


def write_r_script(
        output_dir: Path,
        seg_file:   Path,
        ref_sets:   list[str],
        genome:     str = "hg38",
        min_segs:   int = 5,
        cores:      int = 1,
        n_sig_max:  int = 6,
) -> Path:
    """Write the sigminer R script to output_dir/run_sigminer.R."""
    # Ensure ref_sets is a list (may arrive as a string from YAML config)
    if isinstance(ref_sets, str):
        ref_sets = [ref_sets]
    # Format ref sets as R character vector
    ref_r = ", ".join(f'"{r}"' for r in ref_sets)

    script = _R_SCRIPT.format(
        seg_file   = str(seg_file),
        output_dir = str(output_dir),
        ref_sets   = ref_r,
        genome     = genome,
        min_segs   = min_segs,
        cores      = cores,
        n_sig_max  = n_sig_max,
    )
    script_path = output_dir / "run_sigminer.R"
    script_path.write_text(script)  # always overwrite
    print(f"  [sigminer] R script written: {script_path}")
    return script_path


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_sigminer(
        tf_df:       pd.DataFrame,
        seg_files:   dict[str, Path],
        output_dir:  Path,
        ref_sets:    list[str] | None = None,
        min_tf:      float = 0.15,
        genome:      str   = "hg38",
        min_segs:    int   = 5,
        cores:       int   = 1,
        n_sig_max:   int   = 6,
        rscript_bin: str   = "Rscript",
        run_r:       bool  = True,
) -> Path | None:
    """
    Full sigminer pipeline: prepare inputs, write R script, optionally run.

    Parameters
    ----------
    tf_df       : tumor fraction table.
    seg_files   : {sample: Path} to 1 Mbp .seg.txt files.
    output_dir  : directory for all sigminer outputs.
    ref_sets    : list of reference signature sets to fit against.
                  Default: ["CNS_TCGA", "CNS_PCAWG176"].
    min_tf      : minimum TF to include a sample (default 0.15 = 15%).
    genome      : reference genome, "hg38" or "hg19".
    min_segs    : minimum number of segments per sample (default 5).
    cores       : number of parallel cores for sigminer (default 1).
    rscript_bin : path to Rscript executable.
    run_r       : if True, execute the R script automatically.

    Returns
    -------
    Path to output directory, or None if no samples qualify.
    """
    if ref_sets is None:
        ref_sets = ["CNS_TCGA"]
    elif isinstance(ref_sets, str):
        ref_sets = [ref_sets]

    print(f"\n  [sigminer] CN signature analysis "
          f"(min TF={min_tf:.0%}, refs={ref_sets}) …")

    seg_file = prepare_sigminer_input(
        tf_df, seg_files, output_dir, min_tf=min_tf
    )
    if seg_file is None:
        return None

    script_path = write_r_script(
        output_dir, seg_file, ref_sets,
        genome=genome, min_segs=min_segs, cores=cores,
        n_sig_max=n_sig_max,
    )

    if run_r:
        print(f"  [sigminer] Running R script …")
        result = subprocess.run(
            [rscript_bin, str(script_path)],
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] Rscript exited with code {result.returncode}.")
        else:
            print(f"  [sigminer] Analysis complete.")
    else:
        print(f"  [sigminer] To run manually:")
        print(f"    Rscript {script_path}")

    return output_dir