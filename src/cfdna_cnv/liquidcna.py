"""
liquidCNA integration module.

Prepares input data from ichorCNA seg.txt files and runs the liquidCNA
R script (mixture_estimation_functions.R from github.com/elakatos/liquidCNA)
to estimate purity and subclonal ratio across longitudinal samples.

Pipeline
--------
1. Filter: patients with >= 2 timepoints, both above min_tf (default 0.20).
2. Load 1 Mbp segmented .seg.txt files for qualifying samples.
3. Bin logR values to a fixed 1 Mbp grid so all samples share coordinates.
4. Build a wide matrix: rows = genomic bins, columns = samples
   (column names include patient + timepoint, e.g. B46_TP1, B46_TP2).
5. Write the matrix, a sample metadata TSV, and the purity (TF) estimates.
6. Call an R script that sources mixture_estimation_functions.R and runs
   purity and subclonal-ratio estimation.
7. Return the output directory.

Usage (CLI)
-----------
    cfdna-cnv locate ... --liquidcna --liquidcna-functions /path/to/mixture_estimation_functions.R
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .common.constants import CHROM_ORDER, CHROM_LENGTHS
from .common.io import parse_seg_file


# ---------------------------------------------------------------------------
# Genomic bin grid (1 Mbp, same as broad CNA)
# ---------------------------------------------------------------------------

_BIN_SIZE = 1_000_000


def _make_bin_index() -> pd.DataFrame:
    """Return a sorted DataFrame of all 1 Mbp genome bins."""
    rows = []
    for chrom in CHROM_ORDER:
        length = CHROM_LENGTHS[chrom]
        b = 0
        while b < length:
            rows.append({"chrom": chrom, "bin_start": b,
                         "bin_end": min(b + _BIN_SIZE, length)})
            b += _BIN_SIZE
    return pd.DataFrame(rows)


def _seg_to_bins(seg_df: pd.DataFrame, logr_col: str) -> pd.Series:
    """
    Map a segmented logR DataFrame onto 1 Mbp bins.
    Each bin receives the logR of the segment with the largest overlap.
    Returns a Series indexed by (chrom, bin_start).
    """
    bin_idx = _make_bin_index()
    bin_vals = np.full(len(bin_idx), np.nan)

    # Build per-chrom segment list
    seg_by_chrom: dict[str, list] = {}
    for _, row in seg_df.iterrows():
        if pd.isna(row.get(logr_col)):
            continue
        seg_by_chrom.setdefault(row["chrom"], []).append(
            (int(row["start"]), int(row["end"]), float(row[logr_col]))
        )

    for i, bin_row in bin_idx.iterrows():
        chrom, bs, be = bin_row["chrom"], bin_row["bin_start"], bin_row["bin_end"]
        segs = seg_by_chrom.get(chrom, [])
        best_ov, best_val = 0, np.nan
        for ss, se, lv in segs:
            ov = min(se, be) - max(ss, bs)
            if ov > best_ov:
                best_ov, best_val = ov, lv
        bin_vals[i] = best_val

    bin_idx["logr"] = bin_vals
    bin_idx["bin_id"] = (bin_idx["chrom"] + ":" +
                         bin_idx["bin_start"].astype(str))
    return bin_idx.set_index("bin_id")["logr"]


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_liquidcna_input(
        tf_df:        pd.DataFrame,
        seg_files:    dict[str, Path],
        output_dir:   Path,
        min_tf:       float = 0.20,
        min_timepoints: int = 2,
) -> tuple[Path, Path, Path] | None:
    """
    Select qualifying patients and build the wide logR matrix for liquidCNA.

    Parameters
    ----------
    tf_df         : tumor fraction table (sample, patient, timepoint,
                    tumor_fraction).
    seg_files     : {sample: Path} to 1 Mbp .seg.txt files.
    output_dir    : directory where input files are written.
    min_tf        : minimum tumor fraction for a sample to be included
                    (default 0.20 = 20%).
    min_timepoints: minimum number of qualifying timepoints per patient
                    (default 2).

    Returns
    -------
    (matrix_tsv, metadata_tsv, purity_tsv) paths, or None if no patients
    qualify.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to samples with TF >= min_tf and a known timepoint
    eligible = tf_df[
        (tf_df["tumor_fraction"] >= min_tf) &
        tf_df["timepoint"].notna() &
        tf_df["sample"].isin(seg_files)
    ].copy()
    eligible["timepoint"] = eligible["timepoint"].astype(int)
    eligible = eligible.sort_values(["patient", "timepoint"])

    # Keep only patients with >= min_timepoints qualifying samples
    counts = eligible.groupby("patient")["sample"].count()
    qualifying_patients = counts[counts >= min_timepoints].index
    eligible = eligible[eligible["patient"].isin(qualifying_patients)]

    if eligible.empty:
        print(f"  [liquidCNA] No patients with >= {min_timepoints} timepoints "
              f"above {min_tf:.0%} TF.")
        return None

    n_patients = eligible["patient"].nunique()
    n_samples  = len(eligible)
    print(f"  [liquidCNA] {n_patients} patients, {n_samples} samples qualifying.")

    # Build logR column name lookup
    def _get_logr_col(df: pd.DataFrame) -> str | None:
        for c in ["logr", "seg.mean", "log2_ratio", "median"]:
            if c in df.columns:
                return c
        return None

    # Build wide matrix
    bin_idx  = _make_bin_index()
    bin_ids  = (bin_idx["chrom"] + ":" +
                bin_idx["bin_start"].astype(str)).tolist()
    columns: dict[str, pd.Series] = {}

    for _, row in eligible.iterrows():
        sample  = row["sample"]
        patient = row["patient"]
        tp      = int(row["timepoint"])
        col_name = f"{patient}_TP{tp}"

        seg_df = parse_seg_file(seg_files[sample], sample)
        if seg_df is None or seg_df.empty:
            print(f"    [WARN] Could not load seg file for {sample} — skipping.")
            continue

        lc = _get_logr_col(seg_df)
        if lc is None:
            print(f"    [WARN] No logR column in {sample} — skipping.")
            continue

        s = _seg_to_bins(seg_df, lc)
        s = s.reindex(bin_ids)
        columns[col_name] = s

    if not columns:
        print("  [liquidCNA] No valid seg files loaded.")
        return None

    matrix = pd.DataFrame(columns, index=bin_ids)

    # Drop bins that are NA in all samples
    matrix = matrix.dropna(how="all")
    print(f"  [liquidCNA] Matrix: {len(matrix)} bins × {len(matrix.columns)} samples")

    # Write matrix
    matrix_tsv = output_dir / "liquidcna_matrix.tsv"
    matrix.to_csv(matrix_tsv, sep="\t")
    print(f"  [liquidCNA] Written: {matrix_tsv}")

    # Write metadata
    meta_rows = []
    for col in matrix.columns:
        # col = "B46_TP2"
        parts    = col.rsplit("_TP", 1)
        patient  = parts[0]
        tp       = int(parts[1]) if len(parts) == 2 else 0
        tf_val   = eligible.loc[
            (eligible["patient"] == patient) &
            (eligible["timepoint"] == tp), "tumor_fraction"
        ].values
        meta_rows.append({
            "column":         col,
            "patient":        patient,
            "timepoint":      tp,
            "tumor_fraction": float(tf_val[0]) if len(tf_val) else np.nan,
            "is_baseline":    tp == eligible.loc[
                eligible["patient"] == patient, "timepoint"
            ].min(),
        })
    metadata_df = pd.DataFrame(meta_rows)
    metadata_tsv = output_dir / "liquidcna_metadata.tsv"
    metadata_df.to_csv(metadata_tsv, sep="\t", index=False)
    print(f"  [liquidCNA] Written: {metadata_tsv}")

    # Write purity file (ichorCNA TF estimates, used as starting values)
    purity_df   = metadata_df[["column", "tumor_fraction"]].rename(
        columns={"column": "sample", "tumor_fraction": "purity_ichorCNA"}
    )
    purity_tsv  = output_dir / "liquidcna_purity_prior.tsv"
    purity_df.to_csv(purity_tsv, sep="\t", index=False)
    print(f"  [liquidCNA] Written: {purity_tsv}")

    return matrix_tsv, metadata_tsv, purity_tsv


# ---------------------------------------------------------------------------
# R script template
# ---------------------------------------------------------------------------


_R_SCRIPT = """\
#!/usr/bin/env Rscript
# Auto-generated by cfdna-cnv liquidCNA module

# ── Load required packages ───────────────────────────────────────────────────
# Pre-install via pixi.toml: r-pracma, r-mixtools, r-fitdistrplus, r-gtools, r-dplyr
suppressPackageStartupMessages({{
  library(pracma)
  library(mixtools)
  library(fitdistrplus)
  library(gtools)
  library(dplyr)
}})

# ── Source liquidCNA functions ──────────────────────────────────
# All required packages must be pre-installed (add to pixi.toml).
# If a package is missing, R will throw a clear error here.
source('{functions_path}')  # loads estimatePurity, estimateRatio etc.
cat('liquidCNA functions loaded.\n')


# ── Load inputs ───────────────────────────────────────────────────────────────
cat("Loading matrix ...\\n")
mat      <- read.table("{matrix_tsv}", sep="\\t", header=TRUE,
                        row.names=1, check.names=FALSE)
meta     <- read.table("{metadata_tsv}", sep="\\t", header=TRUE,
                        stringsAsFactors=FALSE)
purity_p <- read.table("{purity_tsv}", sep="\\t", header=TRUE,
                        stringsAsFactors=FALSE)
cat(sprintf("Matrix: %d bins x %d samples\\n", nrow(mat), ncol(mat)))
cat(sprintf("Patients: %s\\n", paste(unique(meta$patient), collapse=", ")))

# ── Run per patient ───────────────────────────────────────────────────────────
# Parameters (tune as needed)
w         <- c(0.8, 1, 1, 0.25, 0.05)   # CN state weights for purity estimation
adjVec    <- c(0.6, 0.8, 1.0, 1.2)      # bandwidth adjustments
pVec      <- seq(0.05, 0.70, by=0.005)  # purity grid to search
cutOffVec <- seq(0.025, 0.35, by=0.01)  # clonal-filter cutoff grid
epsilon   <- 0.05                         # ordering threshold

all_purity <- list()
all_ratio  <- list()

for (pat in unique(meta$patient)) {{
  cat(sprintf("\\n=== Patient: %s ===\\n", pat))
  pat_meta <- meta[meta$patient == pat, ]
  pat_cols <- pat_meta$column
  pat_mat  <- mat[, pat_cols, drop=FALSE]
  pat_mat  <- pat_mat[complete.cases(pat_mat), ]

  if (nrow(pat_mat) < 10) {{
    cat(sprintf("  [WARN] Too few complete bins (%d) — skipping\\n", nrow(pat_mat)))
    next
  }}

  baseline_col <- pat_cols[which.min(pat_meta$timepoint)]
  post_cols    <- setdiff(pat_cols, baseline_col)
  cat(sprintf("  Baseline: %s  Post: %s\\n",
      baseline_col, paste(post_cols, collapse=", ")))

  # ── Step 1: Re-centre so mode = CN 2 ────────────────────────────────────
  reNorm <- centreSegs(pat_mat)
  seg.norm <- sweep(pat_mat, 2, reNorm - 2, FUN="-")

  # ── Step 2: Purity estimation via density peak fitting ─────────────────
  purity_est <- setNames(numeric(length(pat_cols)), pat_cols)
  for (col in pat_cols) {{
    x <- na.omit(seg.norm[, col])
    pFits <- sapply(adjVec, function(a)
               sapply(pVec, function(p) evalPurityDensity(p, w, a, x)))
    best_p <- pVec[which.min(apply(pFits, 1, median))]
    purity_est[[col]] <- best_p
    cat(sprintf("  %s: ichorCNA TF=%.3f  liquidCNA purity=%.3f\\n",
        col,
        purity_p$purity_ichorCNA[purity_p$sample == col],
        best_p))
  }}

  pat_meta$purity_liquidCNA <- purity_est[pat_meta$column]
  all_purity[[pat]] <- pat_meta[, c("column","patient","timepoint",
                                     "tumor_fraction","purity_liquidCNA")]

  if (length(post_cols) == 0) {{
    cat("  [SKIP] No post-baseline samples for ratio estimation.\\n")
    next
  }}

  # ── Step 3: Normalise by purity → relative segment CN ───────────────────
  # DeltaCN = (observed - 2) / purity, then ratio to baseline
  seg.pur <- sweep(seg.norm, 2, purity_est, FUN="/")
  seg.rel <- seg.pur / seg.pur[, baseline_col]

  # ── Step 4: Find best ordering of post-baseline samples ─────────────────
  seg.rel.nonBase <- seg.rel[, post_cols, drop=FALSE]
  nCol    <- length(post_cols)
  ordVec  <- if (nCol > 1) permutations(nCol, nCol, post_cols) else matrix(post_cols)

  fitInfo <- list()
  for (cutOff in cutOffVec) {{
    seg.eval <- filterSegmentRatios(seg.rel.nonBase, cutOff, "sd", 0)
    best     <- findBestOrder(seg.eval, ordVec, epsilon, nCol, 0)
    fitInfo[[as.character(cutOff)]] <- best
  }}
  fitInfo.df <- data.frame(
    cutOff      = cutOffVec,
    maxFit      = sapply(fitInfo, function(x) ifelse(is.null(x$max), 0, x$max)),
    segsInOrder = sapply(fitInfo, function(x) max(sapply(x$segs, length)))
  )
  best_cut <- getCutOffAuto(fitInfo.df, minNum=5)
  best_fit <- fitInfo[[as.character(best_cut)]]
  cat(sprintf("  Best cutoff: %.3f  Subclonal segs: %d\\n",
      best_cut, max(sapply(best_fit$segs, length))))

  if (is.null(best_fit$ord) || length(best_fit$segs[[1]]) == 0) {{
    cat("  [WARN] No subclonal segments found — skipping ratio estimation.\\n")
    next
  }}

  # ── Step 5: Estimate subclonal ratio via Gaussian fit ───────────────────
  ord_idx <- 1
  seg.rel.sub <- seg.rel[best_fit$segs[[ord_idx]],
                          best_fit$ord[ord_idx,], drop=FALSE]
  final.medians <- data.frame(time=post_cols, rat=NA, rat_sd=NA)

  tryCatch({{
    for (samp in post_cols) {{
      final.medians <- estimateRGaussianFit(seg.rel.sub, samp, final.medians)
    }}
    cat("  Subclonal ratio estimates:\\n")
    print(final.medians)
    all_ratio[[pat]] <- data.frame(
      column    = final.medians$time,
      patient   = pat,
      ratio_est = final.medians$rat,
      ratio_sd  = final.medians$rat_sd
    )
  }}, error=function(e) {{
    cat(sprintf("  [ERROR] Ratio estimation failed: %s\\n", conditionMessage(e)))
  }})
}}

# ── Write outputs ─────────────────────────────────────────────────────────────
if (length(all_purity) > 0) {{
  purity_out <- do.call(rbind, all_purity)
  write.table(purity_out, "{output_dir}/liquidcna_purity_results.tsv",
              sep="\\t", row.names=FALSE, quote=FALSE)
  cat("\\nWritten: {output_dir}/liquidcna_purity_results.tsv\\n")
}}
if (length(all_ratio) > 0) {{
  ratio_out <- do.call(rbind, all_ratio)
  write.table(ratio_out, "{output_dir}/liquidcna_ratio_results.tsv",
              sep="\\t", row.names=FALSE, quote=FALSE)
  cat("Written: {output_dir}/liquidcna_ratio_results.tsv\\n")
}}
cat("\\nliquidCNA analysis complete.\\n")
"""

def write_r_script(
        output_dir:     Path,
        functions_path: Path,
        matrix_tsv:     Path,
        metadata_tsv:   Path,
        purity_tsv:     Path,
) -> Path:
    """Write the R analysis script to output_dir/run_liquidcna.R."""
    script_path = output_dir / "run_liquidcna.R"
    script_path.write_text(
        _R_SCRIPT.format(
            functions_path = str(functions_path),
            matrix_tsv     = str(matrix_tsv),
            metadata_tsv   = str(metadata_tsv),
            purity_tsv     = str(purity_tsv),
            output_dir     = str(output_dir),
        )
    )
    print(f"  [liquidCNA] R script written: {script_path}")
    return script_path


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_liquidcna(
        tf_df:          pd.DataFrame,
        seg_files_1mb:  dict[str, Path],
        output_dir:     Path,
        functions_path: Path,
        min_tf:         float = 0.20,
        min_timepoints: int   = 2,
        rscript_bin:    str   = "Rscript",
        run_r:          bool  = True,
) -> Path | None:
    """
    Full liquidCNA pipeline: prepare inputs, write R script, optionally run it.

    Parameters
    ----------
    tf_df           : tumor fraction table.
    seg_files_1mb   : {sample: Path} to 1 Mbp .seg.txt files.
    output_dir      : directory for all liquidCNA outputs.
    functions_path  : path to mixture_estimation_functions.R from the
                      liquidCNA GitHub repo.
    min_tf          : minimum TF to include a sample (default 0.20).
    min_timepoints  : minimum qualifying timepoints per patient (default 2).
    rscript_bin     : path to Rscript executable (default "Rscript").
    run_r           : if True, execute the R script automatically.

    Returns
    -------
    Path to the output directory, or None if no patients qualify.
    """
    print(f"\n  [liquidCNA] Preparing inputs (min TF={min_tf:.0%}, "
          f"min timepoints={min_timepoints}) …")

    result = prepare_liquidcna_input(
        tf_df, seg_files_1mb, output_dir,
        min_tf=min_tf, min_timepoints=min_timepoints,
    )
    if result is None:
        return None

    matrix_tsv, metadata_tsv, purity_tsv = result

    if not functions_path.exists():
        print(f"  [WARN] mixture_estimation_functions.R not found at "
              f"{functions_path}. R script written but not executed.")
        print("  Download from: https://github.com/elakatos/liquidCNA")
        run_r = False

    script_path = write_r_script(
        output_dir, functions_path, matrix_tsv, metadata_tsv, purity_tsv
    )

    if run_r:
        print(f"  [liquidCNA] Running R script …")
        result = subprocess.run(
            [rscript_bin, str(script_path)],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] Rscript exited with code {result.returncode}.")
        else:
            print(f"  [liquidCNA] R analysis complete.")
    else:
        print(f"  [liquidCNA] To run manually:")
        print(f"    Rscript {script_path}")

    return output_dir