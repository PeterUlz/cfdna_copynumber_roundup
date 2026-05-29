"""
LiftOver support for cBioPortal segmentation files (hg19 → hg38).

Workflow per seg file:
  1. Write coordinates to a 6-column BED file, preserving a row index so
     every segment can be matched back after lifting even when rows are
     reordered or dropped.
  2. Run the UCSC liftOver binary.
  3. Parse the lifted BED and the unmapped file.
  4. Join lifted coordinates back onto the original DataFrame by row index,
     replacing chrom/start/end in-place and dropping segments that could
     not be lifted.

Requirements
------------
- liftOver binary on PATH  (or specify path with --liftover-bin)
- hg19ToHg38.over.chain.gz  (download once; specify with --chain-file)

Downloading the chain file
--------------------------
    wget https://hgdownload.gi.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz

The liftOver binary (Linux x86_64):
    wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver
    chmod +x liftOver
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Chain file / binary helpers
# ---------------------------------------------------------------------------

CHAIN_URL = (
    "https://hgdownload.gi.ucsc.edu/goldenPath/hg19/liftOver/"
    "hg19ToHg38.over.chain.gz"
)
LIFTOVER_URL = (
    "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
)


def check_liftover_binary(liftover_bin: str = "liftOver") -> str:
    """
    Return the path to the liftOver binary, or raise RuntimeError with
    download instructions if it cannot be found.
    """
    resolved = shutil.which(liftover_bin)
    if resolved:
        return resolved
    if Path(liftover_bin).is_file():
        return liftover_bin
    raise RuntimeError(
        f"liftOver binary not found: '{liftover_bin}'.\n"
        f"Download it with:\n"
        f"  wget {LIFTOVER_URL}\n"
        f"  chmod +x liftOver\n"
        f"Then either place it on your PATH or pass --liftover-bin /path/to/liftOver"
    )


def check_chain_file(chain_file: str | Path) -> Path:
    """
    Verify the chain file exists, or raise RuntimeError with download
    instructions.
    """
    p = Path(chain_file)
    if p.is_file():
        return p
    raise RuntimeError(
        f"Chain file not found: '{chain_file}'.\n"
        f"Download it with:\n"
        f"  wget {CHAIN_URL}\n"
        f"Then pass --chain-file /path/to/hg19ToHg38.over.chain.gz"
    )


# ---------------------------------------------------------------------------
# Core liftover function
# ---------------------------------------------------------------------------

def liftover_seg_df(
        seg_df:        pd.DataFrame,
        chain_file:    str | Path,
        liftover_bin:  str = "liftOver",
        min_match:     float = 0.95,
        verbose:       bool = True,
) -> pd.DataFrame:
    """
    Lift coordinates in a cBioPortal seg DataFrame from hg19 to hg38.

    Parameters
    ----------
    seg_df       : DataFrame with columns patient, chrom, start, end, seg_mean
                   (output of cbio.loader.load_seg_file).
    chain_file   : Path to hg19ToHg38.over.chain.gz.
    liftover_bin : Path or name of the liftOver executable.
    min_match    : Minimum fraction of bases that must remap (passed to
                   liftOver as -minMatch). Default 0.95 is conservative;
                   lower to 0.1 to keep more segments.
    verbose      : Print a summary of lifted/dropped segments.

    Returns
    -------
    A new DataFrame with hg38 coordinates. Segments that could not be
    lifted are dropped. The original column order is preserved.
    """
    liftover_bin = check_liftover_binary(liftover_bin)
    chain_file   = check_chain_file(chain_file)

    if seg_df.empty:
        return seg_df.copy()

    df = seg_df.reset_index(drop=True).copy()

    # Ensure chrom has 'chr' prefix (liftOver requires it)
    df["chrom"] = df["chrom"].astype(str).apply(
        lambda c: c if c.startswith("chr") else f"chr{c}"
    )

    with tempfile.TemporaryDirectory(prefix="liftover_") as tmp:
        tmp_dir      = Path(tmp)
        bed_in       = tmp_dir / "input.bed"
        bed_lifted   = tmp_dir / "lifted.bed"
        bed_unmapped = tmp_dir / "unmapped.bed"

        # ── Write BED4+ ──────────────────────────────────────────────────────
        # BED columns: chrom  start  end  name(=row_index)
        # We encode the original row index in the name column so we can
        # join back after liftOver reorders / drops rows.
        bed_rows = df[["chrom", "start", "end"]].copy()
        bed_rows["name"] = df.index.astype(str)
        bed_rows.to_csv(bed_in, sep="\t", index=False, header=False)

        # ── Run liftOver ─────────────────────────────────────────────────────
        cmd = [
            liftover_bin,
            f"-minMatch={min_match}",
            str(bed_in),
            str(chain_file),
            str(bed_lifted),
            str(bed_unmapped),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"liftOver failed (exit {e.returncode}).\n"
                f"stderr: {e.stderr.strip()}"
            ) from e

        # ── Parse results ────────────────────────────────────────────────────
        if not bed_lifted.exists() or bed_lifted.stat().st_size == 0:
            raise RuntimeError(
                "liftOver produced no output. Check that the chain file "
                "matches the input assembly (expected hg19ToHg38)."
            )

        lifted = pd.read_csv(
            bed_lifted, sep="\t", header=None,
            names=["chrom_hg38", "start_hg38", "end_hg38", "row_idx"],
            dtype={"row_idx": str},
        )
        lifted["row_idx"] = lifted["row_idx"].astype(int)

        # Count unmapped segments
        n_unmapped = 0
        if bed_unmapped.exists():
            with open(bed_unmapped) as fh:
                # liftOver unmapped file has comment lines starting with '#'
                n_unmapped = sum(1 for ln in fh if not ln.startswith("#"))

    # ── Join lifted coordinates back onto original DataFrame ─────────────────
    lifted = lifted.set_index("row_idx")
    df["chrom"] = lifted["chrom_hg38"].reindex(df.index)
    df["start"] = lifted["start_hg38"].reindex(df.index)
    df["end"]   = lifted["end_hg38"].reindex(df.index)

    n_in     = len(df)
    df_out   = df.dropna(subset=["chrom", "start", "end"]).copy()
    df_out["start"] = df_out["start"].astype(int)
    df_out["end"]   = df_out["end"].astype(int)
    n_out    = len(df_out)
    n_dropped = n_in - n_out

    if verbose:
        entity = df_out["entity"].iloc[0] if "entity" in df_out.columns else "?"
        print(f"      [{entity}] liftOver hg19→hg38: "
              f"{n_out}/{n_in} segments lifted "
              f"({n_dropped} dropped, {n_unmapped} unmapped by liftOver).")

    return df_out.reset_index(drop=True)


def liftover_entity_files(
        seg_by_entity: dict[str, pd.DataFrame],
        chain_file:    str | Path,
        liftover_bin:  str = "liftOver",
        min_match:     float = 0.95,
) -> dict[str, pd.DataFrame]:
    """
    Run liftover on every entity DataFrame in seg_by_entity.
    Returns a new dict with the same keys and lifted DataFrames.
    """
    return {
        entity: liftover_seg_df(
            seg_df,
            chain_file   = chain_file,
            liftover_bin = liftover_bin,
            min_match    = min_match,
        )
        for entity, seg_df in seg_by_entity.items()
    }