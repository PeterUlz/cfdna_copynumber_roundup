# LOCATE

Unified toolkit for copy-number analysis of cell-free DNA samples from two sources:
- **LOCATE** output (broad CNAs + focal amplifications + longitudinal comparison)
- **cBioPortal** segmentation files (broad CNAs + focal amplifications across entities)

---

## Installation (pixi)

```bash
cd cfdna_copy_number_roundup
pixi install
```

This installs the package in editable mode via the `pypi-dependencies` entry in
`pixi.toml`, so any edits to `src/cfdna_cnv/` are immediately reflected without
reinstalling.

---

## Usage

### LOCATE workflow

```bash
pixi run cfdna-cnv locate \
    --dir-1mb    output_1Mbp \
    --dir-50kb   output_50kbp \
    --output     results/locate \
    --cosmic-cgc ref/CancerGeneCensus.tsv

# Skip validation for large cohorts
pixi run cfdna-cnv locate \
    --dir-1mb  output_1Mbp \
    --dir-50kb output_50kbp \
    --skip-validation \
    --focal-min-bp  50000 \
    --focal-max-bp 500000
```

**All LOCATE arguments:**

| Flag | Default | Description |
|---|---|---|
| `--dir-1mb` | required | 1 Mbp LOCATE input directory (ichorCNA output) |
| `--dir-50kb` | required | 50 kbp LOCATE input directory (ichorCNA output) |
| `--output` | `locate_results` | Output directory |
| `--tf-threshold` | `0.03` | Minimum tumor fraction (3%) |
| `--amp-min-cn` | `3` | Minimum CN to call amplification |
| `--focal-min-bp` | `50000` | Minimum focal amp segment size |
| `--focal-max-bp` | `500000` | Maximum focal amp segment size |
| `--log2-delta` | `0.2` | Min log2 elevation above local 1 Mbp background |
| `--flank-bp` | `1000000` | Flanking window for background calculation |
| `--skip-validation` | off | Skip seg file validation |
| `--cosmic-cgc` | None | COSMIC CGC file for annotation |

**Output files:**

| File | Contents |
|---|---|
| `tf_summary.tsv` | Tumor fractions for all samples |
| `incomplete_runs.tsv` | Samples with missing output files |
| `seg_validation.tsv` | Per-sample seg file QC (if not skipped) |
| `broad_cna_frequencies.tsv` | Per-locus AMP/DEL frequencies (1 Mbp) |
| `focal_amplifications.tsv` | All focal amp events with COSMIC annotation |
| `focal_amp_frequencies.tsv` | Population-level focal amp frequencies |
| `longitudinal_comparison.tsv` | CN changes across timepoints per patient |

---

### cBioPortal workflow

```bash
pixi run cfdna-cnv cbio \
    --breast      data/breast.seg \
    --prostate    data/prostate.seg \
    --colorectal  data/colorectal.seg \
    --output      results/cbio \
    --cosmic-cgc  ref/CancerGeneCensus.tsv

# Multiple files per entity (semicolon-separated)
pixi run cfdna-cnv cbio \
    --breast "cohort1.seg;cohort2.seg" \
    --output results/cbio

# Custom entities
pixi run cfdna-cnv cbio \
    --seg lung=lung.seg \
    --seg "ovarian=ov_tcga.seg;ov_geo.seg"
```

**All cBioPortal arguments:**

| Flag | Default | Description |
|---|---|---|
| `--breast` / `--prostate` / `--colorectal` | None | Seg file(s) per entity |
| `--seg NAME=FILE` | None | Custom entity (repeatable) |
| `--output` | `cbio_cna_results` | Output directory |
| `--bin-size` | `5000000` | Bin size for broad CNA frequencies |
| `--gain-threshold` | `0.2` | log2 cutoff for GAIN |
| `--loss-threshold` | `-0.2` | log2 cutoff for LOSS |
| `--min-seg-size` | `0` | Drop segments shorter than this |
| `--focal-min-bp` | `50000` | Min focal amp size |
| `--focal-max-bp` | `500000` | Max focal amp size |
| `--focal-log2-min` | `0.5` | Min absolute seg.mean for focal amp |
| `--focal-log2-delta` | `0.2` | Min elevation above local background |
| `--focal-flank-bp` | `10000000` | Flanking window for background |
| `--focal-min-freq` | `0.01` | Min frequency shown in focal amp plot |
| `--cosmic-cgc` | None | COSMIC CGC file for annotation |

---

## Package structure

```
src/cfdna_cnv/
├── cli.py                   ← entry point, subcommand routing
├── commands/
│   ├── locate_cmd.py      ← LOCATE workflow (7 steps)
│   └── cbio_cmd.py          ← cBioPortal workflow (4 steps)
├── common/
│   ├── constants.py         ← hg38 lengths, thresholds, palette
│   ├── io.py                ← file discovery, seg parsing, alteration calling
│   └── cosmic.py            ← COSMIC CGC loader + annotator
├── locate/
│   ├── qc.py                ← incomplete run detection, seg validation
│   ├── broad_cna.py         ← streaming broad CNA frequencies
│   ├── focal_amp.py         ← focal amp detection + streaming + aggregation
│   └── longitudinal.py      ← longitudinal CN comparison
└── cbio/
    ├── loader.py            ← seg loading, bin creation
    ├── frequencies.py       ← broad + focal CNA frequencies
    └── plots.py             ← all matplotlib output
```