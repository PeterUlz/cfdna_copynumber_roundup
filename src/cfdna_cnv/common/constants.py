"""
Shared constants: genome coordinates, CNA thresholds, palette.
"""

# ---------------------------------------------------------------------------
# hg38 chromosome lengths and ordering
# ---------------------------------------------------------------------------
CHROM_LENGTHS: dict[str, int] = {
    "chr1":  248_956_422, "chr2":  242_193_529, "chr3":  198_295_559,
    "chr4":  190_214_555, "chr5":  181_538_259, "chr6":  170_805_979,
    "chr7":  159_345_973, "chr8":  145_138_636, "chr9":  138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16":  90_338_345, "chr17":  83_257_441, "chr18":  80_373_285,
    "chr19":  58_617_616, "chr20":  64_444_167, "chr21":  46_709_983,
    "chr22":  50_818_468, "chrX":  156_040_895,
}
CHROM_ORDER: list[str] = list(CHROM_LENGTHS.keys())

# Cumulative genome offsets for linear plotting
_OFFSETS: dict[str, int] = {}
_offset = 0
for _c in CHROM_ORDER:
    _OFFSETS[_c] = _offset
    _offset += CHROM_LENGTHS[_c]
GENOME_SIZE: int = _offset

VALID_CHROMS: set[str] = set(CHROM_LENGTHS) | {"chrY", "chrM", "chrMT"}

# ---------------------------------------------------------------------------
# CNA calling thresholds
# ---------------------------------------------------------------------------
DIPLOID_CN:        int   = 2
AMP_MIN_CN:        int   = 3      # integer CN >= 3 → AMP
FOCAL_MIN_BP:      int   = 50_000
FOCAL_MAX_BP:      int   = 500_000
LOG2_AMP_BROAD:    float =  0.2   # broad calling: log2 > 0.2 → AMP
LOG2_DEL_BROAD:    float = -0.2   # broad calling: log2 < -0.2 → DEL
LOG2_AMP_FOCAL:    float =  0.5   # focal amp: minimum absolute log2
LOG2_DELTA_FOCAL:  float =  0.2   # focal amp: must exceed background by this

# ---------------------------------------------------------------------------
# Segment validation bounds (hg38)
# ---------------------------------------------------------------------------
CN_MAX:     int = 20
SEG_MIN_BP: int = 1_000

# ---------------------------------------------------------------------------
# Plotting palette (cbio entities)
# ---------------------------------------------------------------------------
ENTITY_PALETTE: dict[str, dict] = {
    "breast":     {"gain": "#e63946", "loss": "#457b9d", "label": "Breast Cancer"},
    "prostate":   {"gain": "#f4a261", "loss": "#264653", "label": "Prostate Cancer"},
    "colorectal": {"gain": "#2a9d8f", "loss": "#e9c46a", "label": "Colorectal Cancer"},
}
