"""
Central configuration — the one file that knows where everything lives.
Every other script will import settings from here instead of repeating paths.
"""

from pathlib import Path

# ---- Paths (matches YOUR actual folder names) ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # goes up from src/preprocessing to CMB_DETECTION
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
PATCHES_DIR = PROJECT_ROOT / "data" / "patches"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# ---- Filenames (confirmed from YOUR sub-101 folder) ----
SWI_PATTERN = "{subj}_space-T2S_desc-masked_T2S.nii.gz"   # the scan we detect CMBs on
T1_PATTERN = "{subj}_space-T2S_desc-masked_T1.nii.gz"      # optional context
T2_PATTERN = "{subj}_space-T2S_desc-masked_T2.nii.gz"      # optional context
MASK_PATTERN = "{subj}_space-T2S_CMB.nii.gz"               # the answer key

# ---- Preprocessing parameters (we'll use these properly in a later lesson) ----
TARGET_SPACING = (1.0, 1.0, 1.0)
INTENSITY_CLIP_PERCENTILES = (0.5, 99.5)