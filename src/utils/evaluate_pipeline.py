"""
Lesson 10: Run the FULL two-stage pipeline on the TEST set (subjects neither
model has ever seen) and compare:
  - Stage 1 alone (candidate detector only)
  - Stage 1 + Stage 2 (candidate detector -> mimic classifier)
This before/after comparison is your project's headline result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PATCHES_DIR, SPLITS_DIR
from model import SimpleCNN3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Load BOTH trained models
stage1_model = SimpleCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
stage1_model.eval()

stage2_model = SimpleCNN3D().to(DEVICE)
stage2_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
stage2_model.eval()

# Only the TEST subjects — untouched until now
with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
test_subjects = set(splits["test"])

metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
test_metadata = metadata[metadata["subject"].isin(test_subjects)].reset_index(drop=True)
print(f"Evaluating on {len(test_metadata)} test patches from {len(test_subjects)} subjects\n")


def predict(model, patch_tensor):
    with torch.no_grad():
        return torch.sigmoid(model(patch_tensor)).item()


# ---- Counters ----
stage1_tp = stage1_fp = stage1_fn = 0
final_tp = final_fp = final_fn = 0

for _, row in test_metadata.iterrows():
    patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    true_label = row["label"]

    stage1_prob = predict(stage1_model, patch_tensor)
    stage1_says_positive = stage1_prob > 0.5

    # --- Stage 1 alone scoring ---
    if stage1_says_positive and true_label == 1:
        stage1_tp += 1
    elif stage1_says_positive and true_label == 0:
        stage1_fp += 1
    elif (not stage1_says_positive) and true_label == 1:
        stage1_fn += 1

    # --- Full pipeline: only patches Stage 1 flagged go through Stage 2 ---
    if stage1_says_positive:
        stage2_prob = predict(stage2_model, patch_tensor)
        final_says_positive = stage2_prob > 0.5
    else:
        final_says_positive = False  # Stage 1 already rejected it

    if final_says_positive and true_label == 1:
        final_tp += 1
    elif final_says_positive and true_label == 0:
        final_fp += 1
    elif (not final_says_positive) and true_label == 1:
        final_fn += 1


def summarize(name, tp, fp, fn, n_subjects):
    sensitivity = tp / (tp + fn + 1e-8)
    fp_per_subject = fp / n_subjects
    print(f"--- {name} ---")
    print(f"  Sensitivity (real microbleeds caught): {sensitivity:.3f}")
    print(f"  False positives: {fp}")
    print(f"  False positives per subject: {fp_per_subject:.2f}")
    print()
    return sensitivity, fp_per_subject


print("=" * 50)
s1_sens, s1_fp = summarize("STAGE 1 ONLY (candidate detector)", stage1_tp, stage1_fp, stage1_fn, len(test_subjects))
s2_sens, s2_fp = summarize("FULL PIPELINE (stage 1 + mimic classifier)", final_tp, final_fp, final_fn, len(test_subjects))
print("=" * 50)
print(f"RESULT: Mimic classifier reduced false positives per subject from "
      f"{s1_fp:.2f} to {s2_fp:.2f} "
      f"({(1 - s2_fp/max(s1_fp, 1e-8)) * 100:.1f}% reduction)")
print(f"        while sensitivity changed from {s1_sens:.3f} to {s2_sens:.3f}")