"""
Final comprehensive evaluation: computes every standard metric your
literature survey reports, at BOTH levels:
  1. Patch-level (curated test patches) — Precision, Recall, F1, Specificity, Accuracy
  2. Whole-scan level (sliding window) — False Positives per Scan, Sensitivity
Produces one clean summary table for your report.
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

DEVICE = torch.device("cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------- PART 1: Patch-level metrics ----------
def evaluate_patch_level():
    stage1_model = SimpleCNN3D().to(DEVICE)
    stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
    stage1_model.eval()

    stage2_model = SimpleCNN3D().to(DEVICE)
    stage2_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
    stage2_model.eval()

    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    test_subjects = set(splits["test"])

    metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
    test_metadata = metadata[metadata["subject"].isin(test_subjects)].reset_index(drop=True)

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for _, row in test_metadata.iterrows():
            patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            true_label = row["label"]

            s1_prob = torch.sigmoid(stage1_model(patch_tensor)).item()
            if s1_prob > 0.5:
                s2_prob = torch.sigmoid(stage2_model(patch_tensor)).item()
                predicted = 1 if s2_prob > 0.5 else 0
            else:
                predicted = 0

            if predicted == 1 and true_label == 1:
                tp += 1
            elif predicted == 1 and true_label == 0:
                fp += 1
            elif predicted == 0 and true_label == 0:
                tn += 1
            elif predicted == 0 and true_label == 1:
                fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)                 # sensitivity
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("=== PATCH-LEVEL METRICS (curated test patches) ===")
    print(f"Confusion Matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall/Sens: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"F1-Score:    {f1:.3f}\n")

    return {
        "level": "patch", "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "specificity": round(specificity, 3), "accuracy": round(accuracy, 3),
        "f1_score": round(f1, 3),
    }


# ---------- PART 2: Whole-scan (localization) metrics ----------
def evaluate_scan_level():
    sweep_path = REPORTS_DIR / "threshold_sweep.csv"
    if not sweep_path.exists():
        print("⚠️  threshold_sweep.csv not found — skipping scan-level summary. Run Lesson 14 first.")
        return None

    sweep = pd.read_csv(sweep_path)
    # Selected operating point from the FROC analysis (Lesson 14)
    chosen = sweep[
        (sweep["stage1_thresh"] == 0.7) &
        (sweep["stage2_thresh"] == 0.7) &
        (sweep["nms_distance"] == 15)
    ]
    if chosen.empty:
        chosen = sweep.iloc[[0]]  # fallback

    row = chosen.iloc[0]
    print("=== WHOLE-SCAN LEVEL METRICS (sliding window + NMS) ===")
    print(f"Operating point: stage1_thresh={row['stage1_thresh']}, "
          f"stage2_thresh={row['stage2_thresh']}, nms_distance={row['nms_distance']}")
    print(f"Sensitivity (lesion-level): {row['sensitivity']:.3f}")
    print(f"False Positives per Scan:   {row['fp_per_subject']:.2f}\n")

    return {
        "level": "whole_scan",
        "sensitivity": row["sensitivity"],
        "fp_per_scan": row["fp_per_subject"],
    }


def main():
    patch_results = evaluate_patch_level()
    scan_results = evaluate_scan_level()

    summary = {"patch_level": patch_results, "whole_scan_level": scan_results}
    out_path = REPORTS_DIR / "final_evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full summary saved to: {out_path}")
    print("\n✅ This JSON has every number you need for your report's Results section.")


if __name__ == "__main__":
    main()