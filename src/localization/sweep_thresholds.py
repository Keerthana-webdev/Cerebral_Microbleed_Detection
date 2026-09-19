"""
Lesson 14d: Test many threshold + NMS combinations instantly using the
cached probabilities, and print a table so we can SEE the sensitivity vs
false-positive tradeoff and pick a justified operating point.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "localization_cache"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
test_subjects = splits["test"]


def get_ground_truth_centers(mask):
    labeled, num_blobs = ndimage.label(mask)
    centers = []
    for i in range(1, num_blobs + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) >= 1:
            centers.append(tuple(coords.mean(axis=0)))
    return centers


def apply_nms(centers, probs, min_distance):
    order = np.argsort(-probs)  # strongest first
    kept_centers, kept_probs = [], []
    for i in order:
        center = centers[i]
        too_close = any(np.linalg.norm(center - kc) < min_distance for kc in kept_centers)
        if not too_close:
            kept_centers.append(center)
            kept_probs.append(probs[i])
    return kept_centers


def match_to_ground_truth(predicted_centers, true_centers, max_distance):
    matched_true = set()
    tp = 0
    for pred in predicted_centers:
        for i, true in enumerate(true_centers):
            if i in matched_true:
                continue
            if np.linalg.norm(np.array(pred) - np.array(true)) <= max_distance:
                tp += 1
                matched_true.add(i)
                break
    fp = len(predicted_centers) - tp
    fn = len(true_centers) - len(matched_true)
    return tp, fp, fn


# Load all cached data once
cached = {}
for subject_name in test_subjects:
    data = np.load(CACHE_DIR / f"{subject_name}.npz")
    mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
    cached[subject_name] = {
        "centers": data["centers"],
        "stage1_probs": data["stage1_probs"],
        "stage2_probs": data["stage2_probs"],
        "true_centers": get_ground_truth_centers(mask),
    }

# Try a grid of threshold combinations
stage1_thresholds = [0.5, 0.6, 0.7]
stage2_thresholds = [0.5, 0.6, 0.7, 0.8]
nms_distances = [10, 15, 20]
MATCH_DISTANCE = 6

results = []
for s1_thresh in stage1_thresholds:
    for s2_thresh in stage2_thresholds:
        for nms_dist in nms_distances:
            total_tp, total_fp, total_fn = 0, 0, 0
            for subject_name, data in cached.items():
                # Stage 1 gate, then Stage 2 gate (both computed for every position already)
                passed_mask = (data["stage1_probs"] > s1_thresh) & (data["stage2_probs"] > s2_thresh)
                surviving_centers = data["centers"][passed_mask]
                surviving_probs = data["stage2_probs"][passed_mask]

                if len(surviving_centers) > 0:
                    final_centers = apply_nms(surviving_centers, surviving_probs, nms_dist)
                else:
                    final_centers = []

                tp, fp, fn = match_to_ground_truth(final_centers, data["true_centers"], MATCH_DISTANCE)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            fp_per_subject = total_fp / len(test_subjects)

            results.append({
                "stage1_thresh": s1_thresh,
                "stage2_thresh": s2_thresh,
                "nms_distance": nms_dist,
                "sensitivity": round(sensitivity, 3),
                "fp_per_subject": round(fp_per_subject, 2),
            })

results_df = pd.DataFrame(results).sort_values("sensitivity", ascending=False)
print(results_df.to_string(index=False))

out_path = REPORTS_DIR / "threshold_sweep.csv"
results_df.to_csv(out_path, index=False)
print(f"\nFull sweep saved to: {out_path}")

print("\n=== Top 5 by sensitivity, among those with FP/subject under 20 ===")
practical = results_df[results_df["fp_per_subject"] < 20].sort_values("sensitivity", ascending=False)
print(practical.head(5).to_string(index=False))