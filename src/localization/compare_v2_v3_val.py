"""
Lesson 21b: Compare v2 vs v3 whole-volume performance on the VALIDATION
set (test set still untouched). This decides whether v3 becomes our new
final model or whether v2 was already the better choice.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR

CACHE_V2_DIR = Path(__file__).resolve().parents[2] / "data" / "localization_cache_val"
CACHE_V3_DIR = Path(__file__).resolve().parents[2] / "data" / "localization_cache_v3_val"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
val_subjects = splits["val"]


def get_ground_truth_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1) if (labeled == i).sum() >= 1]


def apply_nms(centers, probs, min_distance):
    order = np.argsort(-probs)
    kept = []
    for i in order:
        c = centers[i]
        if not any(np.linalg.norm(c - kc) < min_distance for kc in kept):
            kept.append(c)
    return kept


def match_to_ground_truth(predicted, true, max_distance):
    matched = set()
    tp = 0
    for p in predicted:
        for i, t in enumerate(true):
            if i in matched:
                continue
            if np.linalg.norm(np.array(p) - np.array(t)) <= max_distance:
                tp += 1
                matched.add(i)
                break
    fp = len(predicted) - tp
    fn = len(true) - len(matched)
    return tp, fp, fn


def load_cache(cache_dir):
    cached = {}
    for subject_name in val_subjects:
        data = np.load(cache_dir / f"{subject_name}.npz")
        mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
        cached[subject_name] = {
            "centers": data["centers"],
            "stage1_probs": data["stage1_probs"],
            "stage2_probs": data["stage2_probs"],
            "true_centers": get_ground_truth_centers(mask),
        }
    return cached


def sweep(cached, label):
    stage1_thresholds = [0.5, 0.6, 0.7, 0.8]
    stage2_thresholds = [0.5, 0.6, 0.7, 0.8]
    nms_distances = [15, 20, 25]
    MATCH_DISTANCE = 6

    results = []
    for s1 in stage1_thresholds:
        for s2 in stage2_thresholds:
            for nms in nms_distances:
                total_tp = total_fp = total_fn = 0
                for subject_name, data in cached.items():
                    mask = (data["stage1_probs"] > s1) & (data["stage2_probs"] > s2)
                    surv_centers = data["centers"][mask]
                    surv_probs = data["stage2_probs"][mask]
                    final = apply_nms(surv_centers, surv_probs, nms) if len(surv_centers) else []
                    tp, fp, fn = match_to_ground_truth(final, data["true_centers"], MATCH_DISTANCE)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                fp_per_subj = total_fp / len(val_subjects)
                results.append({"model": label, "s1": s1, "s2": s2, "nms": nms,
                                 "sensitivity": round(sens, 3), "fp_per_subject": round(fp_per_subj, 2)})
    return pd.DataFrame(results)


print("Sweeping v2 on VALIDATION...")
v2_cache = load_cache(CACHE_V2_DIR)
v2_results = sweep(v2_cache, "v2")

print("Sweeping v3 on VALIDATION...")
v3_cache = load_cache(CACHE_V3_DIR)
v3_results = sweep(v3_cache, "v3")

combined = pd.concat([v2_results, v3_results], ignore_index=True)
combined.to_csv(REPORTS_DIR / "v2_vs_v3_val_sweep.csv", index=False)

print("\n=== Best sensitivity at FP/subject < 50, per model (VALIDATION) ===")
for label in ["v2", "v3"]:
    subset = combined[(combined["model"] == label) & (combined["fp_per_subject"] < 50)]
    if len(subset):
        best = subset.sort_values("sensitivity", ascending=False).iloc[0]
        print(f"{label}: sensitivity={best['sensitivity']:.3f} at FP/subject={best['fp_per_subject']:.2f} "
              f"(s1={best['s1']}, s2={best['s2']}, nms={best['nms']})")
    else:
        print(f"{label}: no point found under 50 FP/subject")

print("\n=== FP/subject at matched sensitivity ~0.15-0.25 (VALIDATION) ===")
for label in ["v2", "v3"]:
    subset = combined[(combined["model"] == label) & (combined["sensitivity"] >= 0.15) & (combined["sensitivity"] <= 0.25)]
    if len(subset):
        best = subset.sort_values("fp_per_subject").iloc[0]
        print(f"{label}: sensitivity={best['sensitivity']:.3f}, FP/subject={best['fp_per_subject']:.2f}")
    else:
        print(f"{label}: no point in that sensitivity range")

print(f"\nFull sweep saved to: {REPORTS_DIR / 'v2_vs_v3_val_sweep.csv'}")
print("\nDECISION RULE: if v3 beats v2 on validation, v3 becomes final. If not, keep v2 as final.")