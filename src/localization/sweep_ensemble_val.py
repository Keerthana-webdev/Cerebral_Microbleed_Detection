"""
Lesson 26b: Sweep thresholds for whole-MRI ensemble detection on the
VALIDATION set, testing both averaged-probability and majority-vote
combination rules against the single-model baselines.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "ensemble_cache_val"
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


# Load cache
cached = {}
for subject_name in val_subjects:
    data = np.load(CACHE_DIR / f"{subject_name}.npz")
    mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
    cached[subject_name] = {
        "centers": data["centers"],
        "cnn_probs": data["cnn_probs"],
        "resnet_probs": data["resnet_probs"],
        "densenet_probs": data["densenet_probs"],
        "true_centers": get_ground_truth_centers(mask),
    }

MATCH_DISTANCE = 6
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
nms_distances = [15, 20, 25]

results = []

for combo_rule in ["average", "majority_vote"]:
    for thresh in thresholds:
        for nms in nms_distances:
            total_tp = total_fp = total_fn = 0
            for subject_name, data in cached.items():
                cnn_p = data["cnn_probs"]
                resnet_p = data["resnet_probs"]
                densenet_p = data["densenet_probs"]

                if combo_rule == "average":
                    combined_prob = (cnn_p + resnet_p + densenet_p) / 3.0
                    passed = combined_prob > thresh
                    scores = combined_prob
                else:  # majority_vote: at least 2 of 3 individually exceed threshold
                    votes = (cnn_p > thresh).astype(int) + (resnet_p > thresh).astype(int) + (densenet_p > thresh).astype(int)
                    passed = votes >= 2
                    scores = (cnn_p + resnet_p + densenet_p) / 3.0  # use avg prob for NMS ranking

                surv_centers = data["centers"][passed]
                surv_scores = scores[passed]

                final = apply_nms(surv_centers, surv_scores, nms) if len(surv_centers) else []
                tp, fp, fn = match_to_ground_truth(final, data["true_centers"], MATCH_DISTANCE)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            fp_per_subj = total_fp / len(val_subjects)
            results.append({
                "combo_rule": combo_rule, "threshold": thresh, "nms": nms,
                "sensitivity": round(sens, 3), "fp_per_subject": round(fp_per_subj, 2),
            })

df = pd.DataFrame(results)
df.to_csv(REPORTS_DIR / "ensemble_wholemri_val_sweep.csv", index=False)

print("=" * 80)
print("ENSEMBLE WHOLE-MRI SWEEP (VALIDATION) — full results")
print("=" * 80)
print(df.sort_values("sensitivity", ascending=False).to_string(index=False))

print("\n=== Best sensitivity at FP/subject < 30 ===")
practical = df[df["fp_per_subject"] < 30].sort_values("sensitivity", ascending=False)
if len(practical):
    print(practical.head(5).to_string(index=False))
else:
    print("No point found under 30 FP/subject")

print(f"\n=== Comparison vs standalone v3 (validation, same s1=0.8,s2=0.8,nms=25 style point) ===")
print("Standalone 3D CNN v3 (from earlier sweep): sensitivity=0.238, FP/subject=19.82")

print(f"\nFull sweep saved to: {REPORTS_DIR / 'ensemble_wholemri_val_sweep.csv'}")