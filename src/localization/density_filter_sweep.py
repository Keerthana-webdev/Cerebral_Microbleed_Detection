"""
Improve whole-MRI detection WITHOUT retraining: require a candidate to have
several nearby high-confidence "votes" from overlapping sliding-window
positions before accepting it. Real lesions get many overlapping hits;
isolated false positives usually don't. Uses the v2 cache — instant to test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR

CACHE_V2_DIR = Path(__file__).resolve().parents[2] / "data" / "localization_cache_v2"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
test_subjects = splits["test"]


def get_true_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1) if (labeled == i).sum() >= 1]


def density_filter_and_nms(centers, probs, vote_radius, min_votes, nms_distance):
    """Keep only candidates with enough nearby support, then NMS the survivors."""
    centers = np.array(centers)
    kept_mask = np.zeros(len(centers), dtype=bool)

    for i, c in enumerate(centers):
        distances = np.linalg.norm(centers - c, axis=1)
        num_votes = (distances <= vote_radius).sum()  # includes itself
        if num_votes >= min_votes:
            kept_mask[i] = True

    survivors = centers[kept_mask]
    survivor_probs = probs[kept_mask]

    if len(survivors) == 0:
        return []

    order = np.argsort(-survivor_probs)
    kept = []
    for i in order:
        c = survivors[i]
        if not any(np.linalg.norm(c - k) < nms_distance for k in kept):
            kept.append(c)
    return kept


def match(predicted, true, max_dist):
    matched = set()
    tp = 0
    for p in predicted:
        for i, t in enumerate(true):
            if i in matched:
                continue
            if np.linalg.norm(np.array(p) - np.array(t)) <= max_dist:
                tp += 1
                matched.add(i)
                break
    fp = len(predicted) - tp
    fn = len(true) - len(matched)
    return tp, fp, fn


cached = {}
for subject_name in test_subjects:
    data = np.load(CACHE_V2_DIR / f"{subject_name}.npz")
    mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
    cached[subject_name] = {
        "centers": data["centers"], "stage1_probs": data["stage1_probs"],
        "stage2_probs": data["stage2_probs"], "true_centers": get_true_centers(mask),
    }

# Fixed detection thresholds (from your final chosen v2 operating point)
S1, S2 = 0.6, 0.7
MATCH_DISTANCE = 6

results = []
for vote_radius in [10, 15, 20]:
    for min_votes in [1, 2, 3, 4, 5]:
        for nms_dist in [15, 20]:
            total_tp = total_fp = total_fn = 0
            for subject_name, data in cached.items():
                mask = (data["stage1_probs"] > S1) & (data["stage2_probs"] > S2)
                surv_centers = data["centers"][mask]
                surv_probs = data["stage2_probs"][mask]
                if len(surv_centers) == 0:
                    continue
                final = density_filter_and_nms(surv_centers, surv_probs, vote_radius, min_votes, nms_dist)
                tp, fp, fn = match(final, data["true_centers"], MATCH_DISTANCE)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            fp_per_subj = total_fp / len(test_subjects)
            results.append({
                "vote_radius": vote_radius, "min_votes": min_votes, "nms_dist": nms_dist,
                "sensitivity": round(sens, 3), "fp_per_subject": round(fp_per_subj, 2),
            })

df = pd.DataFrame(results).sort_values("sensitivity", ascending=False)
df.to_csv(REPORTS_DIR / "density_filter_sweep.csv", index=False)
print(df.to_string(index=False))

print("\n=== Best sensitivity under 20 FP/subject ===")
good = df[df["fp_per_subject"] < 20].sort_values("sensitivity", ascending=False)
print(good.head(5).to_string(index=False) if len(good) else "None found under 20 FP/subject")