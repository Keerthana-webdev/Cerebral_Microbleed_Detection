"""
ResNet3D candidate-support filtering.

Uses ONLY validation cached predictions.

Idea:
Instead of treating every high-probability sliding-window patch as an
independent detection, group nearby high-probability windows together.

This helps reduce isolated false positives produced during whole-MRI scanning.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

CACHE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "resnet3d"
    / "validation_cache"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "resnet3d_candidate_support_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
]

# Maximum distance for windows to be considered neighbours.
NEIGHBOR_DISTANCE = 12.0

# Minimum number of nearby positive windows required
# for a candidate to survive.
SUPPORT_COUNTS = [
    1,
    2,
    3,
    4,
    5,
]

# Distance used to compare predictions against true CMB centroids.
MATCH_DISTANCE = 6.0


# ============================================================
# LOAD VALIDATION SUBJECTS
# ============================================================

def load_validation_subjects():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        splits = json.load(f)

    return splits["val"]


# ============================================================
# GROUND TRUTH
# ============================================================

def get_ground_truth(mask):

    from scipy import ndimage

    binary = mask > 0

    labeled, num = ndimage.label(binary)

    centers = []

    for i in range(1, num + 1):

        coords = np.argwhere(
            labeled == i
        )

        if len(coords) == 0:
            continue

        centroid = coords.mean(axis=0)

        centers.append(
            centroid
        )

    return centers


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return np.linalg.norm(
        np.asarray(a, dtype=float)
        -
        np.asarray(b, dtype=float)
    )

def apply_support_filter(
    centers,
    probabilities,
    threshold,
    support_count
):

    positive_indices = np.where(
        probabilities >= threshold
    )[0]

    if len(positive_indices) == 0:
        return []

    positive_centers = centers[
        positive_indices
    ]

    positive_probabilities = probabilities[
        positive_indices
    ]

    detections = []

    for i in range(
        len(positive_centers)
    ):

        center = positive_centers[i]

        support = 0

        for j in range(
            len(positive_centers)
        ):

            if i == j:
                continue

            d = distance(
                center,
                positive_centers[j]
            )

            if d <= NEIGHBOR_DISTANCE:

                support += 1

        # Include the candidate itself.
        total_support = support + 1

        if total_support >= support_count:

            detections.append(
                {
                    "center": center,
                    "probability":
                        float(
                            positive_probabilities[i]
                        ),
                    "support":
                        total_support,
                }
            )

    return detections

def apply_nms(
    detections,
    nms_distance=15.0
):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["probability"],
        reverse=True
    )

    selected = []

    for detection in detections:

        keep = True

        for chosen in selected:

            if distance(
                detection["center"],
                chosen["center"]
            ) < nms_distance:

                keep = False
                break

        if keep:

            selected.append(
                detection
            )

    return selected

def match_detections(
    detections,
    gt_centers
):

    matched_gt = set()

    tp = 0
    fp = 0

    for detection in detections:

        center = detection["center"]

        best_distance = float("inf")
        best_index = None

        for i, gt in enumerate(
            gt_centers
        ):

            if i in matched_gt:
                continue

            d = distance(
                center,
                gt
            )

            if d < best_distance:

                best_distance = d
                best_index = i

        if (
            best_index is not None
            and best_distance <= MATCH_DISTANCE
        ):

            tp += 1
            matched_gt.add(
                best_index
            )

        else:

            fp += 1

    fn = (
        len(gt_centers)
        -
        len(matched_gt)
    )

    return tp, fp, fn

def evaluate(
    subjects,
    threshold,
    support_count
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for subject in subjects:

        cache_file = (
            CACHE_DIR
            / f"{subject}.npz"
        )

        data = np.load(
            cache_file
        )

        centers = data[
            "centers"
        ]

        probabilities = data[
            "probabilities"
        ]

        mask = data[
            "mask"
        ]

        detections = apply_support_filter(
            centers,
            probabilities,
            threshold,
            support_count
        )

        detections = apply_nms(
            detections
        )

        gt_centers = get_ground_truth(
            mask
        )

        tp, fp, fn = match_detections(
            detections,
            gt_centers
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    sensitivity = (
        total_tp
        /
        (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    precision = (
        total_tp
        /
        (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    fp_per_scan = (
        total_fp / len(subjects)
    )

    return {
        "threshold": threshold,
        "support_count": support_count,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "sensitivity": sensitivity,
        "FP_per_scan": fp_per_scan,
        "precision": precision,
    }

def main():

    print("=" * 70)
    print(
        "RESNET3D CANDIDATE SUPPORT FILTER"
    )
    print("=" * 70)

    subjects = load_validation_subjects()

    print(
        f"Validation subjects: "
        f"{len(subjects)}"
    )

    print(
        f"Neighbour distance: "
        f"{NEIGHBOR_DISTANCE}"
    )

    print(
        f"Match distance: "
        f"{MATCH_DISTANCE}"
    )

    results = []

    for threshold in THRESHOLDS:

        for support_count in SUPPORT_COUNTS:

            result = evaluate(
                subjects,
                threshold,
                support_count
            )

            results.append(
                result
            )

            print(
                f"Threshold "
                f"{threshold:.2f} | "
                f"Support >= "
                f"{support_count} | "
                f"Sensitivity: "
                f"{result['sensitivity']:.4f} | "
                f"FP/scan: "
                f"{result['FP_per_scan']:.2f} | "
                f"Precision: "
                f"{result['precision']:.4f} | "
                f"TP: {result['TP']} | "
                f"FP: {result['FP']} | "
                f"FN: {result['FN']}"
            )

    df = pd.DataFrame(
        results
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print(
        "CANDIDATE SUPPORT FILTER COMPLETE"
    )
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )

    print("\nSaved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()