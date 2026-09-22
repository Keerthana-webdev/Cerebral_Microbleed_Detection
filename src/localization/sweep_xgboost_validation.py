import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgboost.feature_extraction import extract_features

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgboost_cmb.json"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

NMS_DISTANCE = 15.0
MATCH_DISTANCE = 6.0

# Whole-MRI thresholds to investigate
THRESHOLDS = [
    0.59,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
]

BATCH_SIZE = 512

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

subjects = splits["val"]

print("=" * 80)
print("XGBOOST WHOLE-MRI VALIDATION THRESHOLD SWEEP")
print("=" * 80)

print("Validation subjects:", len(subjects))
print("Thresholds:", THRESHOLDS)

print()
print("Loading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

print("Model loaded.")

def is_valid_patch(patch):

    return (
        np.std(patch) > 0.3
        and np.mean(patch) > -1.5
    )


def get_windows(volume):

    pz, py, px = PATCH_SIZE
    sz, sy, sx = STRIDE

    z_dim, y_dim, x_dim = volume.shape

    patches = []
    centers = []

    for z in range(0, z_dim - pz + 1, sz):

        for y in range(0, y_dim - py + 1, sy):

            for x in range(0, x_dim - px + 1, sx):

                patch = volume[
                    z:z + pz,
                    y:y + py,
                    x:x + px
                ]

                if not is_valid_patch(patch):
                    continue

                patches.append(patch)

                centers.append(
                    (
                        z + pz // 2,
                        y + py // 2,
                        x + px // 2
                    )
                )

    return patches, centers


def get_gt_centroids(mask):

    labeled, count = ndimage.label(mask > 0)

    centroids = []

    objects = ndimage.find_objects(labeled)

    for label_id, slices in enumerate(objects, start=1):

        if slices is None:
            continue

        coords = np.argwhere(
            labeled[slices] == label_id
        )

        if len(coords) == 0:
            continue

        offsets = np.array(
            [s.start for s in slices],
            dtype=float
        )

        centroid = coords.mean(axis=0) + offsets

        centroids.append(tuple(centroid))

    return centroids


def nms(detections):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["probability"],
        reverse=True
    )

    selected = []

    for candidate in detections:

        candidate_center = np.array(
            candidate["center"],
            dtype=float
        )

        keep = True

        for selected_detection in selected:

            selected_center = np.array(
                selected_detection["center"],
                dtype=float
            )

            distance = np.linalg.norm(
                candidate_center - selected_center
            )

            if distance < NMS_DISTANCE:

                keep = False
                break

        if keep:
            selected.append(candidate)

    return selected


def match(detections, gt_centroids):

    if len(gt_centroids) == 0:

        return 0, len(detections), 0

    matched = set()

    tp = 0
    fp = 0

    for detection in detections:

        center = np.array(
            detection["center"],
            dtype=float
        )

        best_distance = float("inf")
        best_gt = None

        for i, gt in enumerate(gt_centroids):

            if i in matched:
                continue

            gt_array = np.array(
                gt,
                dtype=float
            )

            distance = np.linalg.norm(
                center - gt_array
            )

            if distance < best_distance:

                best_distance = distance
                best_gt = i

        if (
            best_gt is not None
            and best_distance <= MATCH_DISTANCE
        ):

            matched.add(best_gt)
            tp += 1

        else:

            fp += 1

    fn = len(gt_centroids) - len(matched)

    return tp, fp, fn

cache = {}

for subject in subjects:

    print()
    print("-" * 70)
    print("Preparing:", subject)

    subject_dir = PREPROCESSED_DIR / subject

    volume = np.load(
        subject_dir / "swi.npy"
    )

    mask = np.load(
        subject_dir / "mask.npy"
    )

    gt_centroids = get_gt_centroids(mask)

    patches, centers = get_windows(volume)

    print("MRI shape:", volume.shape)
    print("Valid windows:", len(patches))
    print("Ground-truth lesions:", len(gt_centroids))

    features = []

    for patch in patches:

        features.append(
            extract_features(patch)
        )

    X = np.stack(features).astype(
        np.float32
    )

    probabilities = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        batch = X[start:end]

        probabilities.extend(
            model.predict_proba(batch)[:, 1]
        )

    probabilities = np.asarray(
        probabilities
    )

    cache[subject] = {
        "centers": centers,
        "probabilities": probabilities,
        "gt": gt_centroids,
    }

    print("Prediction cache ready.")

results = []

print()
print("=" * 80)
print("RUNNING THRESHOLD SWEEP")
print("=" * 80)

for threshold in THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true = 0

    print()
    print(f"Threshold = {threshold:.2f}")

    for subject in subjects:

        data = cache[subject]

        centers = data["centers"]
        probabilities = data["probabilities"]
        gt = data["gt"]

        detections = []

        for center, probability in zip(
            centers,
            probabilities
        ):

            if probability >= threshold:

                detections.append(
                    {
                        "center": center,
                        "probability": float(
                            probability
                        ),
                    }
                )

        detections = nms(detections)

        tp, fp, fn = match(
            detections,
            gt
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true += len(gt)

    sensitivity = (
        total_tp / total_true
        if total_true > 0
        else 0
    )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0
    )

    fp_per_scan = (
        total_fp / len(subjects)
    )

    print(
        f"TP={total_tp} "
        f"FP={total_fp} "
        f"FN={total_fn} "
        f"Sensitivity={sensitivity * 100:.2f}% "
        f"Precision={precision * 100:.2f}% "
        f"FP/scan={fp_per_scan:.2f}"
    )

    results.append(
        {
            "threshold": threshold,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "sensitivity": sensitivity,
            "precision": precision,
            "fp_per_scan": fp_per_scan,
        }
    )

df = pd.DataFrame(results)

output_file = (
    REPORT_DIR /
    "xgboost_whole_mri_validation_threshold_sweep.csv"
)

df.to_csv(
    output_file,
    index=False
)

print()
print("=" * 80)
print("FINAL THRESHOLD SWEEP")
print("=" * 80)

print(
    df.to_string(index=False)
)

print()
print("Saved:")
print(output_file)