"""
XGBoost whole-MRI localization

Runs the trained XGBoost patch classifier on sliding windows
across complete preprocessed MRI scans.

This script:
1. Loads the trained XGBoost model
2. Slides a (16,16,8) window through the MRI
3. Extracts the SAME 43 features used during training
4. Predicts CMB probability
5. Applies thresholding
6. Applies NMS
7. Matches detections with ground-truth CMB locations
8. Reports lesion sensitivity, FP/scan and precision
"""
import json
import sys
from anyio import Path


from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgboost.feature_extraction import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgboost_cmb.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xgboost"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

# Patch-level validation-selected threshold
DEFAULT_THRESHOLD = 0.59

NMS_DISTANCE = 15.0
MATCH_DISTANCE = 6.0

BATCH_SIZE = 512

def load_subjects(split_name):

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    return splits[split_name]

def find_subject_file(subject):

    subject_dir = PREPROCESSED_DIR / subject

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed directory not found: {subject_dir}"
        )

    swi_file = subject_dir / "swi.npy"
    mask_file = subject_dir / "mask.npy"

    if not swi_file.exists():
        raise FileNotFoundError(
            f"SWI file not found: {swi_file}"
        )

    if not mask_file.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_file}"
        )

    return swi_file, mask_file

def is_valid_patch(patch):

    patch_std = float(np.std(patch))
    patch_mean = float(np.mean(patch))

    return patch_std > 0.3 and patch_mean > -1.5

def generate_windows(volume):

    z_dim, y_dim, x_dim = volume.shape

    pz, py, px = PATCH_SIZE
    sz, sy, sx = STRIDE

    windows = []

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

                center = (
                    z + pz // 2,
                    y + py // 2,
                    x + px // 2
                )

                windows.append((patch, center))

    return windows

def non_max_suppression(detections, min_distance):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d["probability"],
        reverse=True
    )

    selected = []

    for candidate in detections:

        keep = True

        c = np.array(candidate["center"], dtype=float)

        for chosen in selected:

            p = np.array(chosen["center"], dtype=float)

            distance = np.linalg.norm(c - p)

            if distance < min_distance:
                keep = False
                break

        if keep:
            selected.append(candidate)

    return selected

def get_ground_truth_centroids(mask):

    binary_mask = mask > 0

    labeled, num_objects = ndimage.label(binary_mask)

    if num_objects == 0:
        return []

    objects = ndimage.find_objects(labeled)

    centroids = []

    for label_id, slices in enumerate(objects, start=1):

        if slices is None:
            continue

        coords = np.argwhere(labeled[slices] == label_id)

        if len(coords) == 0:
            continue

        offsets = np.array(
            [s.start for s in slices],
            dtype=float
        )

        centroid = coords.mean(axis=0) + offsets

        centroids.append(tuple(centroid))

    return centroids

def match_detections(detections, gt_centroids):

    if len(gt_centroids) == 0:

        return 0, len(detections), 0

    matched_gt = set()

    tp = 0
    fp = 0

    for detection in detections:

        center = np.array(
            detection["center"],
            dtype=float
        )

        best_distance = float("inf")
        best_gt = None

        for gt_index, gt in enumerate(gt_centroids):

            if gt_index in matched_gt:
                continue

            gt_array = np.array(gt, dtype=float)

            distance = np.linalg.norm(
                center - gt_array
            )

            if distance < best_distance:

                best_distance = distance
                best_gt = gt_index

        if (
            best_gt is not None
            and best_distance <= MATCH_DISTANCE
        ):

            matched_gt.add(best_gt)
            tp += 1

        else:

            fp += 1

    fn = len(gt_centroids) - len(matched_gt)

    return tp, fp, fn

def process_subject(model, subject, threshold):

    swi_file, mask_file = find_subject_file(subject)

    print()
    print("=" * 70)
    print(f"Processing {subject}")
    print("=" * 70)

    volume = np.load(swi_file)
    mask = np.load(mask_file)

    print("MRI shape:", volume.shape)

    gt_centroids = get_ground_truth_centroids(mask)

    print("Ground-truth lesions:", len(gt_centroids))

    windows = generate_windows(volume)

    print("Valid sliding windows:", len(windows))

    if len(windows) == 0:

        return {
            "subject": subject,
            "true_lesions": len(gt_centroids),
            "detections": 0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt_centroids),
        }

    feature_list = []
    centers = []

    for patch, center in windows:

        features = extract_features(patch)

        feature_list.append(features)
        centers.append(center)

    X = np.stack(feature_list).astype(np.float32)

    print("Feature matrix:", X.shape)

    probabilities = []

    for start in range(0, len(X), BATCH_SIZE):

        end = start + BATCH_SIZE

        batch = X[start:end]

        probs = model.predict_proba(batch)[:, 1]

        probabilities.extend(probs.tolist())

    probabilities = np.asarray(probabilities)

    detections = []

    for center, probability in zip(
        centers,
        probabilities
    ):

        if probability >= threshold:

            detections.append(
                {
                    "center": center,
                    "probability": float(probability),
                }
            )

    print(
        f"Raw detections above threshold "
        f"{threshold:.2f}: {len(detections)}"
    )

    detections = non_max_suppression(
        detections,
        NMS_DISTANCE
    )

    print(
        f"Detections after NMS: {len(detections)}"
    )

    tp, fp, fn = match_detections(
        detections,
        gt_centroids
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)

    print(
        f"Sensitivity: {sensitivity * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    return {
        "subject": subject,
        "true_lesions": len(gt_centroids),
        "detections": len(detections),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "precision": precision,
    }

def run_split(split_name, threshold):

    print()
    print("=" * 80)
    print("XGBOOST WHOLE-MRI LOCALIZATION")
    print("=" * 80)

    print("Split:", split_name)
    print("Model:", MODEL_PATH)
    print("Threshold:", threshold)
    print("Patch size:", PATCH_SIZE)
    print("Stride:", STRIDE)
    print("NMS distance:", NMS_DISTANCE)
    print("Match distance:", MATCH_DISTANCE)

    print()
    print("Loading XGBoost model...")

    model = xgb.XGBClassifier()

    model.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    subjects = load_subjects(split_name)

    print()
    print("Subjects:", len(subjects))

    results = []

    for subject in subjects:

        result = process_subject(
            model,
            subject,
            threshold
        )

        results.append(result)

    df = pd.DataFrame(results)

    output_csv = (
        OUTPUT_DIR /
        f"xgboost_whole_mri_{split_name}.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    total_tp = int(df["tp"].sum())
    total_fp = int(df["fp"].sum())
    total_fn = int(df["fn"].sum())

    total_true = int(
        df["true_lesions"].sum()
    )

    sensitivity = (
        total_tp / total_true
        if total_true > 0
        else 0.0
    )

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    fp_per_scan = (
        total_fp / len(df)
        if len(df) > 0
        else 0.0
    )

    summary = {
        "algorithm": "XGBoost",
        "split": split_name,
        "threshold": threshold,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "nms_distance": NMS_DISTANCE,
        "match_distance": MATCH_DISTANCE,
        "subjects": len(df),
        "true_lesions": total_true,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "lesion_sensitivity": sensitivity,
        "detection_precision": precision,
        "false_positives_per_scan": fp_per_scan,
    }

    summary_file = (
        REPORT_DIR /
        f"xgboost_whole_mri_{split_name}.json"
    )

    with open(summary_file, "w") as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print(f"XGBOOST WHOLE-MRI {split_name.upper()} RESULTS")
    print("=" * 80)

    print("Subjects:", len(df))
    print("True lesions:", total_true)
    print("TP:", total_tp)
    print("FP:", total_fp)
    print("FN:", total_fn)

    print(
        f"Lesion sensitivity: "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Detection precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"False positives / scan: "
        f"{fp_per_scan:.2f}"
    )

    print()
    print("Saved:")
    print(output_csv)
    print(summary_file)

    return summary

if __name__ == "__main__":

    run_split(
        "val",
        DEFAULT_THRESHOLD
    )