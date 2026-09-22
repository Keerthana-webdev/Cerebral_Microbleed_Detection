"""
Validation-only whole-MRI threshold sweep for 3D ResNet.

Purpose:
1. Run ResNet on VALIDATION subjects only.
2. Cache sliding-window probabilities.
3. Sweep thresholds.
4. Use validation results to choose an operating threshold.

IMPORTANT:
The TEST subjects are not used for threshold selection.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from src.models.resnet3d_model import ResNet3D


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

MODEL_PATH = PROJECT_ROOT / "models" / "resnet3d" / "resnet3d_best.pt"

CACHE_DIR = PROJECT_ROOT / "outputs" / "resnet3d" / "validation_cache"
RESULTS_FILE = PROJECT_ROOT / "reports" / "resnet3d_validation_threshold_sweep.csv"

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

NMS_DISTANCE = 15.0
MATCH_DISTANCE = 6.0

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================

def load_model():

    model = ResNet3D(
        in_channels=1,
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# GRID GENERATION
# ============================================================

def generate_grid_centers(volume_shape):

    half = tuple(x // 2 for x in PATCH_SIZE)

    centers = []

    for x in range(
        half[0],
        volume_shape[0] - half[0],
        STRIDE[0]
    ):

        for y in range(
            half[1],
            volume_shape[1] - half[1],
            STRIDE[1]
        ):

            for z in range(
                half[2],
                volume_shape[2] - half[2],
                STRIDE[2]
            ):

                centers.append((x, y, z))

    return centers


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patch(volume, center):

    half = tuple(x // 2 for x in PATCH_SIZE)

    x, y, z = center

    patch = volume[
        x - half[0]: x + half[0],
        y - half[1]: y + half[1],
        z - half[2]: z + half[2]
    ]

    return patch


# ============================================================
# BACKGROUND FILTER
# ============================================================

def is_valid_patch(patch):

    patch_std = float(patch.std())
    patch_mean = float(patch.mean())

    return (
        patch_std > 0.3
        and patch_mean > -1.5
    )

def nms_detections(detections):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["probability"],
        reverse=True
    )

    selected = []

    for det in detections:

        keep = True

        for chosen in selected:

            distance = np.linalg.norm(
                np.array(det["center"])
                - np.array(chosen["center"])
            )

            if distance < NMS_DISTANCE:
                keep = False
                break

        if keep:
            selected.append(det)

    return selected

def get_ground_truth(mask):

    from scipy import ndimage

    binary = mask > 0

    labeled, num = ndimage.label(binary)

    centers = []

    for i in range(1, num + 1):

        coords = np.argwhere(labeled == i)

        if len(coords) == 0:
            continue

        centroid = coords.mean(axis=0)

        centers.append(
            tuple(centroid.tolist())
        )

    return centers

def match_detections(detections, gt_centers):

    matched_gt = set()

    tp = 0
    fp = 0

    for det in detections:

        det_center = np.array(det["center"])

        best_distance = float("inf")
        best_index = None

        for i, gt in enumerate(gt_centers):

            if i in matched_gt:
                continue

            distance = np.linalg.norm(
                det_center - np.array(gt)
            )

            if distance < best_distance:
                best_distance = distance
                best_index = i

        if (
            best_index is not None
            and best_distance <= MATCH_DISTANCE
        ):

            tp += 1
            matched_gt.add(best_index)

        else:

            fp += 1

    fn = len(gt_centers) - len(matched_gt)

    return tp, fp, fn

def cache_subject(model, subject):

    swi_path = (
        PREPROCESSED_DIR
        / subject
        / "swi.npy"
    )

    mask_path = (
        PREPROCESSED_DIR
        / subject
        / "mask.npy"
    )

    volume = np.load(swi_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)

    centers = generate_grid_centers(
        volume.shape
    )

    valid_centers = []

    patches = []

    for center in centers:

        patch = extract_patch(
            volume,
            center
        )

        if not is_valid_patch(patch):
            continue

        valid_centers.append(center)

        patches.append(patch)

    print(
        f"  Grid positions: {len(centers)}"
    )

    print(
        f"  After background filtering: "
        f"{len(patches)}"
    )

    probabilities = []

    batch_size = 64

    with torch.no_grad():

        for start in range(
            0,
            len(patches),
            batch_size
        ):

            batch = np.stack(
                patches[start:start + batch_size]
            )

            batch = torch.from_numpy(
                batch
            ).unsqueeze(1).to(DEVICE)

            logits = model(batch)

            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    cache_file = CACHE_DIR / f"{subject}.npz"

    np.savez_compressed(
        cache_file,
        centers=np.array(
            valid_centers,
            dtype=np.int32
        ),
        probabilities=np.array(
            probabilities,
            dtype=np.float32
        ),
        mask=mask
    )

    print(
        f"  Cached {len(probabilities)} predictions."
    )

def evaluate_threshold(subjects, threshold):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for subject in subjects:

        cache_file = CACHE_DIR / f"{subject}.npz"

        data = np.load(cache_file)

        centers = data["centers"]
        probabilities = data["probabilities"]
        mask = data["mask"]

        detections = []

        for center, probability in zip(
            centers,
            probabilities
        ):

            if probability >= threshold:

                detections.append(
                    {
                        "center": tuple(
                            center.tolist()
                        ),
                        "probability": float(
                            probability
                        )
                    }
                )

        detections = nms_detections(
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

    total_scans = len(subjects)

    sensitivity = (
        total_tp
        / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    fp_per_scan = (
        total_fp / total_scans
    )

    return {
        "threshold": threshold,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "sensitivity": sensitivity,
        "FP_per_scan": fp_per_scan,
        "precision": precision,
    }

def main():

    print("=" * 70)
    print("3D RESNET VALIDATION WHOLE-MRI THRESHOLD SWEEP")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        splits = json.load(f)

    subjects = splits["val"]

    print(
        f"Validation subjects: {len(subjects)}"
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()

    print(
        "Model loaded successfully."
    )

    print("\n" + "=" * 70)
    print("STEP 1 — CACHING VALIDATION PREDICTIONS")
    print("=" * 70)

    for i, subject in enumerate(
        subjects,
        start=1
    ):

        print(
            f"\n[{i}/{len(subjects)}] {subject}"
        )

        cache_file = (
            CACHE_DIR / f"{subject}.npz"
        )

        if cache_file.exists():

            print(
                "  Cache already exists. Skipping."
            )

            continue

        cache_subject(
            model,
            subject
        )

    print("\n" + "=" * 70)
    print("STEP 2 — VALIDATION THRESHOLD SWEEP")
    print("=" * 70)

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            subjects,
            threshold
        )

        results.append(result)

        print(
            f"Threshold {threshold:.2f} | "
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

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("VALIDATION SWEEP COMPLETE")
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