"""
Sweep whole-MRI thresholds for the trained 3D ResNet.

This first creates/caches sliding-window probabilities for the
11 test subjects, then evaluates several thresholds without
rerunning the neural network.

IMPORTANT:
These test-set results are for diagnostic comparison.
Threshold selection should ultimately be done on validation
subjects before reporting a final test result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "preprocessing")
)

sys.path.append(
    str(Path(__file__).resolve().parents[1] / "models")
)

from config import PROCESSED_DATA_DIR, SPLITS_DIR
from resnet3d_model import ResNet3D


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet3d"
    / "resnet3d_best.pt"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "resnet3d"
    / "localization_cache"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

NMS_MIN_DISTANCE = 15
MATCH_DISTANCE = 6

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("3D RESNET WHOLE-MRI THRESHOLD SWEEP")
print("=" * 70)

print("Device:", DEVICE)

model = ResNet3D(
    num_classes=2
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GRID
# ============================================================

def generate_grid_centers(
    volume_shape,
    patch_size,
    stride
):

    half = [
        s // 2
        for s in patch_size
    ]

    centers = []

    for x in range(
        half[0],
        volume_shape[0] - half[0],
        stride[0]
    ):

        for y in range(
            half[1],
            volume_shape[1] - half[1],
            stride[1]
        ):

            for z in range(
        half[2],
        volume_shape[2] - half[2],
        stride[2]
    ):

                centers.append(
                    (x, y, z)
                )

    return centers


# ============================================================
# PATCH
# ============================================================

def cut_patch(
    volume,
    center,
    size
):

    half = [
        s // 2
        for s in size
    ]

    slices = tuple(
        slice(
            c - h,
            c + h
        )
        for c, h in zip(
            center,
            half
        )
    )

    return volume[slices]


# ============================================================
# BACKGROUND FILTER
# ============================================================

def get_candidate_centers(volume):

    all_centers = generate_grid_centers(
        volume.shape,
        PATCH_SIZE,
        STRIDE
    )

    candidate_centers = []

    for center in all_centers:

        patch = cut_patch(
            volume,
            center,
            PATCH_SIZE
        )

        if (
            patch.std() > 0.3
            and patch.mean() > -1.5
        ):

            candidate_centers.append(
                center
            )

    print(
        f"  Grid positions: {len(all_centers)}"
    )

    print(
        f"  After background filtering: "
        f"{len(candidate_centers)}"
    )

    return candidate_centers


# ============================================================
# CACHE MODEL PREDICTIONS
# ============================================================

def cache_subject(
    subject_name
):

    cache_file = (
        CACHE_DIR
        / f"{subject_name}.npz"
    )

    if cache_file.exists():

        print(
            f"  Cache already exists: "
            f"{cache_file.name}"
        )

        return


    subject_folder = (
        PROCESSED_DATA_DIR
        /
        subject_name
    )

    volume = np.load(
        subject_folder / "swi.npy"
    )

    mask = np.load(
        subject_folder / "mask.npy"
    )

    centers = get_candidate_centers(
        volume
    )

    probabilities = []

    batch_size = 64

    for start in range(
        0,
        len(centers),
        batch_size
    ):

        batch_centers = centers[
            start:start + batch_size
        ]

        patches = np.stack(
            [
                cut_patch(
                    volume,
                    center,
                    PATCH_SIZE
                )
                for center in batch_centers
            ]
        )

        patches_tensor = torch.tensor(
            patches,
            dtype=torch.float32
        ).unsqueeze(1)

        with torch.no_grad():

            logits = model(
                patches_tensor
            )

            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1].cpu().numpy()

        probabilities.extend(
            probs.tolist()
        )

    np.savez_compressed(
        cache_file,
        centers=np.array(
            centers,
            dtype=np.int16
        ),
        probabilities=np.array(
            probabilities,
            dtype=np.float32
        ),
        mask=mask
    )

    print(
        f"  Cached {len(centers)} predictions."
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def get_ground_truth_centers(mask):

    from scipy import ndimage

    labeled, num_blobs = ndimage.label(
        mask
    )

    centers = []

    for i in range(
        1,
        num_blobs + 1
    ):

        coords = np.argwhere(
            labeled == i
        )

        if len(coords) > 0:

            centers.append(
                tuple(
                    coords.mean(
                        axis=0
                    )
                )
            )

    return centers


# ============================================================
# NMS
# ============================================================

def apply_nms(
    detections,
    min_distance
):

    detections = sorted(
        detections,
        key=lambda x: x[1],
        reverse=True
    )

    kept = []

    for center, probability in detections:

        too_close = any(
            np.linalg.norm(
                np.array(center)
                -
                np.array(existing_center)
            )
            < min_distance

            for existing_center, _
            in kept
        )

        if not too_close:

            kept.append(
                (
                    center,
                    probability
                )
            )

    return kept


# ============================================================
# MATCH
# ============================================================

def match_detections(
    predicted_centers,
    true_centers
):

    matched_true = set()

    tp = 0

    for predicted_center in predicted_centers:

        for i, true_center in enumerate(
            true_centers
        ):

            if i in matched_true:
                continue

            distance = np.linalg.norm(
                np.array(predicted_center)
                -
                np.array(true_center)
            )

            if distance <= MATCH_DISTANCE:

                tp += 1

                matched_true.add(i)

                break

    fp = (
        len(predicted_centers)
        -
        tp
    )

    fn = (
        len(true_centers)
        -
        len(matched_true)
    )

    return tp, fp, fn


# ============================================================
# EVALUATE THRESHOLD
# ============================================================

def evaluate_threshold(
    subject_names,
    threshold
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for subject_name in subject_names:

        cache_file = (
            CACHE_DIR
            / f"{subject_name}.npz"
        )

        data = np.load(
            cache_file
        )

        centers = data["centers"]

        probabilities = (
            data["probabilities"]
        )

        mask = data["mask"]

        detections = []

        for center, probability in zip(
            centers,
            probabilities
        ):

            if probability >= threshold:

                detections.append(
                    (
                        tuple(
                            center.tolist()
                        ),
                        float(probability)
                    )
                )

        final_detections = apply_nms(
            detections,
            NMS_MIN_DISTANCE
        )

        true_centers = (
            get_ground_truth_centers(
                mask
            )
        )

        predicted_centers = [
            center
            for center, _
            in final_detections
        ]

        tp, fp, fn = match_detections(
            predicted_centers,
            true_centers
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_true = (
        total_tp
        +
        total_fn
    )

    sensitivity = (
        total_tp / total_true
        if total_true > 0
        else 0
    )

    fp_per_scan = (
        total_fp / len(subject_names)
    )

    precision = (
        total_tp
        /
        (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0
    )

    return {
        "threshold": threshold,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "sensitivity": sensitivity,
        "FP_per_scan": fp_per_scan,
        "precision": precision
    }


# ============================================================
# MAIN
# ============================================================

def main():

    with open(
        SPLITS_DIR / "splits.json"
    ) as f:

        splits = json.load(f)

    test_subjects = splits["test"]

    print()
    print(
        f"Test subjects: {len(test_subjects)}"
    )

    # --------------------------------------------------------
    # STEP 1: CACHE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 1 — CACHING RESNET PREDICTIONS")
    print("=" * 70)

    for i, subject in enumerate(
        test_subjects,
        start=1
    ):

        print()
        print(
            f"[{i}/{len(test_subjects)}] "
            f"{subject}"
        )

        cache_subject(
            subject
        )

    # --------------------------------------------------------
    # STEP 2: SWEEP
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2 — THRESHOLD SWEEP")
    print("=" * 70)

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            test_subjects,
            threshold
        )

        results.append(
            result
        )

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

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        REPORTS_DIR
        /
        "resnet3d_threshold_sweep.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP COMPLETE")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()