"""
Whole-MRI localization using the trained 3D ResNet.

Pipeline:
Whole MRI
    -> sliding 3D patches
    -> 3D ResNet
    -> CMB probability
    -> threshold
    -> NMS
    -> ground-truth matching

This is a single-stage ResNet experiment and is kept separate
from the original two-stage CNN localization pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src" / "preprocessing")
)

sys.path.append(
    str(PROJECT_ROOT / "src" / "models")
)

from config import PROCESSED_DATA_DIR, SPLITS_DIR
from resnet3d_model import ResNet3D


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

DEVICE = torch.device("cpu")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet3d"
    / "resnet3d_best.pt"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PATCH_SIZE = (16, 16, 8)

# Same sliding-window spacing used by the existing
# whole-scan localization experiment.
STRIDE = (8, 8, 4)

# Initial baseline threshold.
# We will NOT call this the final optimized threshold yet.
RESNET_THRESHOLD = 0.50

NMS_MIN_DISTANCE = 15

MATCH_DISTANCE = 6


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 70)
print("3D RESNET WHOLE-MRI LOCALIZATION")
print("=" * 70)

print("Device:", DEVICE)
print("Model:", MODEL_PATH)

model = ResNet3D(num_classes=2).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("ResNet model loaded successfully.")


# ------------------------------------------------------------
# GRID GENERATION
# ------------------------------------------------------------

def generate_grid_centers(
    volume_shape,
    patch_size,
    stride
):
    """
    Generate sliding-window centers across the MRI volume.
    """

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


# ------------------------------------------------------------
# PATCH EXTRACTION
# ------------------------------------------------------------

def cut_patch(
    volume,
    center,
    size
):
    """
    Extract a 3D patch around a center.
    """

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


# ------------------------------------------------------------
# SLIDING-WINDOW RESNET
# ------------------------------------------------------------

def run_sliding_window(
    volume,
    batch_size=64
):

    all_centers = generate_grid_centers(
        volume.shape,
        PATCH_SIZE,
        STRIDE
    )

    # --------------------------------------------------------
    # Background filtering
    # --------------------------------------------------------

    candidate_centers = []

    for center in all_centers:

        patch = cut_patch(
            volume,
            center,
            PATCH_SIZE
        )

        # Same basic background filtering principle
        # used in the existing localization pipeline.
        if (
            patch.std() > 0.3
            and patch.mean() > -1.5
        ):

            candidate_centers.append(
                center
            )

    print(
        f"  Grid positions: {len(all_centers)}"
        f" total, {len(candidate_centers)}"
        f" after background filtering"
    )

    # --------------------------------------------------------
    # ResNet prediction
    # --------------------------------------------------------

    detections = []

    for i in range(
        0,
        len(candidate_centers),
        batch_size
    ):

        batch_centers = candidate_centers[
            i:i + batch_size
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

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1].cpu().numpy()

        for center, probability in zip(
            batch_centers,
            probabilities
        ):

            if probability >= RESNET_THRESHOLD:

                detections.append(
                    (
                        center,
                        float(probability)
                    )
                )

    print(
        f"  ResNet detections before NMS: "
        f"{len(detections)}"
    )

    return detections


# ------------------------------------------------------------
# NMS
# ------------------------------------------------------------

def apply_nms(
    detections,
    min_distance
):

    detections = sorted(
        detections,
        key=lambda item: item[1],
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


# ------------------------------------------------------------
# GROUND-TRUTH CENTERS
# ------------------------------------------------------------

def get_ground_truth_centers(
    mask
):

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

        if len(coords) >= 1:

            centers.append(
                tuple(
                    coords.mean(
                        axis=0
                    )
                )
            )

    return centers

def match_detections_to_ground_truth(
    predicted_centers,
    true_centers,
    max_distance
):

    matched_true = set()

    true_positives = 0

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

            if distance <= max_distance:

                true_positives += 1

                matched_true.add(i)

                break

    false_positives = (
        len(predicted_centers)
        -
        true_positives
    )

    false_negatives = (
        len(true_centers)
        -
        len(matched_true)
    )

    return (
        true_positives,
        false_positives,
        false_negatives
    )

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

    print(
        f"Threshold: {RESNET_THRESHOLD}"
    )

    print(
        f"NMS distance: {NMS_MIN_DISTANCE}"
    )

    print(
        f"Match distance: {MATCH_DISTANCE}"
    )

    all_results = []

    for subject_name in test_subjects:

        print()
        print(
            "-" * 60
        )

        print(
            f"Processing {subject_name}..."
        )

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

        print(
            "MRI shape:",
            volume.shape
        )

        raw_detections = run_sliding_window(
            volume
        )

        final_detections = apply_nms(
            raw_detections,
            NMS_MIN_DISTANCE
        )

        print(
            f"  After NMS: "
            f"{len(final_detections)} detections"
        )

        true_centers = (
            get_ground_truth_centers(
                mask
            )
        )

        predicted_centers = [
            center
            for center, probability
            in final_detections
        ]

        tp, fp, fn = (
            match_detections_to_ground_truth(
                predicted_centers,
                true_centers,
                MATCH_DISTANCE
            )
        )

        print(
            f"  True lesions: "
            f"{len(true_centers)}"
        )

        print(
            f"  TP: {tp} | "
            f"FP: {fp} | "
            f"FN: {fn}"
        )

        all_results.append(
            {
                "subject": subject_name,
                "true_lesions": len(
                    true_centers
                ),
                "detections": len(
                    final_detections
                ),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            }
        )

    results_df = pd.DataFrame(
        all_results
    )

    print()
    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    total_tp = int(
        results_df[
            "true_positives"
        ].sum()
    )

    total_fp = int(
        results_df[
            "false_positives"
        ].sum()
    )

    total_fn = int(
        results_df[
            "false_negatives"
        ].sum()
    )

    total_true = (
        total_tp
        +
        total_fn
    )

    lesion_sensitivity = (
        total_tp / total_true
        if total_true > 0
        else 0
    )

    fp_per_scan = (
        total_fp
        /
        len(test_subjects)
    )

    precision = (
        total_tp
        /
        (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    print()
    print(
        "=" * 70
    )

    print(
        "3D RESNET WHOLE-SCAN RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Test subjects        : "
        f"{len(test_subjects)}"
    )

    print(
        f"Total true lesions   : "
        f"{total_true}"
    )

    print(
        f"True positives       : "
        f"{total_tp}"
    )

    print(
        f"False positives      : "
        f"{total_fp}"
    )

    print(
        f"False negatives      : "
        f"{total_fn}"
    )

    print(
        f"Lesion sensitivity   : "
        f"{lesion_sensitivity:.4f}"
    )

    print(
        f"False positives/scan : "
        f"{fp_per_scan:.2f}"
    )

    print(
        f"Detection precision  : "
        f"{precision:.4f}"
    )

    output_file = (
        REPORTS_DIR
        /
        "resnet3d_localization_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
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