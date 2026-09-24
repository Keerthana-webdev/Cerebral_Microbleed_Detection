import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet3d_model import UNet3D


# ================================================================
# PATHS
# ================================================================

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "unet3d"
    / "unet3d_real_best.pt"
)

SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# SETTINGS
# ================================================================

SPLIT = "val"

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
    0.95
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# LOAD SUBJECTS
# ================================================================

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)


def get_subjects(split_name):

    value = splits[split_name]

    if isinstance(value, dict):
        value = value.get("subjects", [])

    return list(value)


SUBJECTS = get_subjects(SPLIT)


# ================================================================
# LOAD MODEL
# ================================================================

print("=" * 80)
print("3D U-NET VALIDATION THRESHOLD SWEEP")
print("=" * 80)

print(f"Split: {SPLIT}")
print(f"Model: {MODEL_PATH}")
print(f"Patch size: {PATCH_SIZE}")
print(f"Stride: {STRIDE}")
print(f"NMS distance: {NMS_DISTANCE}")
print(f"Match distance: {MATCH_DISTANCE}")
print(f"Thresholds: {THRESHOLDS}")
print()

print("Loading U-Net model...")

model = UNet3D(
    in_channels=1,
    out_channels=1
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")
print(f"Validation subjects: {len(SUBJECTS)}")


# ================================================================
# GROUND TRUTH
# ================================================================

def get_ground_truth_centroids(mask):

    binary_mask = (
        mask > 0
    ).astype(np.uint8)

    labeled, num_features = ndimage.label(
        binary_mask
    )

    objects = ndimage.find_objects(
        labeled
    )

    centroids = []

    for label_id, location in enumerate(
        objects,
        start=1
    ):

        if location is None:
            continue

        coords = np.argwhere(
            labeled[location] == label_id
        )

        if len(coords) == 0:
            continue

        starts = np.array(
            [sl.start for sl in location]
        )

        center = (
            coords.mean(axis=0)
            + starts
        )

        centroids.append(
            center.tolist()
        )

    return centroids


# ================================================================
# NMS
# ================================================================

def apply_nms(candidates, min_distance):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    for candidate in candidates:

        center = np.array(
            candidate["center"],
            dtype=np.float32
        )

        keep = True

        for existing in selected:

            existing_center = np.array(
                existing["center"],
                dtype=np.float32
            )

            distance = np.linalg.norm(
                center - existing_center
            )

            if distance < min_distance:

                keep = False
                break

        if keep:
            selected.append(candidate)

    return selected


# ================================================================
# MATCHING
# ================================================================

def match_detections(
    detections,
    ground_truth
):

    matched_gt = set()

    tp = 0
    fp = 0

    for detection in detections:

        detection_center = np.array(
            detection["center"],
            dtype=np.float32
        )

        best_gt = None
        best_distance = float("inf")

        for gt_index, gt_center in enumerate(
            ground_truth
        ):

            if gt_index in matched_gt:
                continue

            gt_center = np.array(
                gt_center,
                dtype=np.float32
            )

            distance = np.linalg.norm(
                detection_center
                - gt_center
            )

            if distance < best_distance:

                best_distance = distance
                best_gt = gt_index

        if (
            best_gt is not None
            and best_distance <= MATCH_DISTANCE
        ):

            tp += 1
            matched_gt.add(best_gt)

        else:

            fp += 1

    fn = (
        len(ground_truth)
        - len(matched_gt)
    )

    return tp, fp, fn

def run_subject(subject):

    subject_dir = (
        PREPROCESSED_DIR
        / subject
    )

    swi = np.load(
        subject_dir / "swi.npy"
    ).astype(np.float32)

    mask = np.load(
        subject_dir / "mask.npy"
    )

    ground_truth = get_ground_truth_centroids(
        mask
    )

    pz, py, px = PATCH_SIZE
    sz, sy, sx = STRIDE

    candidates = []

    with torch.no_grad():

        for z in range(
            0,
            swi.shape[0] - pz + 1,
            sz
        ):

            for y in range(
                0,
                swi.shape[1] - py + 1,
                sy
            ):

                for x in range(
                    0,
                    swi.shape[2] - px + 1,
                    sx
                ):

                    patch = swi[
                        z:z + pz,
                        y:y + py,
                        x:x + px
                    ]

                    # Background rejection
                    if np.std(patch) <= 0.3:
                        continue

                    if np.mean(patch) <= -1.5:
                        continue

                    patch_tensor = (
                        torch.from_numpy(patch)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    logits = model(
                        patch_tensor
                    )

                    probabilities = (
                        torch.sigmoid(logits)
                        .cpu()
                        .numpy()[0, 0]
                    )

                    max_probability = float(
                        probabilities.max()
                    )

                    candidates.append(
                        {
                            "z": z,
                            "y": y,
                            "x": x,
                            "probabilities":
                                probabilities,
                            "max_probability":
                                max_probability
                        }
                    )

    subject_results = []

    for threshold in THRESHOLDS:

        detections = []

        for candidate in candidates:

            probabilities = candidate[
                "probabilities"
            ]

            if (
                candidate["max_probability"]
                < threshold
            ):
                continue

            binary_prediction = (
                probabilities >= threshold
            ).astype(np.uint8)

            labeled, _ = ndimage.label(
                binary_prediction
            )

            objects = ndimage.find_objects(
                labeled
            )

            for label_id, location in enumerate(
                objects,
                start=1
            ):

                if location is None:
                    continue

                component = (
                    labeled[location]
                    == label_id
                )

                voxel_count = int(
                    component.sum()
                )

                if voxel_count == 0:
                    continue

                local_coords = np.argwhere(
                    component
                )

                local_center = (
                    local_coords.mean(axis=0)
                )

                starts = np.array(
                    [sl.start for sl in location]
                )

                center = (
                    local_center
                    + starts
                    + np.array(
                        [
                            candidate["z"],
                            candidate["y"],
                            candidate["x"]
                        ]
                    )
                )

                component_scores = (
                    probabilities[location][
                        component
                    ]
                )

                score = float(
                    component_scores.max()
                )

                detections.append(
                    {
                        "center":
                            center.tolist(),

                        "score":
                            score,

                        "voxels":
                            voxel_count
                    }
                )

        detections = apply_nms(
            detections,
            NMS_DISTANCE
        )

        tp, fp, fn = match_detections(
            detections,
            ground_truth
        )

        subject_results.append(
            {
                "threshold": threshold,
                "true_lesions":
                    len(ground_truth),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detections":
                    len(detections)
            }
        )

    return subject_results

all_results = []

for index, subject in enumerate(
    SUBJECTS,
    start=1
):

    print()
    print(
        f"[{index}/{len(SUBJECTS)}] "
        f"Processing {subject}"
    )

    results = run_subject(subject)

    for result in results:

        result["subject"] = subject

        all_results.append(result)

df = pd.DataFrame(all_results)

summary = []

for threshold in THRESHOLDS:

    subset = df[
        df["threshold"] == threshold
    ]

    total_true = int(
        subset["true_lesions"].sum()
    )

    total_tp = int(
        subset["tp"].sum()
    )

    total_fp = int(
        subset["fp"].sum()
    )

    total_fn = int(
        subset["fn"].sum()
    )

    sensitivity = (
        total_tp / total_true
        if total_true > 0
        else 0.0
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    fp_per_scan = (
        total_fp / len(SUBJECTS)
    )

    summary.append(
        {
            "threshold": threshold,
            "true_lesions": total_true,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "sensitivity": sensitivity,
            "precision": precision,
            "fp_per_scan": fp_per_scan
        }
    )


summary_df = pd.DataFrame(
    summary
)

csv_path = (
    REPORT_DIR
    / "unet3d_whole_mri_validation_threshold_sweep.csv"
)

json_path = (
    REPORT_DIR
    / "unet3d_whole_mri_validation_threshold_sweep.json"
)

summary_df.to_csv(
    csv_path,
    index=False
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print()
print("=" * 80)
print("U-NET VALIDATION THRESHOLD SWEEP RESULTS")
print("=" * 80)

print(
    f"{'Threshold':<12}"
    f"{'TP':<8}"
    f"{'FP':<10}"
    f"{'FN':<8}"
    f"{'Sensitivity':<15}"
    f"{'Precision':<15}"
    f"{'FP/Scan':<10}"
)

print("-" * 80)

for row in summary:

    print(
        f"{row['threshold']:<12.2f}"
        f"{row['TP']:<8}"
        f"{row['FP']:<10}"
        f"{row['FN']:<8}"
        f"{row['sensitivity'] * 100:<15.2f}"
        f"{row['precision'] * 100:<15.2f}"
        f"{row['fp_per_scan']:<10.2f}"
    )

print()
print("Saved:")
print(csv_path)
print(json_path)

print("=" * 80)