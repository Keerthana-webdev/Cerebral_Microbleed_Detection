import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage


# ================================================================
# PROJECT PATH
# ================================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet3d"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# SETTINGS
# ================================================================

SPLIT = "test"

PATCH_SIZE = (16, 16, 8)

STRIDE = (8, 8, 4)

# Segmentation probability threshold
THRESHOLD = 0.50

# Minimum distance between detected candidates
NMS_DISTANCE = 15.0

# Maximum distance for matching detection to ground truth
MATCH_DISTANCE = 6.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# SUBJECT LIST
# ================================================================

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "splits.json"
)

import json

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
print("3D U-NET WHOLE-MRI LOCALIZATION")
print("=" * 80)

print(f"Split: {SPLIT}")
print(f"Model: {MODEL_PATH}")
print(f"Threshold: {THRESHOLD}")
print(f"Patch size: {PATCH_SIZE}")
print(f"Stride: {STRIDE}")
print(f"NMS distance: {NMS_DISTANCE}")
print(f"Match distance: {MATCH_DISTANCE}")
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

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("Model loaded successfully.")
print()

print(
    f"Subjects: {len(SUBJECTS)}"
)


# ================================================================
# NMS
# ================================================================

def apply_nms(candidates, min_distance):

    if len(candidates) == 0:
        return []

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []

    for candidate in candidates:

        keep = True

        cz = np.array(
            candidate["center"],
            dtype=np.float32
        )

        for selected_candidate in selected:

            sz = np.array(
                selected_candidate["center"],
                dtype=np.float32
            )

            distance = np.linalg.norm(
                cz - sz
            )

            if distance < min_distance:

                keep = False
                break

        if keep:
            selected.append(candidate)

    return selected


# ================================================================
# GROUND-TRUTH CENTROIDS
# ================================================================

def get_ground_truth_centroids(mask):

    binary_mask = (
        mask > 0
    ).astype(np.uint8)

    labeled, num_features = (
        ndimage.label(binary_mask)
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
            [
                sl.start
                for sl in location
            ]
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
# MATCH DETECTIONS TO GROUND TRUTH
# ================================================================

def match_detections(
    detections,
    ground_truth,
    max_distance
):

    matched_gt = set()

    true_positive = 0

    false_positive = 0

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
            and best_distance <= max_distance
        ):

            true_positive += 1
            matched_gt.add(best_gt)

        else:

            false_positive += 1

    false_negative = (
        len(ground_truth)
        - len(matched_gt)
    )

    return (
        true_positive,
        false_positive,
        false_negative
    )


# ================================================================
# WHOLE-MRI INFERENCE
# ================================================================

def process_subject(subject):

    print()
    print("=" * 70)
    print(f"Processing {subject}")
    print("=" * 70)

    subject_dir = (
        PREPROCESSED_DIR
        / subject
    )

    swi_path = (
        subject_dir
        / "swi.npy"
    )

    mask_path = (
        subject_dir
        / "mask.npy"
    )

    swi = np.load(
        swi_path
    ).astype(np.float32)

    mask = np.load(
        mask_path
    )

    print(
        f"MRI shape: {swi.shape}"
    )

    ground_truth = get_ground_truth_centroids(
        mask
    )

    print(
        f"Ground-truth lesions: "
        f"{len(ground_truth)}"
    )

    pz, py, px = PATCH_SIZE

    sz, sy, sx = STRIDE

    candidates = []

    valid_windows = 0

    # ------------------------------------------------------------
    # Sliding-window inference
    # ------------------------------------------------------------

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

                    # ------------------------------------------------
                    # Background rejection
                    # ------------------------------------------------

                    if np.std(patch) <= 0.3:

                        continue

                    if np.mean(patch) <= -1.5:

                        continue

                    valid_windows += 1

                    patch_tensor = (
                        torch.from_numpy(
                            patch
                        )
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

                    if (
                        max_probability
                        < THRESHOLD
                    ):

                        continue

                    # ------------------------------------------------
                    # Threshold segmentation mask
                    # ------------------------------------------------

                    binary_prediction = (
                        probabilities
                        >= THRESHOLD
                    ).astype(np.uint8)

                    labeled, num_features = (
                        ndimage.label(
                            binary_prediction
                        )
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
                            local_coords.mean(
                                axis=0
                            )
                        )

                        starts = np.array(
                            [
                                sl.start
                                for sl in location
                            ]
                        )

                        center = (
                            local_center
                            + starts
                        )

                        global_center = (
                            center
                            + np.array(
                                [
                                    z,
                                    y,
                                    x
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

                        candidates.append(
                            {
                                "center":
                                    global_center.tolist(),

                                "score":
                                    score,

                                "voxels":
                                    voxel_count
                            }
                        )

    print(
        f"Valid sliding windows: "
        f"{valid_windows}"
    )

    print(
        f"Raw detections above "
        f"threshold {THRESHOLD:.2f}: "
        f"{len(candidates)}"
    )

    # ------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------

    detections = apply_nms(
        candidates,
        NMS_DISTANCE
    )

    print(
        f"Detections after NMS: "
        f"{len(detections)}"
    )

    # ------------------------------------------------------------
    # Match
    # ------------------------------------------------------------

    tp, fp, fn = match_detections(
        detections,
        ground_truth,
        MATCH_DISTANCE
    )

    sensitivity = (
        tp / len(ground_truth)
        if len(ground_truth) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    print(
        f"TP: {tp}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"Sensitivity: "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    return {
        "subject": subject,
        "true_lesions":
            len(ground_truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity":
            sensitivity,
        "precision":
            precision,
        "valid_windows":
            valid_windows,
        "raw_detections":
            len(candidates),
        "nms_detections":
            len(detections)
    }

results = []

for subject in SUBJECTS:

    try:

        result = process_subject(
            subject
        )

        results.append(
            result
        )

    except Exception as e:

        print()
        print(
            f"[ERROR] {subject}: {e}"
        )

        results.append(
            {
                "subject": subject,
                "error": str(e)
            }
        )

valid_results = [
    r
    for r in results
    if "error" not in r
]

total_true = sum(
    r["true_lesions"]
    for r in valid_results
)

total_tp = sum(
    r["tp"]
    for r in valid_results
)

total_fp = sum(
    r["fp"]
    for r in valid_results
)

total_fn = sum(
    r["fn"]
    for r in valid_results
)

num_scans = len(
    valid_results
)

overall_sensitivity = (
    total_tp / total_true
    if total_true > 0
    else 0.0
)

overall_precision = (
    total_tp / (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0.0
)

fp_per_scan = (
    total_fp / num_scans
    if num_scans > 0
    else 0.0
)

results_df = pd.DataFrame(
    results
)

csv_path = (
    OUTPUT_DIR
    / "unet3d_whole_mri_test.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

report = {
    "algorithm":
        "3D U-Net",

    "split":
        SPLIT,

    "threshold":
        THRESHOLD,

    "patch_size":
        PATCH_SIZE,

    "stride":
        STRIDE,

    "nms_distance":
        NMS_DISTANCE,

    "match_distance":
        MATCH_DISTANCE,

    "subjects":
        num_scans,

    "true_lesions":
        total_true,

    "tp":
        total_tp,

    "fp":
        total_fp,

    "fn":
        total_fn,

    "lesion_sensitivity":
        overall_sensitivity,

    "detection_precision":
        overall_precision,

    "false_positives_per_scan":
        fp_per_scan
}

report_path = (
    REPORT_DIR
    / "unet3d_whole_mri_test.json"
)

with open(
    report_path,
    "w"
) as f:

    import json

    json.dump(
        report,
        f,
        indent=4
    )

print()
print("=" * 80)
print("3D U-NET WHOLE-MRI TEST RESULTS")
print("=" * 80)

print(
    f"Subjects: {num_scans}"
)

print(
    f"True lesions: {total_true}"
)

print(
    f"TP: {total_tp}"
)

print(
    f"FP: {total_fp}"
)

print(
    f"FN: {total_fn}"
)

print(
    f"Lesion sensitivity: "
    f"{overall_sensitivity * 100:.2f}%"
)

print(
    f"Detection precision: "
    f"{overall_precision * 100:.2f}%"
)

print(
    f"False positives / scan: "
    f"{fp_per_scan:.2f}"
)

print()
print("Saved:")
print(csv_path)
print(report_path)

print("=" * 80)