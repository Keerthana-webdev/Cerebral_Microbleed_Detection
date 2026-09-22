import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.densenet3d_model import DenseNet3D


# ============================================================
# PATHS
# ============================================================

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "densenet3d"
    / "densenet3d_best.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

NMS_DISTANCE = 15.0
MATCH_DISTANCE = 6.0

BATCH_SIZE = 128

THRESHOLDS = [
    0.50,
    0.60,
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

DEVICE = torch.device("cpu")


# ============================================================
# LOAD VALIDATION SUBJECTS
# ============================================================

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

subjects = splits["val"]


print("=" * 80)
print("DENSENET3D WHOLE-MRI VALIDATION THRESHOLD SWEEP")
print("=" * 80)

print("Validation subjects:", len(subjects))
print("Thresholds:", THRESHOLDS)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading DenseNet3D model...")

model = DenseNet3D(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# HELPER: BACKGROUND FILTER
# ============================================================

def is_valid_patch(patch):

    return (
        np.std(patch) > 0.3
        and np.mean(patch) > -1.5
    )


# ============================================================
# HELPER: GENERATE WINDOWS
# ============================================================

def generate_windows(volume):

    pz, py, px = PATCH_SIZE
    sz, sy, sx = STRIDE

    z_dim, y_dim, x_dim = volume.shape

    patches = []
    centers = []

    for z in range(
        0,
        z_dim - pz + 1,
        sz
    ):

        for y in range(
            0,
            y_dim - py + 1,
            sy
        ):

            for x in range(
                0,
                x_dim - px + 1,
                sx
            ):

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


# ============================================================
# HELPER: GROUND TRUTH CENTROIDS
# ============================================================

def get_gt_centroids(mask):

    labeled, count = ndimage.label(
        mask > 0
    )

    if count == 0:
        return []

    objects = ndimage.find_objects(
        labeled
    )

    centroids = []

    for label_id, slices in enumerate(
        objects,
        start=1
    ):

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

        centroid = (
            coords.mean(axis=0)
            + offsets
        )

        centroids.append(
            tuple(centroid)
        )

    return centroids


# ============================================================
# HELPER: NMS
# ============================================================

def non_max_suppression(
    detections
):

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
                candidate_center
                - selected_center
            )

            if distance < NMS_DISTANCE:

                keep = False
                break

        if keep:

            selected.append(
                candidate
            )

    return selected


# ============================================================
# HELPER: MATCH
# ============================================================

def match_detections(
    detections,
    gt_centroids
):

    if len(gt_centroids) == 0:

        return (
            0,
            len(detections),
            0
        )

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

        for gt_index, gt in enumerate(
            gt_centroids
        ):

            if gt_index in matched_gt:
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
                best_gt = gt_index

        if (
            best_gt is not None
            and best_distance <= MATCH_DISTANCE
        ):

            matched_gt.add(
                best_gt
            )

            tp += 1

        else:

            fp += 1

    fn = (
        len(gt_centroids)
        - len(matched_gt)
    )

    return tp, fp, fn

cache = {}

print()
print("=" * 80)
print("BUILDING VALIDATION PREDICTION CACHE")
print("=" * 80)

for subject in subjects:

    print()
    print("-" * 70)
    print("Preparing:", subject)

    subject_dir = (
        PREPROCESSED_DIR
        / subject
    )

    volume = np.load(
        subject_dir / "swi.npy"
    )

    mask = np.load(
        subject_dir / "mask.npy"
    )

    gt_centroids = get_gt_centroids(
        mask
    )

    patches, centers = generate_windows(
        volume
    )

    print(
        "MRI shape:",
        volume.shape
    )

    print(
        "Valid windows:",
        len(patches)
    )

    print(
        "Ground-truth lesions:",
        len(gt_centroids)
    )

    probabilities = []

    for start in range(
        0,
        len(patches),
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        batch_patches = patches[
            start:end
        ]

        batch = np.stack(
            batch_patches
        ).astype(
            np.float32
        )

        batch = torch.from_numpy(
            batch
        ).unsqueeze(1)

        batch = batch.to(
            DEVICE
        )

        with torch.no_grad():

            outputs = model(
                batch
            )

            probs = F.softmax(
                outputs,
                dim=1
            )[:, 1]

        probabilities.extend(
            probs.cpu().numpy().tolist()
        )

    cache[subject] = {
        "centers": centers,
        "probabilities": np.asarray(
            probabilities,
            dtype=np.float32
        ),
        "gt": gt_centroids,
    }

    print(
        "Prediction cache ready."
    )

results = []

print()
print("=" * 80)
print("RUNNING DENSENET3D THRESHOLD SWEEP")
print("=" * 80)

for threshold in THRESHOLDS:

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true = 0

    print()
    print(
        f"Threshold = {threshold:.2f}"
    )

    for subject in subjects:

        data = cache[subject]

        centers = data["centers"]

        probabilities = data[
            "probabilities"
        ]

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

        detections = (
            non_max_suppression(
                detections
            )
        )

        tp, fp, fn = (
            match_detections(
                detections,
                gt
            )
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true += len(gt)

    sensitivity = (
        total_tp / total_true
        if total_true > 0
        else 0.0
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
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

df = pd.DataFrame(
    results
)

output_file = (
    REPORT_DIR
    / "densenet3d_whole_mri_validation_threshold_sweep.csv"
)

df.to_csv(
    output_file,
    index=False
)

print()
print("=" * 80)
print("FINAL DENSENET3D THRESHOLD SWEEP")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(output_file)