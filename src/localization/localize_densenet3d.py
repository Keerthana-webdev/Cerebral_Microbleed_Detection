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

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "densenet3d"
    / "densenet3d_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "densenet3d"
)

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

# Initial baseline threshold.
# We will tune this on validation later.
THRESHOLD = 0.50
NMS_DISTANCE = 15.0
MATCH_DISTANCE = 6.0
BATCH_SIZE = 128
DEVICE = torch.device("cpu")

def load_subjects(split_name):

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    return splits[split_name]

def is_valid_patch(patch):

    patch_std = float(
        np.std(patch)
    )

    patch_mean = float(
        np.mean(patch)
    )

    return (
        patch_std > 0.3
        and patch_mean > -1.5
    )

def generate_windows(volume):

    z_dim, y_dim, x_dim = volume.shape

    pz, py, px = PATCH_SIZE

    sz, sy, sx = STRIDE

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

def get_ground_truth_centroids(mask):

    binary_mask = mask > 0

    labeled, num_objects = ndimage.label(
        binary_mask
    )

    if num_objects == 0:
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

def non_max_suppression(
    detections,
    min_distance
):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d["probability"],
        reverse=True
    )

    selected = []

    for candidate in detections:

        candidate_center = np.array(
            candidate["center"],
            dtype=float
        )

        keep = True

        for chosen in selected:

            chosen_center = np.array(
                chosen["center"],
                dtype=float
            )

            distance = np.linalg.norm(
                candidate_center
                - chosen_center
            )

            if distance < min_distance:

                keep = False
                break

        if keep:
            selected.append(
                candidate
            )

    return selected

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

def load_model():

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

    return model

def predict_patches(
    model,
    patches
):

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
        )

        batch = batch.unsqueeze(1)

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

    return np.asarray(
        probabilities,
        dtype=np.float32
    )

def process_subject(
    model,
    subject
):

    print()
    print("=" * 70)
    print(
        f"Processing {subject}"
    )
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

    volume = np.load(
        swi_path
    )

    mask = np.load(
        mask_path
    )

    print(
        "MRI shape:",
        volume.shape
    )

    gt_centroids = (
        get_ground_truth_centroids(
            mask
        )
    )

    print(
        "Ground-truth lesions:",
        len(gt_centroids)
    )

    patches, centers = (
        generate_windows(volume)
    )

    print(
        "Valid sliding windows:",
        len(patches)
    )

    if len(patches) == 0:

        return {
            "subject": subject,
            "true_lesions": len(
                gt_centroids
            ),
            "detections": 0,
            "tp": 0,
            "fp": 0,
            "fn": len(
                gt_centroids
            )
        }

    probabilities = (
        predict_patches(
            model,
            patches
        )
    )

    detections = []

    for center, probability in zip(
        centers,
        probabilities
    ):

        if probability >= THRESHOLD:

            detections.append(
                {
                    "center": center,
                    "probability": float(
                        probability
                    )
                }
            )

    print(
        "Raw detections above "
        f"threshold {THRESHOLD:.2f}:",
        len(detections)
    )

    detections = (
        non_max_suppression(
            detections,
            NMS_DISTANCE
        )
    )

    print(
        "Detections after NMS:",
        len(detections)
    )

    tp, fp, fn = (
        match_detections(
            detections,
            gt_centroids
        )
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
        f"Sensitivity: "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    return {
        "subject": subject,
        "true_lesions": len(
            gt_centroids
        ),
        "detections": len(
            detections
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "precision": precision
    }

def run_split(
    split_name
):

    print()
    print("=" * 80)
    print("DENSENET3D WHOLE-MRI LOCALIZATION")
    print("=" * 80)

    print(
        "Split:",
        split_name
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Threshold:",
        THRESHOLD
    )

    print(
        "Patch size:",
        PATCH_SIZE
    )

    print(
        "Stride:",
        STRIDE
    )

    print(
        "NMS distance:",
        NMS_DISTANCE
    )

    print(
        "Match distance:",
        MATCH_DISTANCE
    )

    model = load_model()

    subjects = load_subjects(
        split_name
    )

    print()
    print(
        "Subjects:",
        len(subjects)
    )

    results = []

    for subject in subjects:

        result = process_subject(
            model,
            subject
        )

        results.append(
            result
        )

    df = pd.DataFrame(
        results
    )

    output_csv = (
        OUTPUT_DIR
        / f"densenet3d_whole_mri_{split_name}.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    total_tp = int(
        df["tp"].sum()
    )

    total_fp = int(
        df["fp"].sum()
    )

    total_fn = int(
        df["fn"].sum()
    )

    total_true = int(
        df["true_lesions"].sum()
    )

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
        total_fp / len(df)
        if len(df) > 0
        else 0.0
    )

    summary = {
        "algorithm": "3D DenseNet",
        "split": split_name,
        "threshold": THRESHOLD,
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
        "false_positives_per_scan": fp_per_scan
    }

    summary_file = (
        REPORT_DIR
        / f"densenet3d_whole_mri_{split_name}.json"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print(
        f"DENSENET3D WHOLE-MRI "
        f"{split_name.upper()} RESULTS"
    )
    print("=" * 80)

    print(
        "Subjects:",
        len(df)
    )

    print(
        "True lesions:",
        total_true
    )

    print(
        "TP:",
        total_tp
    )

    print(
        "FP:",
        total_fp
    )

    print(
        "FN:",
        total_fn
    )

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

if __name__ == "__main__":

    # IMPORTANT:
    # Start with validation.
    # Do NOT run the test set yet.

    run_split("val")