"""
Lesson 14: Run the FULL two-stage pipeline on whole, uncut brain scans
(not pre-cut patches) and output real (x, y, z) coordinates. This is the
true "localization" your project objectives promise.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)             # how far the window moves each step (half the patch size = 50% overlap)
STAGE1_THRESHOLD = 0.85        # was 0.5 — require much higher confidence to pass Stage 1
STAGE2_THRESHOLD = 0.85        # was 0.5 — same for Stage 2
NMS_MIN_DISTANCE = 15          # was 8 — merge detections within a wider radius
MATCH_DISTANCE = 6             # voxels — how close a prediction must be to count as a true match

stage1_model = SimpleCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
stage1_model.eval()

stage2_model = SimpleCNN3D().to(DEVICE)
stage2_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
stage2_model.eval()


def generate_grid_centers(volume_shape, patch_size, stride):
    """All candidate window centers across the volume, on a grid."""
    half = [s // 2 for s in patch_size]
    centers = []
    for x in range(half[0], volume_shape[0] - half[0], stride[0]):
        for y in range(half[1], volume_shape[1] - half[1], stride[1]):
            for z in range(half[2], volume_shape[2] - half[2], stride[2]):
                centers.append((x, y, z))
    return centers


def cut_patch(volume, center, size):
    half = [s // 2 for s in size]
    slices = tuple(slice(c - h, c + h) for c, h in zip(center, half))
    return volume[slices]


def run_sliding_window(volume, batch_size=64):
    """Classify every grid position, skipping obvious background, in batches for speed."""
    all_centers = generate_grid_centers(volume.shape, PATCH_SIZE, STRIDE)

    # Skip pure-background windows (near-zero variance = empty/black region)
    candidate_centers = []
    for center in all_centers:
        patch = cut_patch(volume, center, PATCH_SIZE)
        if patch.std() > 0.05:   # a real brain tissue window has some texture
            candidate_centers.append(center)

    print(f"  Grid positions: {len(all_centers)} total, {len(candidate_centers)} after background skip")

    stage1_probs = []
    for i in range(0, len(candidate_centers), batch_size):
        batch_centers = candidate_centers[i:i + batch_size]
        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
        patches_tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)  # add channel dim
        with torch.no_grad():
            probs = torch.sigmoid(stage1_model(patches_tensor)).squeeze(1).numpy()
        stage1_probs.extend(probs.tolist())

    # Keep only what Stage 1 flagged, then refine with Stage 2
    stage1_hits = [(c, p) for c, p in zip(candidate_centers, stage1_probs) if p > STAGE1_THRESHOLD]
    print(f"  Stage 1 flagged: {len(stage1_hits)} positions")

    final_detections = []
    for i in range(0, len(stage1_hits), batch_size):
        batch = stage1_hits[i:i + batch_size]
        batch_centers = [c for c, p in batch]
        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
        patches_tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            probs = torch.sigmoid(stage2_model(patches_tensor)).squeeze(1).numpy()
        for center, prob in zip(batch_centers, probs):
            if prob > STAGE2_THRESHOLD:
                final_detections.append((center, float(prob)))

    print(f"  Stage 2 confirmed: {len(final_detections)} positions (before NMS)")
    return final_detections


def apply_nms(detections, min_distance):
    """Greedy NMS: keep the highest-confidence detection in each cluster, drop nearby duplicates."""
    detections = sorted(detections, key=lambda d: d[1], reverse=True)  # strongest first
    kept = []
    for center, prob in detections:
        too_close = any(
            np.linalg.norm(np.array(center) - np.array(kc)) < min_distance
            for kc, kp in kept
        )
        if not too_close:
            kept.append((center, prob))
    return kept


def get_ground_truth_centers(mask):
    labeled, num_blobs = ndimage.label(mask)
    centers = []
    for i in range(1, num_blobs + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) >= 1:
            centers.append(tuple(coords.mean(axis=0)))
    return centers


def match_detections_to_ground_truth(predicted_centers, true_centers, max_distance):
    """Distance-based matching: how many predictions correctly land near a real lesion?"""
    matched_true = set()
    true_positives = 0
    for pred_center in predicted_centers:
        for i, true_center in enumerate(true_centers):
            if i in matched_true:
                continue
            distance = np.linalg.norm(np.array(pred_center) - np.array(true_center))
            if distance <= max_distance:
                true_positives += 1
                matched_true.add(i)
                break
    false_positives = len(predicted_centers) - true_positives
    false_negatives = len(true_centers) - len(matched_true)
    return true_positives, false_positives, false_negatives


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    test_subjects = splits["test"]

    all_results = []
    for subject_name in test_subjects:
        print(f"\nProcessing {subject_name}...")
        subject_folder = PROCESSED_DATA_DIR / subject_name
        volume = np.load(subject_folder / "swi.npy")
        mask = np.load(subject_folder / "mask.npy")

        raw_detections = run_sliding_window(volume)
        final_detections = apply_nms(raw_detections, NMS_MIN_DISTANCE)
        print(f"  After NMS: {len(final_detections)} final detections")

        true_centers = get_ground_truth_centers(mask)
        predicted_centers = [c for c, p in final_detections]

        tp, fp, fn = match_detections_to_ground_truth(predicted_centers, true_centers, MATCH_DISTANCE)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None

        print(f"  True lesions: {len(true_centers)} | TP: {tp} | FP: {fp} | FN: {fn}")

        all_results.append({
            "subject": subject_name,
            "true_lesions": len(true_centers),
            "detections": len(final_detections),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print(results_df.to_string(index=False))

    total_tp = results_df["true_positives"].sum()
    total_fp = results_df["false_positives"].sum()
    total_fn = results_df["false_negatives"].sum()
    overall_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fp_per_subject = total_fp / len(test_subjects)

    print("=" * 60)
    print(f"WHOLE-SCAN LOCALIZATION RESULTS (test set, {len(test_subjects)} subjects)")
    print(f"Sensitivity (lesion-level): {overall_sensitivity:.3f}")
    print(f"False positives per subject: {fp_per_subject:.2f}")

    out_path = REPORTS_DIR / "localization_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()