"""
Lesson 29: U-Net-as-candidate-generator cascade.
U-Net (high sensitivity, ~41%) generates candidate blobs by sliding
across the volume and aggregating per-voxel probabilities. The mimic
classifier (Stage 2) then filters these candidates to cut U-Net's high
false-positive rate. Run on VALIDATION set - test set untouched.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR
from src.models.unet3d_model import UNet3D
from model import SimpleCNN3D  # mimic classifier architecture

DEVICE = torch.device("cpu")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)
MATCH_DISTANCE = 6
MIN_BLOB_VOXELS = 1
NMS_DISTANCE = 15

# ---- Load U-Net (candidate generator) ----
unet_model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
unet_checkpoint = torch.load(MODELS_DIR / "unet3d" / "unet3d_real_best.pt", map_location=DEVICE)
unet_model.load_state_dict(unet_checkpoint["model_state_dict"])
unet_model.eval()

# ---- Load mimic classifier (Stage 2 filter) ----
mimic_model = SimpleCNN3D().to(DEVICE)
mimic_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
mimic_model.eval()

print("U-Net (candidate generator) and mimic classifier (filter) loaded.\n")


def cut_patch(volume, center, size):
    half = [s // 2 for s in size]
    slices = tuple(slice(max(c - h, 0), min(c + h, volume.shape[i])) for i, (c, h) in enumerate(zip(center, half)))
    patch = volume[slices]
    pad = [(0, size[i] - patch.shape[i]) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="constant")
    return patch


def generate_grid_centers(volume_shape, patch_size, stride):
    half = [s // 2 for s in patch_size]
    centers = []
    for x in range(half[0], volume_shape[0] - half[0], stride[0]):
        for y in range(half[1], volume_shape[1] - half[1], stride[1]):
            for z in range(half[2], volume_shape[2] - half[2], stride[2]):
                centers.append((x, y, z))
    return centers


def build_unet_probability_map(volume, batch_size=32):
    """Slide U-Net across the volume, aggregate overlapping predictions with max."""
    prob_map = np.zeros(volume.shape, dtype=np.float32)
    count_map = np.zeros(volume.shape, dtype=np.float32)

    centers = generate_grid_centers(volume.shape, PATCH_SIZE, STRIDE)
    half = [s // 2 for s in PATCH_SIZE]

    for i in range(0, len(centers), batch_size):
        batch_centers = centers[i:i + batch_size]
        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            outputs = torch.sigmoid(unet_model(tensor)).squeeze(1).numpy()  # shape: (batch, 16, 16, 8)

        for center, patch_probs in zip(batch_centers, outputs):
            slices = tuple(slice(max(c - h, 0), min(c + h, volume.shape[j])) for j, (c, h) in enumerate(zip(center, half)))
            actual_shape = tuple(s.stop - s.start for s in slices)
            cropped_probs = patch_probs[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
            # Max aggregation: keep the highest probability seen for each voxel across overlapping windows
            prob_map[slices] = np.maximum(prob_map[slices], cropped_probs)

    return prob_map


def extract_blob_candidates(prob_map, threshold):
    binary_mask = prob_map > threshold
    labeled, num_blobs = ndimage.label(binary_mask)
    if num_blobs == 0:
        return []
    sizes = ndimage.sum(binary_mask, labeled, index=np.arange(1, num_blobs + 1))
    valid_ids = np.where(sizes >= MIN_BLOB_VOXELS)[0] + 1
    if len(valid_ids) == 0:
        return []
    centroids = ndimage.center_of_mass(binary_mask, labeled, valid_ids.tolist())
    return [tuple(int(round(v)) for v in c) for c in centroids]


def get_true_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1) if (labeled == i).sum() >= 1]


def match_to_ground_truth(predicted, true, max_distance):
    matched = set()
    tp = 0
    for p in predicted:
        for i, t in enumerate(true):
            if i in matched:
                continue
            if np.linalg.norm(np.array(p) - np.array(t)) <= max_distance:
                tp += 1
                matched.add(i)
                break
    fp = len(predicted) - tp
    fn = len(true) - len(matched)
    return tp, fp, fn


def apply_nms(centers, probs, min_distance):
    order = np.argsort(-np.array(probs))
    kept = []
    for i in order:
        c = centers[i]
        if not any(np.linalg.norm(np.array(c) - np.array(k)) < min_distance for k in kept):
            kept.append(c)
    return kept


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    val_subjects = splits["val"]

    UNET_THRESHOLD = 0.5   # threshold on U-Net's aggregated probability map
    MIMIC_THRESHOLD = 0.5  # threshold on Stage-2 mimic classifier

    print("=" * 80)
    print("STEP 1: U-Net candidate generation - checking recall per subject")
    print("=" * 80)

    per_subject_data = {}
    total_unet_tp = total_true = 0

    for subject_name in val_subjects:
        volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
        mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
        true_centers = get_true_centers(mask)

        prob_map = build_unet_probability_map(volume)
        candidates = extract_blob_candidates(prob_map, UNET_THRESHOLD)

        tp, fp, fn = match_to_ground_truth(candidates, true_centers, MATCH_DISTANCE)
        total_unet_tp += tp
        total_true += len(true_centers)

        per_subject_data[subject_name] = (volume, candidates, true_centers)

        print(f"  {subject_name}: U-Net candidates={len(candidates)}, true={len(true_centers)}, "
              f"U-Net-stage recall TP={tp}/{len(true_centers)}")

    unet_recall = total_unet_tp / total_true if total_true > 0 else 0
    print(f"\n🔑 U-Net candidate-stage recall: {unet_recall:.3f} ({total_unet_tp}/{total_true})")

    print("\n" + "=" * 80)
    print("STEP 2: Filter U-Net candidates with mimic classifier")
    print("=" * 80)

    total_tp = total_fp = total_fn = 0

    for subject_name in val_subjects:
        volume, candidates, true_centers = per_subject_data[subject_name]

        if not candidates:
            print(f"  {subject_name}: no candidates, true={len(true_centers)}, TP=0, FP=0, FN={len(true_centers)}")
            total_fn += len(true_centers)
            continue

        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in candidates])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            mimic_probs = torch.sigmoid(mimic_model(tensor)).squeeze(1).numpy()

        passed = [(c, float(p)) for c, p in zip(candidates, mimic_probs) if p > MIMIC_THRESHOLD]

        if passed:
            centers, probs = zip(*passed)
            kept = apply_nms(list(centers), list(probs), NMS_DISTANCE)
        else:
            kept = []

        tp, fp, fn = match_to_ground_truth(kept, true_centers, MATCH_DISTANCE)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"  {subject_name}: true={len(true_centers)}, final={len(kept)}, TP={tp}, FP={fp}, FN={fn}")

    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fp_per_subject = total_fp / len(val_subjects)

    print(f"\n{'=' * 80}")
    print(f"U-NET CASCADE PIPELINE — VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"FP/subject:  {fp_per_subject:.2f}")
    print(f"\n=== Comparison (validation, matched-style points) ===")
    print(f"U-Net alone (from earlier test):     Sensitivity 41.38%, FP/scan 127.18")
    print(f"CNN v3 alone (validation):            Sensitivity 23.8%, FP/subject 19.82")
    print(f"U-Net cascade (this experiment):      Sensitivity {sensitivity*100:.1f}%, FP/subject {fp_per_subject:.2f}")


if __name__ == "__main__":
    main()