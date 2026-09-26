"""
Lesson 27a: Blob-based candidate generation, replacing exhaustive sliding
window. CMBs are dark, round, small blobs on SWI - we threshold for dark
regions and use connected-component labeling to get a much smaller,
smarter candidate set, then run the existing trained CNN v3 + mimic
classifier only on these candidates.

Run on VALIDATION set first - test set stays untouched.
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
from model_v3 import DeeperCNN3D
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PATCH_SIZE = (16, 16, 8)

# ---- Blob detection parameters ----
DARK_PERCENTILE = 2.0      # bottom 2% darkest voxels (within brain tissue) are candidate dark regions
MIN_BLOB_VOXELS = 1        # ignore single-voxel noise specks
MAX_BLOB_VOXELS = 60       # CMBs are small; large dark blobs are likely vessels/other structures, not CMBs

STAGE1_MODEL = MODELS_DIR / "candidate_detector_v3_best.pt"
STAGE2_MODEL = MODELS_DIR / "mimic_classifier_best.pt"

stage1_model = DeeperCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(STAGE1_MODEL, map_location=DEVICE))
stage1_model.eval()

stage2_model = SimpleCNN3D().to(DEVICE)
stage2_model.load_state_dict(torch.load(STAGE2_MODEL, map_location=DEVICE))
stage2_model.eval()


def cut_patch(volume, center, size):
    half = [s // 2 for s in size]
    slices = tuple(slice(max(c - h, 0), min(c + h, volume.shape[i])) for i, (c, h) in enumerate(zip(center, half)))
    patch = volume[slices]
    pad = [(0, size[i] - patch.shape[i]) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="constant")
    return patch


def get_true_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1) if (labeled == i).sum() >= 1]


def detect_blob_candidates(volume):
    """Physical prior: CMBs are dark, small, roundish blobs. Threshold + connected components."""
    brain_mask = volume > (volume.min() + 0.1)
    brain_voxels = volume[brain_mask]

    dark_threshold = np.percentile(brain_voxels, DARK_PERCENTILE)
    dark_mask = (volume <= dark_threshold) & brain_mask

    labeled, num_blobs = ndimage.label(dark_mask)
    print(f"    (raw blobs before size filtering: {num_blobs})")

    if num_blobs == 0:
        return []

    # Vectorized size computation - avoids looping with argwhere per blob
    sizes = ndimage.sum(dark_mask, labeled, index=np.arange(1, num_blobs + 1))

    valid_label_ids = np.where((sizes >= MIN_BLOB_VOXELS) & (sizes <= MAX_BLOB_VOXELS))[0] + 1

    if len(valid_label_ids) == 0:
        return []

    # Vectorized centroid computation for only the valid blobs
    centroids = ndimage.center_of_mass(dark_mask, labeled, valid_label_ids.tolist())

    candidates = [tuple(int(round(v)) for v in c) for c in centroids]
    return candidates


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

    MATCH_DISTANCE = 6
    STAGE1_THRESHOLD = 0.5
    STAGE2_THRESHOLD = 0.5
    NMS_DISTANCE = 15

    print("=" * 80)
    print("STEP 1: Blob detector recall check (before any classifier)")
    print("=" * 80)

    total_blob_candidates = 0
    total_true_lesions = 0
    total_blob_recall_tp = 0

    per_subject_blobs = {}

    for subject_name in val_subjects:
        volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
        mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
        true_centers = get_true_centers(mask)

        blob_candidates = detect_blob_candidates(volume)
        per_subject_blobs[subject_name] = (volume, blob_candidates, true_centers)

        tp, fp, fn = match_to_ground_truth(blob_candidates, true_centers, MATCH_DISTANCE)

        total_blob_candidates += len(blob_candidates)
        total_true_lesions += len(true_centers)
        total_blob_recall_tp += tp

        print(f"  {subject_name}: {len(blob_candidates)} blob candidates "
              f"(vs sliding window's ~8,000-12,000), true lesions={len(true_centers)}, "
              f"blob-stage recall TP={tp}/{len(true_centers)}")

    blob_recall = total_blob_recall_tp / total_true_lesions if total_true_lesions > 0 else 0
    print(f"\n🔑 CRITICAL NUMBER: Blob detector recall = {blob_recall:.3f} "
          f"({total_blob_recall_tp}/{total_true_lesions} true lesions captured as candidates)")
    print(f"   Average candidates per subject: {total_blob_candidates / len(val_subjects):.1f} "
          f"(vs sliding window's ~9,000)")

    if blob_recall < 0.5:
        print("\n⚠️  WARNING: Blob detector is missing more than half of true lesions.")
        print("   This means even a perfect classifier downstream cannot recover them.")
        print("   We may need to loosen DARK_PERCENTILE/MAX_BLOB_VOXELS before proceeding further.")

    print("\n" + "=" * 80)
    print("STEP 2: Run CNN v3 + mimic classifier on blob candidates only")
    print("=" * 80)

    total_tp = total_fp = total_fn = 0

    for subject_name in val_subjects:
        volume, blob_candidates, true_centers = per_subject_blobs[subject_name]

        if not blob_candidates:
            continue

        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in blob_candidates])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            s1_probs = torch.sigmoid(stage1_model(tensor)).squeeze(1).numpy()

        stage1_pass = [(c, p) for c, p in zip(blob_candidates, s1_probs) if p > STAGE1_THRESHOLD]

        if stage1_pass:
            pass_centers = [c for c, p in stage1_pass]
            pass_patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in pass_centers])
            pass_tensor = torch.tensor(pass_patches, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                s2_probs = torch.sigmoid(stage2_model(pass_tensor)).squeeze(1).numpy()
            final = [(c, float(p)) for c, p in zip(pass_centers, s2_probs) if p > STAGE2_THRESHOLD]
        else:
            final = []

        if final:
            centers, probs = zip(*final)
            kept = apply_nms(list(centers), list(probs), NMS_DISTANCE)
        else:
            kept = []

        tp, fp, fn = match_to_ground_truth(kept, true_centers, MATCH_DISTANCE)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"  {subject_name}: true={len(true_centers)}, final_detections={len(kept)}, "
              f"TP={tp}, FP={fp}, FN={fn}")

    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fp_per_subject = total_fp / len(val_subjects)

    print(f"\n{'=' * 80}")
    print(f"BLOB-BASED PIPELINE — VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"FP/subject:  {fp_per_subject:.2f}")
    print(f"\nComparison vs sliding-window v3 (validation, matched point): sensitivity=0.238, FP/subject=19.82")


if __name__ == "__main__":
    main()