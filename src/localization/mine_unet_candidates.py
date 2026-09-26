"""
Lesson 30a: Generate U-Net candidates on TRAIN subjects, label each as
true CMB or mimic (based on ground truth match), and save these patches.
This becomes training data for a mimic classifier specifically matched
to U-Net's candidate distribution - fixing the domain mismatch found
in Lesson 29.
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
from config import PROCESSED_DATA_DIR, SPLITS_DIR
from src.models.unet3d_model import UNet3D

DEVICE = torch.device("cpu")
MODELS_DIR = PROJECT_ROOT / "models"
PATCHES_DIR = PROJECT_ROOT / "data" / "patches"
OUT_DIR = PATCHES_DIR / "unet_matched_mimic_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)
UNET_THRESHOLD = 0.5
MATCH_DISTANCE = 6
MAX_NEGATIVES_PER_SUBJECT = 60

unet_model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
unet_checkpoint = torch.load(MODELS_DIR / "unet3d" / "unet3d_real_best.pt", map_location=DEVICE)
unet_model.load_state_dict(unet_checkpoint["model_state_dict"])
unet_model.eval()

print("U-Net loaded.\n")


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
    prob_map = np.zeros(volume.shape, dtype=np.float32)
    centers = generate_grid_centers(volume.shape, PATCH_SIZE, STRIDE)
    half = [s // 2 for s in PATCH_SIZE]

    for i in range(0, len(centers), batch_size):
        batch_centers = centers[i:i + batch_size]
        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            outputs = torch.sigmoid(unet_model(tensor)).squeeze(1).numpy()
        for center, patch_probs in zip(batch_centers, outputs):
            slices = tuple(slice(max(c - h, 0), min(c + h, volume.shape[j])) for j, (c, h) in enumerate(zip(center, half)))
            actual_shape = tuple(s.stop - s.start for s in slices)
            cropped = patch_probs[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
            prob_map[slices] = np.maximum(prob_map[slices], cropped)
    return prob_map


def extract_blob_candidates(prob_map, threshold):
    binary_mask = prob_map > threshold
    labeled, num_blobs = ndimage.label(binary_mask)
    if num_blobs == 0:
        return []
    sizes = ndimage.sum(binary_mask, labeled, index=np.arange(1, num_blobs + 1))
    valid_ids = np.where(sizes >= 1)[0] + 1
    if len(valid_ids) == 0:
        return []
    centroids = ndimage.center_of_mass(binary_mask, labeled, valid_ids.tolist())
    return [tuple(int(round(v)) for v in c) for c in centroids]


def get_true_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1) if (labeled == i).sum() >= 1]


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    train_subjects = splits["train"]

    total_positive = 0
    total_negative = 0

    for subject_name in train_subjects:
        volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
        mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
        true_centers = get_true_centers(mask)

        prob_map = build_unet_probability_map(volume)
        candidates = extract_blob_candidates(prob_map, UNET_THRESHOLD)

        subj_out = OUT_DIR / subject_name
        subj_out.mkdir(parents=True, exist_ok=True)

        matched_true = set()
        positives, negatives = [], []

        for candidate in candidates:
            is_match = False
            for i, true_c in enumerate(true_centers):
                if i in matched_true:
                    continue
                if np.linalg.norm(np.array(candidate) - np.array(true_c)) <= MATCH_DISTANCE:
                    is_match = True
                    matched_true.add(i)
                    break
            if is_match:
                positives.append(candidate)
            else:
                negatives.append(candidate)

        # Cap negatives to avoid one noisy subject dominating
        if len(negatives) > MAX_NEGATIVES_PER_SUBJECT:
            negatives = negatives[:MAX_NEGATIVES_PER_SUBJECT]

        for i, center in enumerate(positives):
            patch = cut_patch(volume, center, PATCH_SIZE)
            np.save(subj_out / f"pos_{i}.npy", patch.astype(np.float32))
        for i, center in enumerate(negatives):
            patch = cut_patch(volume, center, PATCH_SIZE)
            np.save(subj_out / f"neg_{i}.npy", patch.astype(np.float32))

        total_positive += len(positives)
        total_negative += len(negatives)

        print(f"  {subject_name}: {len(positives)} positive, {len(negatives)} negative "
              f"(from {len(candidates)} U-Net candidates)")

    print(f"\nTotal: {total_positive} positive, {total_negative} negative U-Net-matched patches")
    print(f"Saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()