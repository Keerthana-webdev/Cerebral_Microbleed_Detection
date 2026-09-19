"""
Lesson 14c: Run inference ONCE, save every candidate position's raw
probabilities to disk. Lets us test many threshold combinations instantly
afterward, without re-running the slow neural network each time.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "localization_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

stage1_model = SimpleCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
stage1_model.eval()

stage2_model = SimpleCNN3D().to(DEVICE)
stage2_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
stage2_model.eval()


def generate_grid_centers(volume_shape, patch_size, stride):
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


def process_subject(subject_name, batch_size=64):
    volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
    all_centers = generate_grid_centers(volume.shape, PATCH_SIZE, STRIDE)

    # Moderate background skip (back to the original, gentler threshold)
    candidate_centers = [c for c in all_centers if cut_patch(volume, c, PATCH_SIZE).std() > 0.05]
    print(f"  {subject_name}: {len(candidate_centers)} candidate positions")

    stage1_probs, stage2_probs = [], []
    for i in range(0, len(candidate_centers), batch_size):
        batch_centers = candidate_centers[i:i + batch_size]
        patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch_centers])
        patches_tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            s1 = torch.sigmoid(stage1_model(patches_tensor)).squeeze(1).numpy()
            s2 = torch.sigmoid(stage2_model(patches_tensor)).squeeze(1).numpy()  # run on EVERYONE, not just stage1 passes

        stage1_probs.extend(s1.tolist())
        stage2_probs.extend(s2.tolist())

    np.savez(
        CACHE_DIR / f"{subject_name}.npz",
        centers=np.array(candidate_centers),
        stage1_probs=np.array(stage1_probs),
        stage2_probs=np.array(stage2_probs),
    )


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    test_subjects = splits["test"]

    for subject_name in test_subjects:
        process_subject(subject_name)

    print(f"\nAll predictions cached to {CACHE_DIR}")


if __name__ == "__main__":
    main()