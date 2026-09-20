"""
Lesson 20: Mine a SECOND round of whole-volume hard negatives, this time
from v2's own mistakes (not v1's). Train subjects only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PROCESSED_DATA_DIR, SPLITS_DIR
from model import SimpleCNN3D   # v2 was trained with the ORIGINAL architecture - load it with that class

DEVICE = torch.device("cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PATCHES_DIR = Path(__file__).resolve().parents[2] / "data" / "patches"
OUT_DIR = PATCHES_DIR / "wholevolume_hard_negatives_round2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)
STAGE1_THRESHOLD = 0.5
MATCH_DISTANCE = 6
MAX_NEGATIVES_PER_SUBJECT = 40

stage1_model = SimpleCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_v2_best.pt", map_location=DEVICE))
stage1_model.eval()


def cut_patch(volume, center, size):
    half = [s // 2 for s in size]
    slices = tuple(slice(c - h, c + h) for c, h in zip(center, half))
    return volume[slices]


def generate_grid(shape):
    half = [s // 2 for s in PATCH_SIZE]
    centers = []
    for x in range(half[0], shape[0] - half[0], STRIDE[0]):
        for y in range(half[1], shape[1] - half[1], STRIDE[1]):
            for z in range(half[2], shape[2] - half[2], STRIDE[2]):
                centers.append((x, y, z))
    return centers


def get_true_centers(mask):
    labeled, n = ndimage.label(mask)
    return [tuple(np.argwhere(labeled == i).mean(axis=0)) for i in range(1, n + 1)]


def main():
    with open(SPLITS_DIR / "splits.json") as f:
        splits = json.load(f)
    train_subjects = splits["train"]

    total_saved = 0
    for subject_name in train_subjects:
        volume = np.load(PROCESSED_DATA_DIR / subject_name / "swi.npy")
        mask = np.load(PROCESSED_DATA_DIR / subject_name / "mask.npy")
        true_centers = get_true_centers(mask)

        all_centers = generate_grid(volume.shape)
        candidates = [c for c in all_centers if cut_patch(volume, c, PATCH_SIZE).std() > 0.05]

        mistakes = []
        for i in range(0, len(candidates), 64):
            batch = candidates[i:i + 64]
            patches = np.stack([cut_patch(volume, c, PATCH_SIZE) for c in batch])
            tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                probs = torch.sigmoid(stage1_model(tensor)).squeeze(1).numpy()

            for center, prob in zip(batch, probs):
                if prob <= STAGE1_THRESHOLD:
                    continue
                is_real = any(
                    np.linalg.norm(np.array(center) - np.array(tc)) <= MATCH_DISTANCE
                    for tc in true_centers
                )
                if not is_real:
                    mistakes.append((center, prob))

        mistakes.sort(key=lambda m: -m[1])
        mistakes = mistakes[:MAX_NEGATIVES_PER_SUBJECT]

        subj_out = OUT_DIR / subject_name
        subj_out.mkdir(exist_ok=True)
        for i, (center, prob) in enumerate(mistakes):
            patch = cut_patch(volume, center, PATCH_SIZE)
            np.save(subj_out / f"hardneg_r2_{i}.npy", patch.astype(np.float32))

        total_saved += len(mistakes)
        print(f"{subject_name}: {len(mistakes)} round-2 hard negatives mined")

    print(f"\nTotal round-2 hard negatives: {total_saved}")
    print(f"Saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()