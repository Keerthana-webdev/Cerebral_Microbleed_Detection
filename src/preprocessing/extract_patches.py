"""
Lesson 6: Cut small 3D cubes ("patches") out of each subject's brain scan —
one cube centered on each real microbleed (positive examples), and several
random cubes with no microbleed (negative examples). This is the actual
training data our neural network will learn from.
"""

import random
import numpy as np
import pandas as pd
from scipy import ndimage
from config import PROCESSED_DATA_DIR, PATCHES_DIR

PATCH_SIZE = (16, 16, 8)          # size of each cube, in voxels (x, y, z)
NEGATIVES_PER_SUBJECT = 30         # random "empty" patches to grab per subject
MIN_LESION_VOXELS = 1              # ignore specks smaller than this (likely noise)
RANDOM_SEED = 42


def cut_patch(volume, center, size):
    """Cuts a small cube out of the big 3D scan, centered at `center`."""
    half = [s // 2 for s in size]
    slices = tuple(
        slice(max(c - h, 0), min(c + h, volume.shape[axis]))
        for axis, (c, h) in enumerate(zip(center, half))
    )
    patch = volume[slices]
    # If the cube got cut off near the edge of the brain, pad it back to full size
    pad = [(0, size[i] - patch.shape[i]) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="constant")
    return patch


def find_lesion_centers(mask):
    """Finds the center coordinate of every distinct microbleed blob in the mask."""
    labeled_blobs, num_blobs = ndimage.label(mask)
    centers = []
    for blob_id in range(1, num_blobs + 1):
        coords = np.argwhere(labeled_blobs == blob_id)
        if len(coords) < MIN_LESION_VOXELS:
            continue
        center = tuple(coords.mean(axis=0).astype(int))
        centers.append(center)
    return centers


def pick_random_negative_centers(volume_shape, count, avoid_centers, margin=8):
    """Picks random coordinates that are NOT near any real microbleed."""
    random.seed(RANDOM_SEED)
    picked = []
    attempts = 0
    while len(picked) < count and attempts < count * 20:
        attempts += 1
        candidate = tuple(random.randint(margin, dim - margin - 1) for dim in volume_shape)
        far_enough = all(
            sum((a - b) ** 2 for a, b in zip(candidate, avoid)) > margin ** 2
            for avoid in avoid_centers
        )
        if far_enough:
            picked.append(candidate)
    return picked


def process_subject(subject_name):
    subject_folder = PROCESSED_DATA_DIR / subject_name
    volume = np.load(subject_folder / "swi.npy")
    mask = np.load(subject_folder / "mask.npy")

    out_folder = PATCHES_DIR / subject_name
    out_folder.mkdir(parents=True, exist_ok=True)

    records = []
    lesion_centers = find_lesion_centers(mask)

    for i, center in enumerate(lesion_centers):
        patch = cut_patch(volume, center, PATCH_SIZE)
        filename = f"positive_{i}.npy"
        np.save(out_folder / filename, patch.astype(np.float32))
        records.append({"subject": subject_name, "file": filename, "label": 1})

    negative_centers = pick_random_negative_centers(volume.shape, NEGATIVES_PER_SUBJECT, lesion_centers)
    for i, center in enumerate(negative_centers):
        patch = cut_patch(volume, center, PATCH_SIZE)
        filename = f"negative_{i}.npy"
        np.save(out_folder / filename, patch.astype(np.float32))
        records.append({"subject": subject_name, "file": filename, "label": 0})

    return records


def main():
    subjects = sorted([d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()])
    print(f"Extracting patches from {len(subjects)} subjects...")

    all_records = []
    for i, subject_name in enumerate(subjects, start=1):
        records = process_subject(subject_name)
        all_records.extend(records)
        print(f"[{i}/{len(subjects)}] {subject_name}: {len(records)} patches")

    df = pd.DataFrame(all_records)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATCHES_DIR / "patch_metadata.csv", index=False)

    print(f"\nTotal patches: {len(df)}")
    print(df["label"].value_counts().rename({1: "positive (microbleed)", 0: "negative (normal)"}))
    print(f"Metadata saved to: {PATCHES_DIR / 'patch_metadata.csv'}")


if __name__ == "__main__":
    main()