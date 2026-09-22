import json
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "unet_patches"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# PATCH SETTINGS
# ================================================================

PATCH_SIZE = (16, 16, 8)
STRIDE = (8, 8, 4)

MIN_POSITIVE_VOXELS = 1

MAX_POSITIVE_PATCHES_PER_SUBJECT = 100
MAX_NEGATIVE_PATCHES_PER_SUBJECT = 100


# ================================================================
# LOAD SPLITS
# ================================================================

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)


def get_subjects(split_name):

    value = splits[split_name]

    if isinstance(value, dict):
        value = value.get("subjects", [])

    return list(value)


TRAIN_SUBJECTS = get_subjects("train")
VAL_SUBJECTS = get_subjects("val")
TEST_SUBJECTS = get_subjects("test")


# ================================================================
# PATCH EXTRACTION
# ================================================================

def extract_patches_for_subject(subject):

    subject_dir = PREPROCESSED_DIR / subject

    swi_path = subject_dir / "swi.npy"
    mask_path = subject_dir / "mask.npy"

    if not swi_path.exists():
        print(f"[SKIP] Missing SWI: {subject}")
        return []

    if not mask_path.exists():
        print(f"[SKIP] Missing mask: {subject}")
        return []

    swi = np.load(swi_path).astype(np.float32)
    mask = np.load(mask_path)

    mask = (mask > 0).astype(np.uint8)

    if swi.shape != mask.shape:
        print(
            f"[SKIP] Shape mismatch: {subject} "
            f"SWI={swi.shape} MASK={mask.shape}"
        )
        return []

    z_size, y_size, x_size = swi.shape

    pz, py, px = PATCH_SIZE

    positive_patches = []
    negative_patches = []

    # ------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------

    for z in range(0, z_size - pz + 1, STRIDE[0]):

        for y in range(0, y_size - py + 1, STRIDE[1]):

            for x in range(0, x_size - px + 1, STRIDE[2]):

                image_patch = swi[
                    z:z + pz,
                    y:y + py,
                    x:x + px
                ]

                mask_patch = mask[
                    z:z + pz,
                    y:y + py,
                    x:x + px
                ]

                positive_voxels = int(
                    np.count_nonzero(mask_patch)
                )

                if positive_voxels >= MIN_POSITIVE_VOXELS:

                    positive_patches.append(
                        (
                            image_patch.copy(),
                            mask_patch.copy(),
                            z,
                            y,
                            x
                        )
                    )

                else:

                    negative_patches.append(
                        (
                            image_patch.copy(),
                            mask_patch.copy(),
                            z,
                            y,
                            x
                        )
                    )

    # ------------------------------------------------------------
    # Limit number of patches
    # ------------------------------------------------------------

    if len(positive_patches) > MAX_POSITIVE_PATCHES_PER_SUBJECT:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(positive_patches),
            size=MAX_POSITIVE_PATCHES_PER_SUBJECT,
            replace=False
        )

        positive_patches = [
            positive_patches[i]
            for i in indices
        ]

    if len(negative_patches) > MAX_NEGATIVE_PATCHES_PER_SUBJECT:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(negative_patches),
            size=MAX_NEGATIVE_PATCHES_PER_SUBJECT,
            replace=False
        )

        negative_patches = [
            negative_patches[i]
            for i in indices
        ]

    all_patches = []

    for item in positive_patches:

        all_patches.append(
            ("positive", item)
        )

    for item in negative_patches:

        all_patches.append(
            ("negative", item)
        )

    return all_patches


# ================================================================
# PROCESS SPLIT
# ================================================================

def process_split(split_name, subjects):

    split_dir = OUTPUT_DIR / split_name

    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    masks_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = []

    total_positive = 0
    total_negative = 0

    print()
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)

    for subject in subjects:

        print(f"Processing {subject}...")

        patches = extract_patches_for_subject(subject)

        subject_positive = 0
        subject_negative = 0

        for index, (label, item) in enumerate(patches):

            image_patch, mask_patch, z, y, x = item

            filename = (
                f"{subject}_"
                f"{label}_"
                f"{index:05d}_"
                f"z{z}_y{y}_x{x}.npy"
            )

            image_path = images_dir / filename
            mask_path = masks_dir / filename

            np.save(
                image_path,
                image_patch.astype(np.float32)
            )

            np.save(
                mask_path,
                mask_patch.astype(np.uint8)
            )

            metadata.append(
                {
                    "subject": subject,
                    "image_file": filename,
                    "mask_file": filename,
                    "label": 1 if label == "positive" else 0,
                    "z": z,
                    "y": y,
                    "x": x,
                    "positive_voxels": int(
                        np.count_nonzero(mask_patch)
                    )
                }
            )

            if label == "positive":
                subject_positive += 1
                total_positive += 1
            else:
                subject_negative += 1
                total_negative += 1

        print(
            f"  Positive patches: {subject_positive}"
        )

        print(
            f"  Negative patches: {subject_negative}"
        )

    metadata_df = pd.DataFrame(metadata)

    metadata_file = (
        split_dir / "metadata.csv"
    )

    metadata_df.to_csv(
        metadata_file,
        index=False
    )

    print()
    print(
        f"{split_name.upper()} SUMMARY"
    )

    print(
        f"Subjects: {len(subjects)}"
    )

    print(
        f"Positive patches: {total_positive}"
    )

    print(
        f"Negative patches: {total_negative}"
    )

    print(
        f"Total patches: "
        f"{total_positive + total_negative}"
    )

    print(
        f"Metadata: {metadata_file}"
    )

    return metadata_df

print("=" * 80)
print("REAL CMB MASK PATCH GENERATION FOR 3D U-NET")
print("=" * 80)

print(
    f"Preprocessed directory: {PREPROCESSED_DIR}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    f"Patch size: {PATCH_SIZE}"
)

print(
    f"Stride: {STRIDE}"
)

print()

train_df = process_split(
    "train",
    TRAIN_SUBJECTS
)

val_df = process_split(
    "val",
    VAL_SUBJECTS
)

test_df = process_split(
    "test",
    TEST_SUBJECTS
)

print()
print("=" * 80)
print("REAL U-NET PATCH GENERATION COMPLETE")
print("=" * 80)

print(
    f"TRAIN patches: {len(train_df)}"
)

print(
    f"VAL patches:   {len(val_df)}"
)

print(
    f"TEST patches:  {len(test_df)}"
)

print()
print(
    "Output:"
)

print(
    OUTPUT_DIR
)

print("=" * 80)