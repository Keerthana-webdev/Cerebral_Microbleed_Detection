import os
import json
import csv
import numpy as np
from scipy import ndimage
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = PROJECT_ROOT / "data" / "patches"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(patch):
    """
    Extract numerical features from a 3D MRI patch.

    Input:
        patch: 3D numpy array

    Output:
        1D numpy feature vector
    """

    patch = np.asarray(patch, dtype=np.float32)

    flat = patch.ravel()

    # ---------------------------------------------------------
    # 1. Basic intensity statistics
    # ---------------------------------------------------------
    features = [
        np.min(flat),
        np.max(flat),
        np.mean(flat),
        np.std(flat),
        np.median(flat),
        np.ptp(flat),
    ]

    # ---------------------------------------------------------
    # 2. Percentile features
    # ---------------------------------------------------------
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    for p in percentiles:
        features.append(np.percentile(flat, p))

    # ---------------------------------------------------------
    # 3. Energy / variance-related features
    # ---------------------------------------------------------
    features.extend([
        np.mean(flat ** 2),
        np.sqrt(np.mean(flat ** 2)),
        np.var(flat),
    ])

    # ---------------------------------------------------------
    # 4. Gradient features
    # ---------------------------------------------------------
    gx, gy, gz = np.gradient(patch)

    gradient_magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2 +
        gz ** 2
    )

    features.extend([
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.max(gradient_magnitude),
        np.percentile(gradient_magnitude, 90),
        np.percentile(gradient_magnitude, 95),
    ])

    # ---------------------------------------------------------
    # 5. Gradient direction / axis statistics
    # ---------------------------------------------------------
    features.extend([
        np.mean(np.abs(gx)),
        np.mean(np.abs(gy)),
        np.mean(np.abs(gz)),
        np.std(gx),
        np.std(gy),
        np.std(gz),
    ])

    # ---------------------------------------------------------
    # 6. Local variation
    # ---------------------------------------------------------
    local_mean = ndimage.uniform_filter(patch, size=3)
    local_variation = np.abs(patch - local_mean)

    features.extend([
        np.mean(local_variation),
        np.std(local_variation),
        np.max(local_variation),
        np.percentile(local_variation, 90),
        np.percentile(local_variation, 95),
    ])

    # ---------------------------------------------------------
    # 7. Simple threshold-based intensity proportions
    # ---------------------------------------------------------
    mean_value = np.mean(flat)
    std_value = np.std(flat)

    thresholds = [
        mean_value,
        mean_value + std_value,
        mean_value + 2 * std_value,
        mean_value - std_value,
    ]

    for threshold in thresholds:
        features.append(np.mean(flat > threshold))

    # ---------------------------------------------------------
    # 8. Spatial center features
    # ---------------------------------------------------------
    z, y, x = patch.shape

    center_z = patch[z // 4: 3 * z // 4,
                     y // 4: 3 * y // 4,
                     x // 4: 3 * x // 4]

    center_flat = center_z.ravel()

    features.extend([
        np.mean(center_flat),
        np.std(center_flat),
        np.max(center_flat),
        np.median(center_flat),
    ])

    nonzero_count = np.count_nonzero(patch)
    total_count = patch.size

    features.append(nonzero_count / max(total_count, 1))

    features = np.asarray(features, dtype=np.float32)

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


def load_split_subjects():
    with open(SPLIT_FILE, "r") as f:
        return json.load(f)


def load_metadata():
    metadata_file = PATCH_DIR / "patch_metadata.csv"

    with open(metadata_file, "r", newline="") as f:
        return list(csv.DictReader(f))


def build_dataset(subjects, metadata):
    """
    Build feature matrix and labels for a list of subjects.
    """

    subject_set = set(subjects)

    X = []
    y = []
    files = []

    for row in metadata:

        subject = row["subject"]

        if subject not in subject_set:
            continue

        filename = row["file"]
        label = int(row["label"])

        patch_path = PATCH_DIR / subject / filename

        if not patch_path.exists():
            print(f"WARNING: missing patch: {patch_path}")
            continue

        patch = np.load(patch_path)

        features = extract_features(patch)

        X.append(features)
        y.append(label)
        files.append(f"{subject}/{filename}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X, y, files


def save_dataset(name, X, y, files):
    np.save(OUTPUT_DIR / f"{name}_X.npy", X)
    np.save(OUTPUT_DIR / f"{name}_y.npy", y)

    with open(OUTPUT_DIR / f"{name}_files.txt", "w") as f:
        for item in files:
            f.write(item + "\n")

    print(f"\n{name.upper()} DATASET")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Positive:", int(np.sum(y == 1)))
    print("Negative:", int(np.sum(y == 0)))


def main():

    print("=" * 70)
    print("XGBOOST FEATURE EXTRACTION")
    print("=" * 70)

    splits = load_split_subjects()
    metadata = load_metadata()

    print("\nLoading patches...")

    for split_name in ["train", "val", "test"]:

        X, y, files = build_dataset(
            splits[split_name],
            metadata
        )

        save_dataset(
            split_name,
            X,
            y,
            files
        )

    print("\nFeature extraction completed successfully.")


if __name__ == "__main__":
    main()