"""
Lesson 13a: Label our existing hard-negative patches into vessel-like vs
other-mimic/normal, using a simple elongation measure (shape-based proxy,
since we don't have hand-labeled mimic sub-types).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR


def elongation_score(patch, threshold_percentile=85):
    """
    Measures how 'stretched out' the dark region in a patch is.
    High score = elongated (vessel-like). Low score = round/blob-like.
    """
    # Vessels/microbleeds are usually DARKER than surrounding tissue on SWI,
    # so we look at the darkest region (bottom percentile of brightness)
    dark_threshold = np.percentile(patch, 100 - threshold_percentile)
    dark_mask = patch <= dark_threshold

    labeled, num_blobs = ndimage.label(dark_mask)
    if num_blobs == 0:
        return 1.0  # no clear dark region, treat as "round" by default

    # Find the largest connected dark blob
    sizes = ndimage.sum(dark_mask, labeled, range(1, num_blobs + 1))
    largest_blob_id = np.argmax(sizes) + 1
    coords = np.argwhere(labeled == largest_blob_id)

    if len(coords) < 3:
        return 1.0

    # PCA-style spread: compare the blob's longest axis to its shortest axis.
    # A perfect sphere/circle -> ratio near 1. A stretched tube -> ratio much higher.
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)  # avoid divide-by-zero
    elongation = np.sqrt(eigenvalues.max() / eigenvalues.min())
    return elongation


def main():
    metadata = pd.read_csv(PATCHES_DIR / "mimic_classifier_metadata.csv")

    labels_3class = []
    elongations = []

    for _, row in metadata.iterrows():
        patch = np.load(PATCHES_DIR / row["subject"] / row["file"])

        if row["mimic_label"] == 1:
            label_3class = 0  # true CMB
            elong = None
        else:
            elong = elongation_score(patch)
            # Threshold chosen by inspecting the elongation distribution —
            # document this choice in your report as an empirical threshold
            label_3class = 1 if elong > 1.8 else 2  # 1 = vessel-like, 2 = other mimic/normal

        labels_3class.append(label_3class)
        elongations.append(elong)

    metadata["label_3class"] = labels_3class
    metadata["elongation_score"] = elongations

    print("=== 3-class label distribution ===")
    class_names = {0: "True CMB", 1: "Vessel-like mimic", 2: "Other mimic/normal"}
    print(metadata["label_3class"].map(class_names).value_counts())

    out_path = PATCHES_DIR / "mimic_classifier_3class_metadata.csv"
    metadata.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()