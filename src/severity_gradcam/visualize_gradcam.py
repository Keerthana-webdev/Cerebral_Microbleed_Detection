"""
Lesson 12c: Run Grad-CAM on a few real test patches and save side-by-side
images: original scan vs. the model's "attention" heatmap.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PATCHES_DIR
from model import SimpleCNN3D
from gradcam_3d import GradCAM3D

DEVICE = torch.device("cpu")  # Grad-CAM backward hooks are simplest/safest on CPU
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

model = SimpleCNN3D().to(DEVICE)
model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))

# Target the last ReLU layer (index 6) — see Lesson 12 explanation for why
target_layer = model.features[6]
gradcam = GradCAM3D(model, target_layer)

# Pick a few real, correctly-detected microbleed patches to visualize
results = pd.read_csv(PATCHES_DIR / "confidence_evaluation_results.csv")
correct_positives = results[(results["true_label"] == 1) & (results["predicted_label"] == 1)]
sample = correct_positives.head(4)

fig, axes = plt.subplots(2, len(sample), figsize=(4 * len(sample), 8))

for i, (_, row) in enumerate(sample.iterrows()):
    # Re-find the original file path (need to look it up from patch_metadata)
    patch_meta = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
    match = patch_meta[(patch_meta["subject"] == row["subject"]) & (patch_meta["label"] == 1)].iloc[0]
    patch_path = PATCHES_DIR / row["subject"] / match["file"]

    patch = np.load(patch_path)
    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    patch_tensor.requires_grad_(True)

    heatmap = gradcam.generate(patch_tensor)

    mid_slice = patch.shape[2] // 2  # middle slice along the depth axis

    axes[0, i].imshow(patch[:, :, mid_slice].T, cmap="gray", origin="lower")
    axes[0, i].set_title(f"{row['subject']}\nOriginal patch")
    axes[0, i].axis("off")

    axes[1, i].imshow(patch[:, :, mid_slice].T, cmap="gray", origin="lower")
    axes[1, i].imshow(heatmap[:, :, mid_slice].T, cmap="jet", alpha=0.5, origin="lower")
    axes[1, i].set_title("Grad-CAM: model's focus")
    axes[1, i].axis("off")

plt.tight_layout()
out_path = REPORTS_DIR / "gradcam_examples.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved Grad-CAM visualization to: {out_path}")
plt.show()