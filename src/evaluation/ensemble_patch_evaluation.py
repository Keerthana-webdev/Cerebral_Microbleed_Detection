"""
Lesson 25: Soft-voting ensemble of 3D CNN v3 + 3D ResNet + 3D DenseNet.
Averages predicted probabilities across all three models and evaluates
on the SAME frozen patch-level test set used throughout this project.
XGBoost excluded (different feature space, can't average raw probs
directly without extra calibration). U-Net excluded (segmentation task,
not classification - not comparable).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PATCHES_DIR, SPLITS_DIR
import json

from model_v3 import DeeperCNN3D          # your 3D CNN v3 (Stage 1 candidate detector)
from src.models.resnet3d_model import ResNet3D
from src.models.densenet3d_model import DenseNet3D

DEVICE = torch.device("cpu")
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Load all three models ----
cnn_model = DeeperCNN3D().to(DEVICE)
cnn_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_v3_best.pt", map_location=DEVICE))
cnn_model.eval()

resnet_model = ResNet3D(num_classes=2).to(DEVICE)
resnet_checkpoint = torch.load(MODELS_DIR / "resnet3d" / "resnet3d_best.pt", map_location=DEVICE)
resnet_model.load_state_dict(resnet_checkpoint["model_state_dict"])
resnet_model.eval()

densenet_model = DenseNet3D(num_classes=2).to(DEVICE)
densenet_checkpoint = torch.load(MODELS_DIR / "densenet3d" / "densenet3d_best.pt", map_location=DEVICE)
# DenseNet checkpoint may be saved directly as a state_dict (per your earlier sweep script)
# or wrapped like ResNet's - handle both safely:
if isinstance(densenet_checkpoint, dict) and "model_state_dict" in densenet_checkpoint:
    densenet_model.load_state_dict(densenet_checkpoint["model_state_dict"])
else:
    densenet_model.load_state_dict(densenet_checkpoint)
densenet_model.eval()

print("All three models loaded successfully.\n")


def get_cnn_prob(patch_tensor):
    """3D CNN v3 outputs a single logit -> sigmoid -> P(CMB)"""
    with torch.no_grad():
        logit = cnn_model(patch_tensor)
        return torch.sigmoid(logit).item()


def get_resnet_prob(patch_tensor):
    """ResNet outputs 2 class logits -> softmax -> P(class 1 = CMB)"""
    with torch.no_grad():
        logits = resnet_model(patch_tensor)
        probs = F.softmax(logits, dim=1)
        return probs[0, 1].item()


def get_densenet_prob(patch_tensor):
    """DenseNet outputs 2 class logits -> softmax -> P(class 1 = CMB)"""
    with torch.no_grad():
        logits = densenet_model(patch_tensor)
        probs = F.softmax(logits, dim=1)
        return probs[0, 1].item()


# ---- Load the SAME frozen test set used throughout the project ----
with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
test_subjects = set(splits["test"])

metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
test_metadata = metadata[metadata["subject"].isin(test_subjects)].reset_index(drop=True)

print(f"Evaluating ensemble on {len(test_metadata)} test patches (same frozen test set)\n")

results = []
for _, row in test_metadata.iterrows():
    patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
    tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    true_label = int(row["label"])

    p_cnn = get_cnn_prob(tensor)
    p_resnet = get_resnet_prob(tensor)
    p_densenet = get_densenet_prob(tensor)

    # Soft voting: simple average of the three probabilities
    # Weighted voting: weight each model by its individual F1 score,
    # giving the strongest model (CNN v3) more influence than naive averaging
    W_CNN, W_RESNET, W_DENSENET = 93.60, 89.92, 88.71
    total_weight = W_CNN + W_RESNET + W_DENSENET

    ensemble_prob = (
        (p_cnn * W_CNN) + (p_resnet * W_RESNET) + (p_densenet * W_DENSENET)
    ) / total_weight
    predicted = 1 if ensemble_prob > 0.5 else 0

    results.append({
        "subject": row["subject"], "true_label": true_label,
        "p_cnn": p_cnn, "p_resnet": p_resnet, "p_densenet": p_densenet,
        "ensemble_prob": ensemble_prob, "predicted": predicted,
    })

df = pd.DataFrame(results)

tp = ((df.predicted == 1) & (df.true_label == 1)).sum()
fp = ((df.predicted == 1) & (df.true_label == 0)).sum()
tn = ((df.predicted == 0) & (df.true_label == 0)).sum()
fn = ((df.predicted == 0) & (df.true_label == 1)).sum()

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("=" * 70)
print("WEIGHTED ENSEMBLE (F1-weighted: CNN v3 + 3D ResNet + 3D DenseNet) — PATCH-LEVEL RESULTS")
print("=" * 70)
print(f"Confusion Matrix:")
print(f"                  Predicted CMB   Predicted Normal")
print(f"  Actual CMB           {tp:4d}              {fn:4d}")
print(f"  Actual Normal        {fp:4d}              {tn:4d}")
print(f"\nPrecision:    {precision:.4f}")
print(f"Recall/Sens:  {recall:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"Accuracy:     {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"F1-Score:     {f1:.4f}")

print("\n=== Comparison vs individual models (from your earlier results) ===")
print(f"3D CNN v3 alone:   Accuracy 98.20%, F1 93.60%")
print(f"3D ResNet alone:   Accuracy 96.65%, F1 89.92%")
print(f"3D DenseNet alone: Accuracy 96.39%, F1 88.71%")
print(f"ENSEMBLE:          Accuracy {accuracy*100:.2f}%, F1 {f1*100:.2f}%")

REPORTS_DIR = PROJECT_ROOT / "reports"
df.to_csv(REPORTS_DIR / "ensemble_weighted_patch_predictions.csv", index=False)
print(f"\nPer-patch predictions saved to: {REPORTS_DIR / 'ensemble_patch_predictions.csv'}")