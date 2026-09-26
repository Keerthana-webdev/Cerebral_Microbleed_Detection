"""
Lesson 28a: Compute per-model patch-level probabilities on the VALIDATION
set (CNN v3, ResNet, DenseNet), needed to TRAIN a meta-learner. Test set
stays untouched until the meta-learner is trained.
"""

import json
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
from model_v3 import DeeperCNN3D
from src.models.resnet3d_model import ResNet3D
from src.models.densenet3d_model import DenseNet3D

DEVICE = torch.device("cpu")
MODELS_DIR = PROJECT_ROOT / "models"

cnn_model = DeeperCNN3D().to(DEVICE)
cnn_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_v3_best.pt", map_location=DEVICE))
cnn_model.eval()

resnet_model = ResNet3D(num_classes=2).to(DEVICE)
resnet_ckpt = torch.load(MODELS_DIR / "resnet3d" / "resnet3d_best.pt", map_location=DEVICE)
resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
resnet_model.eval()

densenet_model = DenseNet3D(num_classes=2).to(DEVICE)
densenet_ckpt = torch.load(MODELS_DIR / "densenet3d" / "densenet3d_best.pt", map_location=DEVICE)
if isinstance(densenet_ckpt, dict) and "model_state_dict" in densenet_ckpt:
    densenet_model.load_state_dict(densenet_ckpt["model_state_dict"])
else:
    densenet_model.load_state_dict(densenet_ckpt)
densenet_model.eval()

print("All three models loaded.\n")

with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
val_subjects = set(splits["val"])

metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
val_metadata = metadata[metadata["subject"].isin(val_subjects)].reset_index(drop=True)

print(f"Computing per-model probabilities on {len(val_metadata)} VALIDATION patches\n")

rows = []
for _, row in val_metadata.iterrows():
    patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
    tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    true_label = int(row["label"])

    with torch.no_grad():
        p_cnn = torch.sigmoid(cnn_model(tensor)).item()
        p_resnet = F.softmax(resnet_model(tensor), dim=1)[0, 1].item()
        p_densenet = F.softmax(densenet_model(tensor), dim=1)[0, 1].item()

    rows.append({
        "subject": row["subject"], "true_label": true_label,
        "p_cnn": p_cnn, "p_resnet": p_resnet, "p_densenet": p_densenet,
    })

df = pd.DataFrame(rows)
REPORTS_DIR = PROJECT_ROOT / "reports"
df.to_csv(REPORTS_DIR / "meta_learner_validation_features.csv", index=False)
print(f"Saved to: {REPORTS_DIR / 'meta_learner_validation_features.csv'}")
print(f"Positive: {(df.true_label==1).sum()}, Negative: {(df.true_label==0).sum()}")