"""
Lesson 9b: Train the mimic-aware classifier — Stage 2 of your pipeline.
Its only job: given something Stage 1 flagged, is it a REAL microbleed
or a mimic (vessel/calcification/artifact)?
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PATCHES_DIR
from model import SimpleCNN3D
from mimic_dataset import MimicDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

full_metadata = pd.read_csv(PATCHES_DIR / "mimic_classifier_metadata.csv")

# Simple internal split just to monitor training (NOT the final evaluation —
# that happens later on the real held-out test set in Phase 7).
train_df, val_df = train_test_split(
    full_metadata, test_size=0.15, stratify=full_metadata["mimic_label"], random_state=42
)
print(f"Internal train: {len(train_df)} | Internal val: {len(val_df)}")

train_loader = DataLoader(MimicDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MimicDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)

n_pos = train_df["mimic_label"].sum()
n_neg = len(train_df) - n_pos
pos_weight_value = n_neg / n_pos
print(f"Positive class weight: {pos_weight_value:.2f}")

model = SimpleCNN3D().to(DEVICE)  # same architecture, fresh weights, focused task
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
best_val_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for patches, labels in train_loader:
        patches, labels = patches.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(patches).squeeze(1)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    tp, fn, tn, fp = 0, 0, 0, 0
    with torch.no_grad():
        for patches, labels in val_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            probs = torch.sigmoid(model(patches).squeeze(1))
            preds = (probs > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Epoch {epoch}/{EPOCHS} | loss: {total_loss/len(train_loader):.4f} | "
          f"precision: {precision:.3f} | recall: {recall:.3f} | F1: {f1:.3f}")

    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), MODELS_DIR / "mimic_classifier_best.pt")
        print(f"  ✅ New best mimic classifier saved (F1 {f1:.3f})")

print("\nTraining complete. Best model saved to models/mimic_classifier_best.pt")