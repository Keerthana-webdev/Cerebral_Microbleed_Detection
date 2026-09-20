"""
Lesson 17b: Retrain Stage 1 on curated patches + whole-volume hard
negatives. Same architecture, better, more realistic training data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset_v2 import CombinedPatchDataset
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
print("Training on:", DEVICE)

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

train_dataset = CombinedPatchDataset("train")
val_dataset = CombinedPatchDataset("val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

n_pos = (train_dataset.metadata["label"] == 1).sum()
n_neg = (train_dataset.metadata["label"] == 0).sum()
pos_weight_value = n_neg / n_pos
print(f"Positive class weight: {pos_weight_value:.2f}")

model = SimpleCNN3D().to(DEVICE)
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
best_val_sensitivity = 0.0

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
    tp = fn = tn = fp = 0
    with torch.no_grad():
        for patches, labels in val_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            preds = (torch.sigmoid(model(patches).squeeze(1)) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    print(f"Epoch {epoch}/{EPOCHS} | loss: {total_loss/len(train_loader):.4f} | "
          f"val sensitivity: {sensitivity:.3f} | val specificity: {specificity:.3f}")

    # Require both good sensitivity AND meaningfully improved specificity this time
    if sensitivity >= 0.90 and specificity > best_val_sensitivity:
        best_val_sensitivity = specificity
        torch.save(model.state_dict(), MODELS_DIR / "candidate_detector_v2_best.pt")
        print(f"  ✅ New best v2 model saved (sensitivity {sensitivity:.3f}, specificity {specificity:.3f})")

print("\nTraining complete. Best model saved to models/candidate_detector_v2_best.pt")
print("NOTE: original model kept as candidate_detector_best.pt — v2 is saved separately for comparison.")