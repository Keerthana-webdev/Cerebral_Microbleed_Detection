"""
Lesson 30b: Train a mimic classifier specifically on U-Net's candidate
distribution, fixing the domain mismatch found in Lesson 29 (where the
original mimic classifier, trained on CNN-style candidates, barely
filtered U-Net's candidates).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from model import SimpleCNN3D

DEVICE = torch.device("cpu")
PATCHES_DIR = PROJECT_ROOT / "data" / "patches" / "unet_matched_mimic_data"
MODELS_DIR = PROJECT_ROOT / "models"

EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


class UNetMatchedDataset(Dataset):
    def __init__(self):
        records = []
        for subj_dir in PATCHES_DIR.iterdir():
            if not subj_dir.is_dir():
                continue
            for f in subj_dir.glob("pos_*.npy"):
                records.append({"path": f, "label": 1})
            for f in subj_dir.glob("neg_*.npy"):
                records.append({"path": f, "label": 0})
        self.records = pd.DataFrame(records)
        print(f"U-Net-matched dataset: {len(self.records)} patches "
              f"({(self.records.label==1).sum()} positive, {(self.records.label==0).sum()} negative)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        patch = np.load(row["path"])
        patch = np.expand_dims(patch, axis=0)
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(row["label"], dtype=torch.float32)


dataset = UNetMatchedDataset()

# Simple 85/15 internal split for monitoring (this is training data, not touching val/test splits)
n_val = int(len(dataset) * 0.15)
n_train = len(dataset) - n_val
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

n_pos = (dataset.records["label"] == 1).sum()
n_neg = (dataset.records["label"] == 0).sum()
pos_weight_value = n_neg / n_pos
print(f"Positive class weight: {pos_weight_value:.2f}")

model = SimpleCNN3D().to(DEVICE)
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0.0

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

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Epoch {epoch}/{EPOCHS} | loss: {total_loss/len(train_loader):.4f} | "
          f"precision: {precision:.3f} | recall: {recall:.3f} | F1: {f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), MODELS_DIR / "mimic_classifier_unet_matched_best.pt")
        print(f"  ✅ New best U-Net-matched mimic classifier saved (F1 {f1:.3f})")

print(f"\nTraining complete. Best model saved to models/mimic_classifier_unet_matched_best.pt")