import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader

from densenet3d_model import DenseNet3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = PROJECT_ROOT / "data" / "patches"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

MODEL_DIR = PROJECT_ROOT / "models" / "densenet3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "densenet3d"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


class CMBPatchDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch = np.load(PATCH_DIR / row.subject / row.file).astype(np.float32)
        patch = torch.tensor(patch).unsqueeze(0)
        label = torch.tensor(int(row.label), dtype=torch.long)
        return patch, label


# -----------------------------
# Load metadata
# -----------------------------
print("=" * 70)
print("3D DENSENET CMB CLASSIFIER")
print("=" * 70)
print("Device:", DEVICE)

metadata = pd.read_csv(PATCH_DIR / "patch_metadata.csv")

with open(SPLIT_FILE) as f:
    splits = json.load(f)

train_df = metadata[metadata.subject.isin(splits["train"])]
val_df = metadata[metadata.subject.isin(splits["val"])]
test_df = metadata[metadata.subject.isin(splits["test"])]

print("\nPatch split:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

train_loader = DataLoader(CMBPatchDataset(train_df), batch_size=32, shuffle=True)
val_loader = DataLoader(CMBPatchDataset(val_df), batch_size=32)
test_loader = DataLoader(CMBPatchDataset(test_df), batch_size=32)

# -----------------------------
# Model
# -----------------------------
model = DenseNet3D(num_classes=2).to(DEVICE)

print("\nModel:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print("\nTotal parameters:", total_params)

pos = (train_df.label == 1).sum()
neg = (train_df.label == 0).sum()

weights = torch.tensor([1.0, neg / pos], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nClass weights:", weights)

best_f1 = 0
best_epoch = 0

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(30):

    model.train()
    train_loss = 0

    for patches, labels in train_loader:
        patches = patches.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(patches)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()

    preds = []
    probs = []
    labels_all = []

    val_loss = 0

    with torch.no_grad():
        for patches, labels in val_loader:
            patches = patches.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(patches)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probability = torch.softmax(outputs, dim=1)[:, 1]

            prediction = outputs.argmax(1)

            preds.extend(prediction.cpu().numpy())
            probs.extend(probability.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds)
    prec = precision_score(labels_all, preds, zero_division=0)
    rec = recall_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    auc = roc_auc_score(labels_all, probs)

    print(
        f"Epoch {epoch+1:02d}/30 | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | "
        f"Val Acc: {acc:.4f} | "
        f"Val Sens: {rec:.4f} | "
        f"Val F1: {f1:.4f} | "
        f"Val AUC: {auc:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            MODEL_DIR / "densenet3d_best.pt",
        )

        print(f"  -> Best model saved (Val F1 = {f1:.4f})")

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model.load_state_dict(torch.load(MODEL_DIR / "densenet3d_best.pt"))

model.eval()

preds = []
probs = []
labels_all = []

with torch.no_grad():
    for patches, labels in test_loader:
        patches = patches.to(DEVICE)

        outputs = model(patches)

        probability = torch.softmax(outputs, dim=1)[:, 1]
        prediction = outputs.argmax(1)

        preds.extend(prediction.cpu().numpy())
        probs.extend(probability.cpu().numpy())
        labels_all.extend(labels.numpy())

acc = accuracy_score(labels_all, preds)
prec = precision_score(labels_all, preds)
rec = recall_score(labels_all, preds)
f1 = f1_score(labels_all, preds)
auc = roc_auc_score(labels_all, probs)

tn, fp, fn, tp = confusion_matrix(labels_all, preds).ravel()
spec = tn / (tn + fp)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Accuracy    : {acc:.4f}")
print(f"Precision   : {prec:.4f}")
print(f"Sensitivity : {rec:.4f}")
print(f"Specificity : {spec:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

print("\nConfusion Matrix:")
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

results = {
    "accuracy": float(acc),
    "precision": float(prec),
    "sensitivity": float(rec),
    "specificity": float(spec),
    "f1": float(f1),
    "roc_auc": float(auc),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "best_epoch": best_epoch,
}

with open(OUTPUT_DIR / "densenet3d_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print(OUTPUT_DIR / "densenet3d_results.json")

print("\nModel saved to:")
print(MODEL_DIR / "densenet3d_best.pt")

print("\n" + "=" * 70)
print("3D DENSENET TRAINING COMPLETE")
print("=" * 70)