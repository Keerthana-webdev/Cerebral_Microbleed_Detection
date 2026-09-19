"""
Lesson 13b: Train the 3-class mimic classifier.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR
from model_3class import SimpleCNN3D_3Class
from mimic_dataset_3class import MimicDataset3Class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
CLASS_NAMES = {0: "True CMB", 1: "Vessel-like", 2: "Other mimic/normal"}

full_metadata = pd.read_csv(PATCHES_DIR / "mimic_classifier_3class_metadata.csv")

train_df, val_df = train_test_split(
    full_metadata, test_size=0.15, stratify=full_metadata["label_3class"], random_state=42
)
print(f"Internal train: {len(train_df)} | Internal val: {len(val_df)}")

train_loader = DataLoader(MimicDataset3Class(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MimicDataset3Class(val_df), batch_size=BATCH_SIZE, shuffle=False)

model = SimpleCNN3D_3Class().to(DEVICE)
loss_function = nn.CrossEntropyLoss()   # handles multi-class directly, no manual weighting needed (balanced data)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for patches, labels in train_loader:
        patches, labels = patches.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(patches)              # shape: (batch, 3)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    per_class_correct = {0: 0, 1: 0, 2: 0}
    per_class_total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for patches, labels in val_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            outputs = model(patches)
            predicted = outputs.argmax(dim=1)   # pick the highest-scoring class

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for c in [0, 1, 2]:
                mask = labels == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += ((predicted == labels) & mask).sum().item()

    accuracy = correct / total

    print(f"Epoch {epoch}/{EPOCHS} | loss: {total_loss/len(train_loader):.4f} | val accuracy: {accuracy:.3f}")
    for c in [0, 1, 2]:
        if per_class_total[c] > 0:
            class_acc = per_class_correct[c] / per_class_total[c]
            print(f"    {CLASS_NAMES[c]}: {class_acc:.1%} ({per_class_correct[c]}/{per_class_total[c]})")

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        torch.save(model.state_dict(), MODELS_DIR / "mimic_classifier_3class_best.pt")
        print(f"  ✅ New best 3-class model saved (accuracy {accuracy:.3f})")

print(f"\nTraining complete. Best model saved to models/mimic_classifier_3class_best.pt")