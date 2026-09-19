"""
Lesson 7c: The actual training loop - where learning happens.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import PatchDataset
from model import SimpleCNN3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

train_dataset = PatchDataset("train")
val_dataset = PatchDataset("val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class weighting: tell the loss function that missing a real positive
# patch matters ~(negatives/positives) times more than a false alarm.
n_pos = train_dataset.metadata["label"].sum()
n_neg = len(train_dataset.metadata) - n_pos
pos_weight_value = n_neg / n_pos
print(f"Positive class weight: {pos_weight_value:.2f}")

model = SimpleCNN3D().to(DEVICE)
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_sensitivity = 0.0
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    # ---- Training pass ----
    model.train()
    total_train_loss = 0.0
    for patches, labels in train_loader:
        patches, labels = patches.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()                       # clear old gradients
        predictions = model(patches).squeeze(1)      # forward pass: model's guess
        loss = loss_function(predictions, labels)    # how wrong was it
        loss.backward()                              # figure out how to adjust weights
        optimizer.step()                             # actually adjust them

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- Validation pass (no learning, just checking) ----
    model.eval()
    true_positives, false_negatives, true_negatives, false_positives = 0, 0, 0, 0
    with torch.no_grad():
        for patches, labels in val_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            predictions = torch.sigmoid(model(patches).squeeze(1))  # convert to 0-1 probability
            predicted_labels = (predictions > 0.5).float()

            true_positives += ((predicted_labels == 1) & (labels == 1)).sum().item()
            false_negatives += ((predicted_labels == 0) & (labels == 1)).sum().item()
            true_negatives += ((predicted_labels == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted_labels == 1) & (labels == 0)).sum().item()

    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)   # % of real microbleeds caught
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)   # % of normal patches correctly ignored

    print(f"Epoch {epoch}/{EPOCHS} | train loss: {avg_train_loss:.4f} | "
          f"val sensitivity: {sensitivity:.3f} | val specificity: {specificity:.3f}")

    if sensitivity > best_val_sensitivity:
        best_val_sensitivity = sensitivity
        torch.save(model.state_dict(), MODELS_DIR / "candidate_detector_best.pt")
        print(f"  ✅ New best model saved (sensitivity {sensitivity:.3f})")

print("\nTraining complete. Best model saved to models/candidate_detector_best.pt")