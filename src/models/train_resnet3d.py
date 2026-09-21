from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from resnet3d_model import ResNet3D

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = PROJECT_ROOT / "data" / "patches"
METADATA_FILE = PATCH_DIR / "patch_metadata.csv"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

MODEL_DIR = PROJECT_ROOT / "models" / "resnet3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet3d"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("3D RESNET CMB CLASSIFIER")
print("=" * 70)

print("Device:", DEVICE)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

class CMBPatchDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        patch_path = PATCH_DIR / row["subject"] / row["file"]

        patch = np.load(patch_path).astype(np.float32)

        # Expected shape:
        # (16, 16, 8)

        patch = torch.from_numpy(patch)

        # Add channel dimension:
        # (1, 16, 16, 8)

        patch = patch.unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return patch, label
    
print("\nLoading metadata...")

metadata = pd.read_csv(METADATA_FILE)

print("Total patches:", len(metadata))

print("\nColumns:")
print(metadata.columns.tolist())

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)


train_subjects = set(splits["train"])
val_subjects = set(splits["val"])
test_subjects = set(splits["test"])


print("\nSubject split:")
print("Train subjects:", len(train_subjects))
print("Val subjects  :", len(val_subjects))
print("Test subjects :", len(test_subjects))

train_df = metadata[
    metadata["subject"].isin(train_subjects)
].copy()

val_df = metadata[
    metadata["subject"].isin(val_subjects)
].copy()

test_df = metadata[
    metadata["subject"].isin(test_subjects)
].copy()


print("\nPatch split:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

train_positive = int((train_df["label"] == 1).sum())
train_negative = int((train_df["label"] == 0).sum())

print("\nTraining class distribution:")
print("Positive:", train_positive)
print("Negative:", train_negative)

train_dataset = CMBPatchDataset(train_df)
val_dataset = CMBPatchDataset(val_df)
test_dataset = CMBPatchDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = ResNet3D(
    num_classes=2
).to(DEVICE)

print("\nModel:")
print(model)


total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTotal parameters:", total_parameters)

# Give more importance to the minority CMB class.

class_weights = torch.tensor(
    [
        1.0,
        train_negative / max(train_positive, 1)
    ],
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:", class_weights)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

def calculate_metrics(y_true, y_prob):

    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = tn / max(
        tn + fp,
        1
    )

    try:
        auc = roc_auc_score(
            y_true,
            y_prob
        )
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for patches, labels in loader:

            patches = patches.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(patches)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * patches.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probabilities.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_probs)
    )

    metrics["loss"] = average_loss

    return metrics

best_f1 = -1.0
best_epoch = 0

history = []


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    for patches, labels in train_loader:

        patches = patches.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(patches)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * patches.size(0)
        )

    train_loss = (
        running_loss / len(train_loader.dataset)
    )

    val_metrics = evaluate(
        model,
        val_loader
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_sensitivity": val_metrics["sensitivity"],
        "val_specificity": val_metrics["specificity"],
        "val_f1": val_metrics["f1"],
        "val_auc": val_metrics["auc"]
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Sens: {val_metrics['sensitivity']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val AUC: {val_metrics['auc']:.4f}"
    )

    # Save best model according to validation F1

    if val_metrics["f1"] > best_f1:

        best_f1 = val_metrics["f1"]
        best_epoch = epoch

        model_path = (
            MODEL_DIR /
            "resnet3d_best.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_f1
            },
            model_path
        )

        print(
            f"  -> Best model saved "
            f"(Val F1 = {best_f1:.4f})"
        )

history_df = pd.DataFrame(history)

history_file = (
    OUTPUT_DIR /
    "training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model_path = (
    MODEL_DIR /
    "resnet3d_best.pt"
)

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation F1:",
    checkpoint["best_val_f1"]
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

test_metrics = evaluate(
    model,
    test_loader
)

print(
    f"Accuracy    : {test_metrics['accuracy']:.4f}"
)

print(
    f"Precision   : {test_metrics['precision']:.4f}"
)

print(
    f"Sensitivity : {test_metrics['sensitivity']:.4f}"
)

print(
    f"Specificity : {test_metrics['specificity']:.4f}"
)

print(
    f"F1-Score    : {test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC     : {test_metrics['auc']:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TN = {test_metrics['tn']}"
)

print(
    f"FP = {test_metrics['fp']}"
)

print(
    f"FN = {test_metrics['fn']}"
)

print(
    f"TP = {test_metrics['tp']}"
)

results_file = (
    OUTPUT_DIR /
    "resnet3d_results.json"
)

with open(results_file, "w") as f:

    json.dump(
        {
            "algorithm": "3D ResNet",
            "best_epoch": best_epoch,
            "best_validation_f1": best_f1,
            "test_metrics": test_metrics
        },
        f,
        indent=4
    )


print("\nResults saved to:")

print(results_file)

print("\nModel saved to:")

print(best_model_path)

print("\n" + "=" * 70)
print("3D RESNET TRAINING COMPLETE")
print("=" * 70)