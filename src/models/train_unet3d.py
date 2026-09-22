import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = PROJECT_ROOT / "data" / "patches"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "splits.json"

MODEL_DIR = PROJECT_ROOT / "models" / "unet3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet3d"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet3d_model import UNet3D


# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL = MODEL_DIR / "unet3d_best.pt"
RESULT_FILE = OUTPUT_DIR / "unet3d_results.json"


# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------

class CMBPatchDataset(Dataset):

    def __init__(self, metadata, patch_dir):

        self.metadata = metadata.reset_index(drop=True)
        self.patch_dir = patch_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        subject = row["subject"]
        filename = row["file"]

        patch_path = self.patch_dir / subject / filename

        patch = np.load(patch_path).astype(np.float32)

        # ----------------------------------------------------------
        # Input
        # ----------------------------------------------------------

        image = torch.from_numpy(patch).unsqueeze(0)

        # ----------------------------------------------------------
        # Create segmentation target
        #
        # The existing patch generator stores positive/negative
        # patches. Therefore positive patches are used as the
        # foreground target for this first U-Net experiment.
        # ----------------------------------------------------------

        label = int(row["label"])

        mask = np.zeros_like(patch, dtype=np.float32)

        if label == 1:
            mask[patch > np.percentile(patch, 99)] = 1.0

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# ------------------------------------------------------------------
# LOAD SPLITS
# ------------------------------------------------------------------

print("=" * 80)
print("3D U-NET TRAINING")
print("=" * 80)

print(f"Device: {DEVICE}")
print(f"Patch directory: {PATCH_DIR}")
print(f"Split file: {SPLIT_FILE}")

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

metadata_path = PATCH_DIR / "patch_metadata.csv"

metadata = pd.read_csv(metadata_path)

print()
print(f"Total patches: {len(metadata)}")


def get_subject_list(split_name):

    subjects = splits[split_name]

    # Support both:
    # {"train": [...]}
    # and
    # {"train": {"subjects": [...]}}
    if isinstance(subjects, dict):
        subjects = subjects.get("subjects", [])

    return set(subjects)


train_subjects = get_subject_list("train")
val_subjects = get_subject_list("val")


train_df = metadata[
    metadata["subject"].isin(train_subjects)
].copy()

val_df = metadata[
    metadata["subject"].isin(val_subjects)
].copy()


print(f"Training subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(val_subjects)}")

print(f"Training patches: {len(train_df)}")
print(f"Validation patches: {len(val_df)}")


# ------------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------------

train_dataset = CMBPatchDataset(
    train_df,
    PATCH_DIR
)

val_dataset = CMBPatchDataset(
    val_df,
    PATCH_DIR
)

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


# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------

model = UNet3D(
    in_channels=1,
    out_channels=1
).to(DEVICE)

print()
print("Model:")
print(model)

print()
print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ------------------------------------------------------------------
# LOSS
# ------------------------------------------------------------------

bce_loss = nn.BCEWithLogitsLoss()

dice_smooth = 1e-6


def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.reshape(
        probabilities.size(0), -1
    )

    targets = targets.reshape(
        targets.size(0), -1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + dice_smooth
    ) / (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
        + dice_smooth
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets):

    return (
        bce_loss(logits, targets)
        + dice_loss(logits, targets)
    )


# ------------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------

def calculate_dice(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.size(0), -1
    )

    targets = targets.reshape(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1
    )

    dice = (
        2.0 * intersection + dice_smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + dice_smooth
    )

    return dice.mean().item()


# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------

best_val_dice = -1.0

history = []

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

    train_loss /= len(train_loader.dataset)

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            val_dice += (
                calculate_dice(
                    outputs,
                    masks
                )
                * images.size(0)
            )

    val_loss /= len(val_loader.dataset)

    val_dice /= len(val_loader.dataset)

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
        }
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # --------------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
            },
            BEST_MODEL
        )

        print(
            f"  -> Best model saved "
            f"(Val Dice: {val_dice:.4f})"
        )

results = {
    "algorithm": "3D U-Net",
    "device": str(DEVICE),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "train_subjects": len(train_subjects),
    "validation_subjects": len(val_subjects),
    "train_patches": len(train_df),
    "validation_patches": len(val_df),
    "best_validation_dice": best_val_dice,
    "best_model": str(BEST_MODEL),
    "history": history,
}

with open(RESULT_FILE, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )


print()
print("=" * 80)
print("U-NET TRAINING COMPLETE")
print("=" * 80)
print(
    f"Best validation Dice: "
    f"{best_val_dice:.4f}"
)
print(f"Best model: {BEST_MODEL}")
print(f"Results: {RESULT_FILE}")
print("=" * 80)