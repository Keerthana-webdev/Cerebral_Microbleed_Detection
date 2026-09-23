import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "unet_patches"

MODEL_DIR = PROJECT_ROOT / "models" / "unet3d"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet3d"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet3d_model import UNet3D


# ================================================================
# SETTINGS
# ================================================================

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BEST_MODEL = MODEL_DIR / "unet3d_real_best.pt"
RESULT_FILE = OUTPUT_DIR / "unet3d_real_results.json"


# ================================================================
# DATASET
# ================================================================

class RealCMBPatchDataset(Dataset):

    def __init__(self, split):

        self.split_dir = DATA_DIR / split

        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"

        metadata_file = self.split_dir / "metadata.csv"

        self.metadata = pd.read_csv(
            metadata_file
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        image_path = (
            self.images_dir /
            row["image_file"]
        )

        mask_path = (
            self.masks_dir /
            row["mask_file"]
        )

        image = np.load(
            image_path
        ).astype(np.float32)

        mask = np.load(
            mask_path
        ).astype(np.float32)

        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


# ================================================================
# LOSS FUNCTIONS
# ================================================================

dice_smooth = 1e-6


def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.reshape(
        probabilities.size(0),
        -1
    )

    targets = targets.reshape(
        targets.size(0),
        -1
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


def dice_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.size(0),
        -1
    )

    targets = targets.reshape(
        targets.size(0),
        -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + dice_smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + dice_smooth
    )

    return dice.mean().item()


def iou_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.size(0),
        -1
    )

    targets = targets.reshape(
        targets.size(0),
        -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + dice_smooth
    ) / (
        union + dice_smooth
    )

    return iou.mean().item()


# ================================================================
# COMBINED LOSS
# ================================================================

# Positive CMB voxels occupy only a small fraction of each patch.
# BCE alone can therefore strongly favor background.
#
# We use a weighted BCE + Dice combination.

def combined_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    positive_voxels = targets.sum()
    total_voxels = targets.numel()

    positive_fraction = (
        positive_voxels / max(total_voxels, 1)
    )

    positive_fraction = torch.clamp(
        positive_fraction,
        min=0.001,
        max=0.5
    )

    pos_weight = (
        (1.0 - positive_fraction)
        / positive_fraction
    )

    pos_weight = torch.clamp(
        pos_weight,
        min=1.0,
        max=20.0
    )

    weight_tensor = torch.ones_like(targets)

    weight_tensor = torch.where(
        targets > 0,
        pos_weight,
        weight_tensor
    )

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weight_tensor
    )

    dloss = dice_loss(
        logits,
        targets
    )

    return bce + dloss


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 80)
print("3D U-NET — REAL CMB MASK TRAINING")
print("=" * 80)

print(f"Device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")
print()

train_dataset = RealCMBPatchDataset(
    "train"
)

val_dataset = RealCMBPatchDataset(
    "val"
)

test_dataset = RealCMBPatchDataset(
    "test"
)

print(
    f"Training patches:   {len(train_dataset)}"
)

print(
    f"Validation patches: {len(val_dataset)}"
)

print(
    f"Test patches:       {len(test_dataset)}"
)


# ================================================================
# DATA LOADERS
# ================================================================

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


# ================================================================
# MODEL
# ================================================================

model = UNet3D(
    in_channels=1,
    out_channels=1
).to(DEVICE)

print()
print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

def evaluate(model, loader):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    total_samples = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_dice += (
                dice_score(
                    outputs,
                    masks
                )
                * batch_size
            )

            total_iou += (
                iou_score(
                    outputs,
                    masks
                )
                * batch_size
            )

            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples
    )

best_val_dice = -1.0

history = []

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    samples_seen = 0

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

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        samples_seen += batch_size

    train_loss = (
        running_loss
        / samples_seen
    )

    val_loss, val_dice, val_iou = evaluate(
        model,
        val_loader
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou
        }
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Val IoU: {val_iou:.4f}"
    )

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_dice":
                    val_dice,

                "val_iou":
                    val_iou
            },
            BEST_MODEL
        )

        print(
            f"  -> Best model saved "
            f"(Dice: {val_dice:.4f})"
        )

checkpoint = torch.load(
    BEST_MODEL,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_dice, test_iou = evaluate(
    model,
    test_loader
)

results = {

    "algorithm":
        "3D U-Net",

    "dataset":
        "VALDO real CMB masks",

    "device":
        str(DEVICE),

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "train_patches":
        len(train_dataset),

    "validation_patches":
        len(val_dataset),

    "test_patches":
        len(test_dataset),

    "best_validation_dice":
        best_val_dice,

    "test_loss":
        test_loss,

    "test_dice":
        test_dice,

    "test_iou":
        test_iou,

    "best_model":
        str(BEST_MODEL),

    "history":
        history
}

with open(
    RESULT_FILE,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print()
print("=" * 80)
print("REAL-MASK U-NET TRAINING COMPLETE")
print("=" * 80)

print(
    f"Best validation Dice: "
    f"{best_val_dice:.4f}"
)

print(
    f"Test Dice: "
    f"{test_dice:.4f}"
)

print(
    f"Test IoU: "
    f"{test_iou:.4f}"
)

print()
print(
    f"Best model: {BEST_MODEL}"
)

print(
    f"Results: {RESULT_FILE}"
)

print("=" * 80)