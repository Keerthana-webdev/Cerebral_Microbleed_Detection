"""
Lesson 8: Find patches our Phase 2 model got WRONG (said "microbleed" on
normal tissue). These become the "mimic" training examples for Phase 3.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "candidate_detector"))
from config import PATCHES_DIR, SPLITS_DIR
from model import SimpleCNN3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Load the Phase 2 model we already trained
model = SimpleCNN3D().to(DEVICE)
model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
model.eval()

# Only look at TRAIN subjects — val/test must stay untouched for honest evaluation later
with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
train_subjects = set(splits["train"])

metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
train_metadata = metadata[metadata["subject"].isin(train_subjects)].reset_index(drop=True)

hard_negatives = []
true_positives = []

with torch.no_grad():
    for _, row in train_metadata.iterrows():
        patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        probability = torch.sigmoid(model(patch_tensor)).item()

        if row["label"] == 0 and probability > 0.5:
            # Truly normal tissue, but the model was fooled -> a mimic
            hard_negatives.append({"subject": row["subject"], "file": row["file"], "mimic_label": 0})
        elif row["label"] == 1:
            # A real microbleed -> keep as the positive class here too
            true_positives.append({"subject": row["subject"], "file": row["file"], "mimic_label": 1})

print(f"Real microbleed patches: {len(true_positives)}")
print(f"Hard negatives (model's false alarms): {len(hard_negatives)}")

# Hard negatives alone are often too few to train on well, so we top them up
# with some "easy" negatives (normal patches the model correctly ignored) —
# a standard mix of hard + easy examples for a more robust classifier.
target_negative_count = max(len(hard_negatives), len(true_positives) * 2)
if len(hard_negatives) < target_negative_count:
    used = {(r["subject"], r["file"]) for r in hard_negatives}
    remaining = train_metadata[
        (train_metadata["label"] == 0)
        & (~train_metadata.apply(lambda r: (r["subject"], r["file"]) in used, axis=1))
    ]
    n_needed = min(target_negative_count - len(hard_negatives), len(remaining))
    extra = remaining.sample(n=n_needed, random_state=42)
    for _, row in extra.iterrows():
        hard_negatives.append({"subject": row["subject"], "file": row["file"], "mimic_label": 0})
    print(f"Added {n_needed} easy negatives to balance the dataset")

combined = pd.DataFrame(true_positives + hard_negatives)
out_path = PATCHES_DIR / "mimic_classifier_metadata.csv"
combined.to_csv(out_path, index=False)

print(f"\nFinal mimic-classifier training set: {len(combined)} patches")
print(combined["mimic_label"].value_counts().rename({1: "true CMB", 0: "mimic/normal"}))
print(f"Saved to: {out_path}")