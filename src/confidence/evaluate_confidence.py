"""
Lesson 11b: Run the full pipeline on the test set again, but this time keep
the raw probability for every prediction and check: do actual mistakes
cluster in the "REVIEW RECOMMENDED" zone? That's what proves confidence
scoring is actually useful, not just decoration.
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
from confidence_scoring import get_confidence_label

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

stage1_model = SimpleCNN3D().to(DEVICE)
stage1_model.load_state_dict(torch.load(MODELS_DIR / "candidate_detector_best.pt", map_location=DEVICE))
stage1_model.eval()

stage2_model = SimpleCNN3D().to(DEVICE)
stage2_model.load_state_dict(torch.load(MODELS_DIR / "mimic_classifier_best.pt", map_location=DEVICE))
stage2_model.eval()

with open(SPLITS_DIR / "splits.json") as f:
    splits = json.load(f)
test_subjects = set(splits["test"])

metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
test_metadata = metadata[metadata["subject"].isin(test_subjects)].reset_index(drop=True)


def predict_prob(model, patch_tensor):
    with torch.no_grad():
        return torch.sigmoid(model(patch_tensor)).item()


results = []

for _, row in test_metadata.iterrows():
    patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
    patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    true_label = row["label"]

    stage1_prob = predict_prob(stage1_model, patch_tensor)

    if stage1_prob > 0.5:
        # This patch made it past Stage 1, so Stage 2 decides the final probability
        final_prob = predict_prob(stage2_model, patch_tensor)
    else:
        # Stage 1 rejected it outright — treat as very low probability
        final_prob = stage1_prob

    predicted_label = 1 if final_prob > 0.5 else 0
    confidence_label = get_confidence_label(final_prob)
    was_correct = (predicted_label == true_label)

    results.append({
        "subject": row["subject"],
        "true_label": true_label,
        "predicted_label": predicted_label,
        "probability": round(final_prob, 3),
        "confidence": confidence_label,
        "correct": was_correct,
    })

df = pd.DataFrame(results)

print("=== Confidence label breakdown ===")
print(df["confidence"].value_counts())
print()

print("=== Accuracy WITHIN each confidence band ===")
for label in df["confidence"].unique():
    subset = df[df["confidence"] == label]
    accuracy = subset["correct"].mean()
    print(f"{label}: {accuracy:.1%} correct ({len(subset)} predictions)")
print()

# The key question: do mistakes concentrate in the "REVIEW RECOMMENDED" zone?
mistakes = df[~df["correct"]]
mistakes_in_review_zone = (mistakes["confidence"] == "REVIEW RECOMMENDED").sum()
total_mistakes = len(mistakes)

print("=== Does confidence flagging actually catch the mistakes? ===")
print(f"Total mistakes on test set: {total_mistakes}")
print(f"Mistakes that WERE flagged as 'REVIEW RECOMMENDED': {mistakes_in_review_zone}")
if total_mistakes > 0:
    print(f"-> {mistakes_in_review_zone/total_mistakes:.1%} of all errors were caught by the confidence flag")

out_path = PATCHES_DIR / "confidence_evaluation_results.csv"
df.to_csv(out_path, index=False)
print(f"\nFull results saved to: {out_path}")