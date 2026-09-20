"""
Lesson 17b: Dataset that combines the ORIGINAL curated patches with the
NEW whole-volume hard negatives, so Stage 1 learns from both clean
examples and realistic whole-scan mistakes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR, SPLITS_DIR

HARD_NEG_DIR = PATCHES_DIR / "wholevolume_hard_negatives"


class CombinedPatchDataset(Dataset):
    def __init__(self, split_name):
        with open(SPLITS_DIR / "splits.json") as f:
            splits = json.load(f)
        subjects_in_split = set(splits[split_name])

        # Original curated patches (positives + easy negatives)
        original = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
        original = original[original["subject"].isin(subjects_in_split)].copy()
        original["full_path"] = original.apply(
            lambda r: PATCHES_DIR / r["subject"] / r["file"], axis=1
        )
        original = original[["full_path", "label"]]

        # New whole-volume hard negatives — ONLY for train split (they were mined from train subjects)
        extra_records = []
        if split_name == "train" and HARD_NEG_DIR.exists():
            for subj_dir in HARD_NEG_DIR.iterdir():
                if subj_dir.is_dir() and subj_dir.name in subjects_in_split:
                    for patch_file in subj_dir.glob("*.npy"):
                        extra_records.append({"full_path": patch_file, "label": 0})

        extra_df = pd.DataFrame(extra_records)
        self.metadata = pd.concat([original, extra_df], ignore_index=True) if len(extra_df) else original

        n_pos = (self.metadata["label"] == 1).sum()
        n_neg = (self.metadata["label"] == 0).sum()
        print(f"{split_name}: {len(self.metadata)} patches ({n_pos} positive, {n_neg} negative, "
              f"including {len(extra_records)} whole-volume hard negatives)")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        patch = np.load(row["full_path"])
        patch = np.expand_dims(patch, axis=0)
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(row["label"], dtype=torch.float32)