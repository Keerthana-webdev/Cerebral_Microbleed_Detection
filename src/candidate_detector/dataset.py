"""
Lesson 7a: A PyTorch "Dataset" — a standard wrapper that knows how to load
one patch + its label at a time. PyTorch's training loop will use this to
pull data in batches automatically.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR, SPLITS_DIR


class PatchDataset(Dataset):
    def __init__(self, split_name):
        # split_name is "train", "val", or "test"
        with open(SPLITS_DIR / "splits.json") as f:
            splits = json.load(f)
        subjects_in_this_split = set(splits[split_name])

        metadata = pd.read_csv(PATCHES_DIR / "patch_metadata.csv")
        # Keep only the rows belonging to subjects in this split
        self.metadata = metadata[metadata["subject"].isin(subjects_in_this_split)].reset_index(drop=True)

        print(f"{split_name}: {len(self.metadata)} patches from {len(subjects_in_this_split)} subjects")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        patch_path = PATCHES_DIR / row["subject"] / row["file"]
        patch = np.load(patch_path)  # shape: (16, 16, 8)

        # PyTorch expects a "channel" dimension first: (1, 16, 16, 8)
        patch = np.expand_dims(patch, axis=0)

        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        label_tensor = torch.tensor(row["label"], dtype=torch.float32)
        return patch_tensor, label_tensor