"""
Lesson 13b: Dataset wrapper for the 3-class mimic data.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR


class MimicDataset3Class(Dataset):
    def __init__(self, dataframe):
        self.metadata = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
        patch = np.expand_dims(patch, axis=0)
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        # CrossEntropyLoss expects a plain integer class index, not a float
        label_tensor = torch.tensor(row["label_3class"], dtype=torch.long)
        return patch_tensor, label_tensor