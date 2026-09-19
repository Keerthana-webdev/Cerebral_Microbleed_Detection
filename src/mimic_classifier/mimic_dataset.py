"""
Lesson 9a: Dataset wrapper for the mimic-classifier training data
(the hard negatives + true CMBs we just mined).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
from config import PATCHES_DIR


class MimicDataset(Dataset):
    def __init__(self, dataframe):
        self.metadata = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        patch = np.load(PATCHES_DIR / row["subject"] / row["file"])
        patch = np.expand_dims(patch, axis=0)  # add channel dimension
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        label_tensor = torch.tensor(row["mimic_label"], dtype=torch.float32)
        return patch_tensor, label_tensor