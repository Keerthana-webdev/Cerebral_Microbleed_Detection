"""
Lesson 13b: Same CNN backbone as before, but the final layer now outputs
3 numbers (one score per class) instead of 1.
"""

import torch.nn as nn


class SimpleCNN3D_3Class(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),   # CHANGED: 3 outputs instead of 1 — one score per class
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x   # raw scores ("logits"), one per class — no sigmoid here