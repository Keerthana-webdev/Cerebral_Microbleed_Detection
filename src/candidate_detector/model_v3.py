"""
Lesson 19: A deeper 3D CNN — 3 conv blocks instead of 2. More capacity to
separate subtle whole-volume mimics from real lesions, while still small
enough to train on CPU.
"""

import torch.nn as nn


class DeeperCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),          # 16x16x8 -> 8x8x4

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),          # 8x8x4 -> 4x4x2

            nn.Conv3d(32, 64, kernel_size=3, padding=1),   # NEW third block
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # no pooling here - 4x4x2 is already small, pooling further would lose too much
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 2, 128),   # wider too, since we have more feature channels now
            nn.ReLU(),
            nn.Dropout(0.4),                    # slightly higher dropout - more params, more overfitting risk on small data
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x