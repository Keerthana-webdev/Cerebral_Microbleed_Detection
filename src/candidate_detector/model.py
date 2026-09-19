"""
Lesson 7b: The neural network itself. A small 3D CNN — a stack of
"convolution" layers (each one scans the patch looking for simple patterns
like edges/blobs) followed by "fully connected" layers that combine those
patterns into a single yes/no decision.
"""

import torch.nn as nn


class SimpleCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),   # look for basic patterns
            nn.BatchNorm3d(16),                              # keeps training stable
            nn.ReLU(),                                       # adds non-linearity (lets it learn complex shapes)
            nn.MaxPool3d(2),                                  # shrinks the cube, keeps the strongest signals

            nn.Conv3d(16, 32, kernel_size=3, padding=1),    # look for more complex patterns
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # turn the 3D cube of features into one long list of numbers
            nn.Linear(32 * 4 * 4 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),              # randomly ignores some neurons during training - prevents "memorizing"
            nn.Linear(64, 1),             # final single number: confidence this is a microbleed
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x