import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            BasicBlock3D(16, 16),
            BasicBlock3D(16, 16)
        )

        self.layer2 = nn.Sequential(
            BasicBlock3D(16, 32, stride=2),
            BasicBlock3D(32, 32)
        )

        self.layer3 = nn.Sequential(
            BasicBlock3D(32, 64, stride=2),
            BasicBlock3D(64, 64)
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = ResNet3D(num_classes=2)

    dummy = torch.randn(2, 1, 16, 16, 8)

    output = model(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", output.shape)
    print("Model created successfully!")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", total_params)