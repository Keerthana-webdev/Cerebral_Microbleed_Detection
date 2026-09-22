import torch
import torch.nn as nn


class DenseLayer3D(nn.Module):
    """
    One 3D DenseNet layer.

    Each layer receives all previous feature maps
    and adds new feature maps to them.
    """

    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv3d(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

    def forward(self, x):

        new_features = self.conv(
            self.relu(
                self.norm(x)
            )
        )

        return torch.cat(
            [x, new_features],
            dim=1
        )


class DenseBlock3D(nn.Module):
    """
    Stack of DenseLayer3D blocks.
    """

    def __init__(
        self,
        in_channels,
        num_layers,
        growth_rate
    ):
        super().__init__()

        layers = []

        channels = in_channels

        for _ in range(num_layers):

            layers.append(
                DenseLayer3D(
                    channels,
                    growth_rate
                )
            )

            channels += growth_rate

        self.block = nn.Sequential(*layers)

        self.out_channels = channels

    def forward(self, x):

        return self.block(x)


class Transition3D(nn.Module):
    """
    Reduces the number of feature channels
    and spatial dimensions between dense blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.norm = nn.BatchNorm3d(
            in_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.pool = nn.AvgPool3d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        x = self.conv(
            self.relu(
                self.norm(x)
            )
        )

        x = self.pool(x)

        return x


class DenseNet3D(nn.Module):
    """
    Lightweight 3D DenseNet for CMB patch classification.

    Input:
        [batch, 1, 16, 16, 8]

    Output:
        [batch, 2]

    Classes:
        0 = Non-CMB
        1 = CMB
    """

    def __init__(
        self,
        num_classes=2,
        growth_rate=8
    ):
        super().__init__()

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Dense Block 1
        # ----------------------------------------------------

        self.block1 = DenseBlock3D(
            in_channels=16,
            num_layers=3,
            growth_rate=growth_rate
        )

        channels1 = self.block1.out_channels

        # 16 + 3*8 = 40

        self.transition1 = Transition3D(
            channels1,
            32
        )

        # ----------------------------------------------------
        # Dense Block 2
        # ----------------------------------------------------

        self.block2 = DenseBlock3D(
            in_channels=32,
            num_layers=3,
            growth_rate=growth_rate
        )

        channels2 = self.block2.out_channels

        # 32 + 3*8 = 56

        self.transition2 = Transition3D(
            channels2,
            48
        )

        # ----------------------------------------------------
        # Dense Block 3
        # ----------------------------------------------------

        self.block3 = DenseBlock3D(
            in_channels=48,
            num_layers=3,
            growth_rate=growth_rate
        )

        channels3 = self.block3.out_channels

        # 48 + 3*8 = 72

        # ----------------------------------------------------
        # Final classification
        # ----------------------------------------------------

        self.norm = nn.BatchNorm3d(
            channels3
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.global_pool = nn.AdaptiveAvgPool3d(
            output_size=1
        )

        self.classifier = nn.Linear(
            channels3,
            num_classes
        )

    def forward(self, x):

        # Stem
        x = self.stem(x)

        # Dense block 1
        x = self.block1(x)

        # Transition 1
        x = self.transition1(x)

        # Dense block 2
        x = self.block2(x)

        # Transition 2
        x = self.transition2(x)

        # Dense block 3
        x = self.block3(x)

        # Final normalization
        x = self.relu(
            self.norm(x)
        )

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Classification
        x = self.classifier(x)

        return x


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    model = DenseNet3D(
        num_classes=2,
        growth_rate=8
    )

    dummy_input = torch.randn(
        2,
        1,
        16,
        16,
        8
    )

    output = model(dummy_input)

    print("=" * 70)
    print("3D DENSENET MODEL TEST")
    print("=" * 70)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters    :", total_params)
    print("Trainable parameters:", trainable_params)

    print()
    print("Model created successfully!")