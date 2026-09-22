import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.enc2 = DoubleConv3D(16, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.enc3 = DoubleConv3D(32, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(64, 128)

        # Decoder
        self.up3 = nn.ConvTranspose3d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv3D(128, 64)

        self.up2 = nn.ConvTranspose3d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv3D(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32, 16, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv3D(32, 16)

        # Final segmentation output
        self.final = nn.Conv3d(
            16, out_channels, kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


if __name__ == "__main__":

    print("=" * 70)
    print("3D U-NET MODEL TEST")
    print("=" * 70)

    model = UNet3D()

    x = torch.randn(2, 1, 16, 16, 8)

    print("Input shape :", x.shape)

    with torch.no_grad():
        y = model(x)

    print("Output shape:", y.shape)

    print("Parameters:", sum(p.numel() for p in model.parameters()))

    print("=" * 70)
    print("MODEL CREATED SUCCESSFULLY")
    print("=" * 70)