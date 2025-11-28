import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMAdapter(nn.Module):
    """
    The 'MIM-Adapter' Block.
    Paper contribution: Enhances the CNN backbone with global context awareness
    to facilitate reconstruction of masked regions.

    It uses a simplified Spatial Attention mechanism to allow visible patches
    to communicate with distant masked regions.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Global Context Excitation
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        # Spatial Refinement (Dilated Conv to see further)
        self.spatial = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Channel Attention
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Spatial Attention
        z = self.spatial(x)

        return (x * y) + z  # Residual connection


class MIMHead(nn.Module):
    """
    The Reconstruction Decoder.
    Takes the feature map from the backbone and upsamples it
    to reconstruct the original image pixels.
    """

    def __init__(self, in_channels=512, hidden_dim=256):
        super().__init__()

        # We use a lightweight decoder (PixelShuffle is efficient)
        self.decoder = nn.Sequential(
            # Layer 1: Upsample
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Layer 2: Upsample
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Layer 3: Upsample
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Layer 4: Final Projection to RGB (3 channels)
            nn.Conv2d(hidden_dim // 4, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output pixels in [0, 1] range
        )

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    # Quick test to ensure shapes match
    dummy_input = torch.randn(2, 512, 20, 20)  # Simulate P5 features
    adapter = MIMAdapter(512)
    head = MIMHead(512)

    feat = adapter(dummy_input)
    print(f"Adapter Output: {feat.shape}")

    img = head(feat)
    print(f"Reconstruction Output: {img.shape} (Should be 20*8 = 160px)")