import torch
import torch.nn as nn

class AsymmetricConvBlock(nn.Module):
    """非对称卷积残差块（仅在子载波维度卷积）"""

    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                      stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 2), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out = out + identity
        out = self.activation(out)
        return out


class ConvBlock1(nn.Module):
    """第一个卷积块（不下采样）"""

    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out = out + identity
        out = self.activation(out)
        return out