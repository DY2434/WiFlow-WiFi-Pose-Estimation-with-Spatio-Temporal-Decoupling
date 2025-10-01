import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from models.convnet import ConvBlock1, AsymmetricConvBlock
from models.attention import DualAxialAttention

class WiFlowPoseModel(nn.Module):
    """WiFlow姿态估计完整模型"""

    def __init__(self, dropout=0.5):
        super().__init__()

        # 输入维度缩减
        # self.reduce = nn.Linear(540, 480)

        # 时序卷积网络
        self.tcn = TemporalConvNet(
            num_inputs=540,
            num_channels=[540, 360, 240],
            kernel_size=3,
            dropout=dropout
        )

        # 初始卷积块
        self.up = ConvBlock1(1, 8)

        # 残差块序列
        self.residual_blocks = nn.ModuleList()
        in_channels = 8
        out_channels_list = [8, 16, 32, 64]

        for out_channels in out_channels_list:
            self.residual_blocks.append(
                AsymmetricConvBlock(in_channels, out_channels)
            )
            in_channels = out_channels

        # 轴向注意力
        self.attention = DualAxialAttention(
            in_planes=64, out_planes=64, groups=8
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.SiLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((15, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: [batch_size, 540, 20]
        batch_size = x.size(0)

        # 维度缩减
        # x = x.transpose(1, 2)  # [B, 20, 540]
        # x = self.reduce(x)  # [B, 20, 480]
        # x = x.transpose(1, 2)  # [B, 480, 20]

        # TCN编码
        x = self.tcn(x)  # [B, 240, 20]

        # 重塑为2D
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, 20, 240]

        # 卷积编码
        x = self.up(x)
        for block in self.residual_blocks:
            x = block(x)  # [B, 64, 20, 15]

        # 转置以匹配注意力输入
        x = x.permute(0, 1, 3, 2)  # [B, 64, 15, 20]

        # 注意力机制
        x = self.attention(x)

        # 解码
        x = self.decoder(x)  # [B, 2, 15, 20]
        x = self.avg_pool(x).squeeze(-1)  # [B, 2, 15]
        x = x.transpose(1, 2)  # [B, 15, 2]

        return x