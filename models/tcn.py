import torch
import torch.nn as nn
import torch.nn.functional as F

# 因果卷积截断层 (保持原版时序因果性)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class InnerGroupedTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, attention_type='none'):
        super(InnerGroupedTemporalBlock, self).__init__()

        self.groups = 20

        self.conv1_group = nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=self.groups, bias=False)
        self.chomp1 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn1_group = nn.BatchNorm1d(n_inputs)
        self.relu1_group = nn.SiLU(inplace=True)

        self.conv1_pw = nn.Conv1d(n_inputs, n_outputs, 1, bias=False)
        self.bn1_pw = nn.BatchNorm1d(n_outputs)
        self.relu1_pw = nn.SiLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)


        self.conv2_group = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     groups=self.groups, bias=False)
        self.chomp2 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn2_group = nn.BatchNorm1d(n_outputs)
        self.relu2_group = nn.SiLU(inplace=True)

        self.conv2_pw = nn.Conv1d(n_outputs, n_outputs, 1, bias=False)
        self.bn2_pw = nn.BatchNorm1d(n_outputs)
        self.relu2_pw = nn.SiLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        # 残差对齐
        self.downsample = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, 1, bias=False),
            nn.BatchNorm1d(n_outputs)
        ) if n_inputs != n_outputs else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)

        # 第一层前向传播
        out = self.conv1_group(x)
        out = self.chomp1(out)
        out = self.bn1_group(out)
        out = self.relu1_group(out)
        out = self.conv1_pw(out)
        out = self.bn1_pw(out)
        out = self.relu1_pw(out)
        out = self.dropout1(out)

        # 第二层前向传播
        out = self.conv2_group(out)
        out = self.chomp2(out)
        out = self.bn2_group(out)
        out = self.relu2_group(out)
        out = self.conv2_pw(out)
        out = self.bn2_pw(out)
        out = self.relu2_pw(out)
        out = self.dropout2(out)

        return F.silu(out + res)

class TemporalBlock(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, attention_type='none'):
        super(TemporalBlock, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                InnerGroupedTemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size, dropout=dropout, attention_type=attention_type
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
