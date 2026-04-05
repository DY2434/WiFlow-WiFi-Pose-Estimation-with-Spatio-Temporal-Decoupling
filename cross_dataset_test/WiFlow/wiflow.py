import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm import tqdm
import copy
import random
import re
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import psutil
import gc
import cv2
import math
from math import sqrt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import stats
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import weight_norm
from collections import deque
import warnings
import pickle
import yaml
import seaborn
from sklearn.model_selection import train_test_split
from mmfi import make_dataset, make_dataloader  # 确保 mmfi.py 在同目录下
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# ============================== #
# 修复字体设置
# ============================== #
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    print("字体设置成功")
except Exception as e:
    print(f"字体设置错误: {e}")

def safe_text(text):
    """如果需要，将中文字符转换为英文等效字符"""
    try:
        translation = {
            '样本': 'Sample',
            '真实姿态': 'True Pose',
            '预测姿态': 'Predicted Pose',
            '帧': 'Frame',
            '身体部位': 'Body Parts',
            '头部': 'Head',
            '躯干': 'Torso',
            '左臂': 'Left Arm',
            '右臂': 'Right Arm',
            '左腿': 'Left Leg',
            '右腿': 'Right Leg',
            '人体姿态': 'Human Pose',
            '中心点/颈部': 'Neck/Center',
            '胸部中心': 'Chest Center',
            '左肩': 'Left Shoulder',
            '右肩': 'Right Shoulder',
            '左肘': 'Left Elbow',
            '右肘': 'Right Elbow',
            '左手腕': 'Left Wrist',
            '右手腕': 'Right Wrist',
            '骨盆': 'Pelvis',
            '左髋': 'Left Hip',
            '右髋': 'Right Hip',
            '左膝': 'Left Knee',
            '右膝': 'Right Knee',
            '左踝': 'Left Ankle',
            '右踝': 'Right Ankle',
            '左颊': 'Left Cheek',
            '右颊': 'Right Cheek',
            '左耳': 'Left Ear',
            '右耳': 'Right Ear',
            '左脚大拇指': 'Left Foot Thumb',
            '右脚大拇指': 'Right Foot Thumb',
            '右脚小拇指': 'Right Foot Pinky',
            '左脚小拇指': 'Left Foot Pinky',
            '左脚跟': 'Left Heel',
            '右脚跟': 'Right Heel',
            '对比视频进度': 'Comparison Video Progress',
            '视频生成进度': 'Video Generation Progress',
            '开始生成视频': 'Starting Video Generation',
            '视频生成完成': 'Video Generation Complete',
            '对比视频生成完成': 'Comparison Video Complete'
        }

        for cn, en in translation.items():
            if cn in text:
                text = text.replace(cn, en)
        return text
    except:
        return text

# ============================== #
# 全局配置和实用函数
# ============================== #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================== #
# 模型组件
# ============================== #
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Chomp1d(nn.Module):
    """裁剪模块，用于因果卷积"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# ============================== #
# SE注意力机制 (推荐用于TCN)
# ============================== #
class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation Block
    适用于时间序列数据，关注通道重要性
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, channels, time_steps]
        batch_size, channels, _ = x.size()

        # Squeeze: 全局平均池化 [B, C, T] -> [B, C, 1]
        squeeze = self.squeeze(x).view(batch_size, channels)

        # Excitation: 学习通道权重 [B, C] -> [B, C]
        excitation = self.excitation(squeeze).view(batch_size, channels, 1)

        # 重新加权
        return x * excitation


# ============================== #
# CBAM注意力机制 (功能更全面)
# ============================== #
class ChannelAttention1D(nn.Module):
    """1D通道注意力"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.size()

        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))

        # 融合并激活
        out = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1)
        return x * out


class SpatialAttention1D(nn.Module):
    """1D空间(时间)注意力"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算统计量
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, T]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, T]

        # 拼接并卷积
        attention = torch.cat([avg_out, max_out], dim=1)  # [B, 2, T]
        attention = self.conv(attention)  # [B, 1, T]
        attention = self.sigmoid(attention)

        return x * attention


class CBAM1D(nn.Module):
    """1D CBAM: 通道注意力 + 空间注意力"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.channel_attention = ChannelAttention1D(channels, reduction)
        self.spatial_attention = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)  # 先通道注意力
        x = self.spatial_attention(x)  # 再空间注意力
        return x


# ============================== #
# 集成注意力的TCN组件
import torch
import torch.nn as nn
import torch.nn.functional as F


# 因果卷积截断层 (保留你原版的因果特性)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# =====================================================================
# 1. 内层基础块：替换原先笨重的 Conv1d，改为极速 Depthwise + Pointwise
# =====================================================================
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


# =====================================================================
# 核心杀器：ResNeXt 风格的分组融合块
# 完美兼顾了：局部时空相干性 (保 PCK@10) + 1x1全局视野 + 轻量化
# =====================================================================
class InnerGroupedTemporalBlock(nn.Module):
    """
    终极防爆版 ResNeXt 1D 块
    加入了严格的组内 BN 归一化，彻底根除 Eval 阶段的方差崩溃！
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, attention_type='none'):
        super(InnerGroupedTemporalBlock, self).__init__()

        self.groups = 18

        # ==========================================================
        # 第一层大循环：分组提取 -> 归一化镇压 -> 全局打通 -> 归一化
        # ==========================================================
        self.conv1_group = nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=self.groups, bias=False)
        self.chomp1 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn1_group = nn.BatchNorm1d(n_inputs)  # 新增：镇压组内洪流
        self.relu1_group = nn.SiLU(inplace=True)  # 新增：非线性提炼

        self.conv1_pw = nn.Conv1d(n_inputs, n_outputs, 1, bias=False)
        self.bn1_pw = nn.BatchNorm1d(n_outputs)
        self.relu1_pw = nn.SiLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # ==========================================================
        # 第二层大循环：深化时空特征
        # ==========================================================
        self.conv2_group = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     groups=self.groups, bias=False)
        self.chomp2 = Chomp1d(padding) if padding > 0 else nn.Identity()
        self.bn2_group = nn.BatchNorm1d(n_outputs)  # 新增：镇压组内洪流
        self.relu2_group = nn.SiLU(inplace=True)  # 新增：非线性提炼

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

        # 第一层前向传播：极其平滑的流水线
        out = self.conv1_group(x)
        out = self.chomp1(out)
        out = self.bn1_group(out)  # 第一道防波堤
        out = self.relu1_group(out)
        out = self.conv1_pw(out)
        out = self.bn1_pw(out)  # 第二道防波堤
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


# =====================================================================
# 外层封装块：完美对接外部的调用 (一字不改)
# =====================================================================
class TemporalBlock(nn.Module):
    """
    TCN 网络的外壳包装器
    """

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


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ConvBlock1, self).__init__()

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

        self.final_activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out = out + identity
        out = self.final_activation(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
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

        self.final_activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        out = out + identity
        out = self.final_activation(out)
        return out


# 1x1卷积用于改变通道数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义一个一维卷积层，用于进行qkv变换
class qkv_transform(nn.Conv1d):
    """定义qkv变换的Conv1d"""

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
        super(qkv_transform, self).__init__(in_planes, out_planes, kernel_size, stride, padding, bias=bias)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.stride = stride
        self.bias = bias
        self.width = width

        # qkv变换层 - 使用3倍的输出通道
        self.qkv_transform = qkv_transform(in_planes, out_planes * 3, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 3)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes)

        # 如果步长大于1，添加池化层
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 初始化qkv变换权重
        nn.init.normal_(self.qkv_transform.weight.data, 0, math.sqrt(1. / self.in_planes))

    def forward(self, x):
        B, C, H, W = x.shape

        # 根据宽度或高度的设置调整张量维度顺序
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # 进行qkv变换
        qkv = self.bn_qkv(self.qkv_transform(x))  # [N*W, out_planes*3, H]

        # 将qkv分解为q、k、v
        qkv = qkv.reshape(N * W, 3, self.out_planes, H)  # [N*W, 3, out_planes, H]
        qkv = qkv.permute(1, 0, 2, 3)  # [3, N*W, out_planes, H]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [N*W, out_planes, H]

        # 重塑为分组形式
        q = q.reshape(N * W, self.groups, self.group_planes, H)
        k = k.reshape(N * W, self.groups, self.group_planes, H)
        v = v.reshape(N * W, self.groups, self.group_planes, H)

        # 直接计算注意力得分
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # 归一化
        qk = self.bn_similarity(qk)
        # 应用softmax
        similarity = F.softmax(qk, dim=-1)
        # 加权求和得到v的注意力输出
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        # 整理输出维度
        sv = sv.reshape(N * W, self.out_planes, H)
        out = self.bn_output(sv)
        out = out.view(N, W, self.out_planes, H)

        # 恢复原始维度顺序
        if self.width:
            out = out.permute(0, 2, 1, 3)
        else:
            out = out.permute(0, 2, 3, 1)

        # 如果步长大于1，进行池化
        if self.stride > 1:
            out = self.pooling(out)

        return out

class DualAxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=2, stride=1, bias=False):
        super(DualAxialAttention, self).__init__()

        # 水平和垂直方向的轴向注意力
        self.height_axis = AxialAttention(
            in_planes,
            out_planes,
            groups,
            stride=stride,
            bias=bias,
            width=False
        )

        self.width_axis = AxialAttention(
            out_planes,
            out_planes,
            groups,
            stride=stride,
            bias=bias,
            width=True
        )

    def forward(self, x):

        x = self.width_axis(x)

        x = self.height_axis(x)

        return x


class SEBlock2D(nn.Module):
    """
    2D Squeeze-and-Excitation Block
    适用于2D特征图：[batch, channels, height, width]
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock2D, self).__init__()

        # 全局平均池化：[B, C, H, W] -> [B, C, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # SE网络：学习通道重要性
        self.se_net = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        batch_size, channels, height, width = x.size()

        # 1. 全局平均池化
        squeeze = self.global_avg_pool(x)  # [B, C, 1, 1]
        squeeze = squeeze.view(batch_size, channels)  # [B, C]

        # 2. 学习通道权重
        excitation = self.se_net(squeeze)  # [B, C]
        excitation = excitation.view(batch_size, channels, 1, 1)  # [B, C, 1, 1]

        # 3. 重新加权
        output = x * excitation  # 广播乘法

        return output


class CSIPoseEstimationModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(CSIPoseEstimationModel, self).__init__()

        self.tcn = TemporalBlock(
            num_inputs=342,
            num_channels=[342, 306, 288],
            kernel_size=3,
            dropout=dropout,
            attention_type='none'  # 'se', 'cbam', 'none'
        )

        self.tcn_proj = nn.Sequential(
            nn.Conv1d(288, 272, kernel_size=1, bias=False),
            nn.BatchNorm1d(272),
            nn.SiLU(inplace=True)
        )

        self.up = ConvBlock1(1, 8)

        self.residual_blocks = nn.ModuleList()
        in_channels = 8
        out_channels_list = [8, 16, 32, 64]

        for out_channels in out_channels_list:
            self.residual_blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.att = DualAxialAttention(in_planes=64, out_planes=64, groups=8)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )

        # self.avg_pool = nn.AdaptiveMaxPool2d((17, 1))
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
        # 此时的输入 x 形状: [batch_size, 3, 114, 10]
        batch_size = x.size(0)
        time_steps = x.size(3)  # 10

        # ========================================== #
        # 核心转换：展平维度 [B, 3, 114, 10] -> [B, 342, 10]
        # ========================================== #
        x = x.reshape(batch_size, -1, time_steps)

        # ===== encoder =====
        x_ant = self.tcn(x)  # [batch_size, 288, 10】

        x_ant = self.tcn_proj(x_ant)  # [B, 272, 10]

        x_ant = x_ant.transpose(1, 2) # [batch_size, 10, 272]
        x_ant = x_ant.unsqueeze(1)  # [batch_size, 1, 10, 272]

        x_ant = self.up(x_ant)

        for block in self.residual_blocks:
            x_ant = block(x_ant)   # [batch_size, 64, 10, 17]

        x = x_ant.permute(0, 1, 3, 2) # 最终: [batch_size, 64, 17, 10]

        x = self.att(x)

        # ===== decoder =====
        # 直接提取因果卷积最后一步的特征，消除动作残影！
        x = x[..., -1:]

        x = self.final_conv(x)  # [B, 3, 17, 1]
        x = x.squeeze(-1)  # [B, 3, 17]
        x = x.transpose(1, 2)  # [batch_size, 17, 3]

        # 返回关键点坐标 [batch_size, 17, 3]
        return x

class SimplePoseLoss(nn.Module):
    """简化的姿态损失函数 - 只包含位置和骨骼长度损失"""
    def __init__(self,
                 position_weight=1.0,
                 bone_length_weight=0.2,
                 loss_type='smooth_l1'):  # 'mse', 'l1', 'smooth_l1'
        super().__init__()

        self.position_weight = position_weight
        self.bone_length_weight = bone_length_weight
        self.loss_type = loss_type

        self.bone_connections = [
            (0, 7), (7, 8), (8, 9), (9, 10),
            # 脊柱与头部: Bot Torso -> Center Torso -> Upper Torso -> Neck Base -> Center Head
            (0, 1), (1, 2), (2, 3),  # 左腿: Bot Torso -> L.Hip -> L.Knee -> L.Foot
            (0, 4), (4, 5), (5, 6),  # 右腿: Bot Torso -> R.Hip -> R.Knee -> R.Foot
            (9, 14), (14, 15), (15, 16),  # 左臂: Neck Base -> L.Shoulder -> L.Elbow -> L.Hand
            (9, 11), (11, 12), (12, 13)  # 右臂: Neck Base -> R.Shoulder -> R.Elbow -> R.Hand
        ]

    def compute_bone_lengths(self, keypoints):
        """计算骨骼长度"""
        bone_lengths = []
        for start_idx, end_idx in self.bone_connections:
            bone_vec = keypoints[..., end_idx, :] - keypoints[..., start_idx, :]
            bone_length = torch.sqrt(torch.sum(bone_vec ** 2, dim=-1) + 1e-8)
            bone_lengths.append(bone_length)
        return torch.stack(bone_lengths, dim=-1)

    def forward(self, pred, target):
        """
        pred: [batch_size, 17, 3] - 预测的关键点
        target: [batch_size, 17, 3] - 真实的关键点
        """
        batch_size = pred.shape[0]

        # 1. 位置损失 - 使用不同的损失函数
        if self.loss_type == 'mse':
            position_loss = F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            position_loss = F.l1_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            position_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 2. 骨骼长度约束损失
        try:
            pred_bone_lengths = self.compute_bone_lengths(pred)
            target_bone_lengths = self.compute_bone_lengths(target)

            if self.loss_type == 'mse':
                bone_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'l1':
                bone_loss = F.l1_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'smooth_l1':
                bone_loss = F.smooth_l1_loss(pred_bone_lengths, target_bone_lengths, beta=0.05)

        except Exception as e:
            print(f"骨骼长度计算错误: {e}")
            bone_loss = torch.tensor(0.0, device=pred.device)

        # 损失字典
        loss_dict = {
            'position': position_loss.item(),
            'bone': bone_loss.item()
        }

        # 总损失
        total_loss = (self.position_weight * position_loss +
                      self.bone_length_weight * bone_loss)

        return total_loss, loss_dict

# ============================== #
# 评估函数
# ============================== #
# ============================== #
# 评估函数 (修复为根节点对齐标准)
# ============================== #
def percentage_correct_keypoints(pred, target, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], num_kpts_type=None):
    batch_size = pred.shape[0]

    # �� 核心修复：根节点 (Pelvis, 索引0) 对齐！消除全局偏移误差！
    pred_rel = pred - pred[:, 0:1, :]
    target_rel = target - target[:, 0:1, :]

    IDX1, IDX2 = 11, 1
    # 基准长度 (Scale) 使用对齐前的原坐标算，不受影响
    scale = torch.sqrt(torch.sum((target[:, IDX1, :] - target[:, IDX2, :]) ** 2, dim=1))
    scale = torch.clamp(scale, min=1e-5)

    # 距离使用【对齐后】的相对坐标算！
    distances = torch.sqrt(torch.sum((pred_rel - target_rel) ** 2, dim=2))
    normalized_distances = distances / scale.unsqueeze(1)

    pck_results = {}
    for threshold in thresholds:
        correct_keypoints = (normalized_distances <= threshold).float()
        pck_overall = correct_keypoints.mean()
        pck_results[threshold] = pck_overall.item()

    return pck_results


def mean_keypoint_error(pred, target):
    # �� 同样需要根节点对齐！
    pred_rel = pred - pred[:, 0:1, :]
    target_rel = target - target[:, 0:1, :]

    distances = torch.sqrt(torch.sum((pred_rel - target_rel) ** 2, dim=2))
    mean_distance = torch.mean(distances)

    return mean_distance.item()


# ============================== #
# 数据增强函数
# ============================== #
def time_masking(x, mask_ratio=0.3, mask_len_range=(5, 10)):
    """时间掩码增强：随机掩码时间段落并用均值填充"""
    device = x.device
    masked_x = x.clone()
    B, C, T = masked_x.shape

    for i in range(B):
        if torch.rand(1).item() < mask_ratio:
            num_masks = torch.randint(1, 3, (1,)).item()
            for _ in range(num_masks):
                mask_len = torch.randint(mask_len_range[0], mask_len_range[1], (1,)).item()
                start = torch.randint(0, T - mask_len, (1,)).item()
                for c in range(C):
                    mean_val = masked_x[i, c, :].mean()
                    masked_x[i, c, start:start + mask_len] = mean_val

    return masked_x

def add_noise(x, noise_level=0.05):
    """添加随机噪声"""
    device = x.device
    noise = torch.randn_like(x).to(device) * noise_level * torch.std(x)
    return x + noise

def random_scaling(x, scale_range=(0.9, 1.1)):
    """随机缩放信号幅度"""
    device = x.device
    if torch.rand(1).item() < 0.5:
        scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).to(device)
        return x * scale_factor
    return x

# ============================== #
# 姿态可视化相关设置
# ============================== #
SKELETON_CONNECTIONS =[
    (0, 7), (7, 8), (8, 9), (9, 10),   # 脊柱与头部
    (0, 1), (1, 2), (2, 3),            # 左腿
    (0, 4), (4, 5), (5, 6),            # 右腿
    (9, 14), (14, 15), (15, 16),       # 左臂
    (9, 11), (11, 12), (12, 13)        # 右臂
]

KEYPOINT_NAMES = {
    0: "Bot Torso", 1: "L.Hip", 2: "L.Knee", 3: "L.Foot",
    4: "R.Hip", 5: "R.Knee", 6: "R.Foot", 7: "Center Torso",
    8: "Upper Torso", 9: "Neck Base", 10: "Center Head",
    11: "R.Shoulder", 12: "R.Elbow", 13: "R.Hand",
    14: "L.Shoulder", 15: "L.Elbow", 16: "L.Hand"
}

BODY_PART_COLORS = {
    'head': 'magenta',
    'torso': 'red',
    'left_arm': 'orange',
    'right_arm': 'green',
    'left_leg': 'cyan',
    'right_leg': 'blue'
}

KEYPOINT_GROUPS = {
    'head': [9, 10],                     # Neck Base, Center Head
    'torso':[0, 7, 8],                  # Bot, Center, Upper Torso
    'left_arm':[14, 15, 16],            # 左臂
    'right_arm':[11, 12, 13],           # 右臂
    'left_leg':[1, 2, 3],               # 左腿
    'right_leg': [4, 5, 6]               # 右腿
}

CONNECTION_COLORS = {
    (0, 7): BODY_PART_COLORS['torso'], (7, 8): BODY_PART_COLORS['torso'], (8, 9): BODY_PART_COLORS['torso'], (9, 10): BODY_PART_COLORS['head'],
    (0, 1): BODY_PART_COLORS['left_leg'], (1, 2): BODY_PART_COLORS['left_leg'], (2, 3): BODY_PART_COLORS['left_leg'],
    (0, 4): BODY_PART_COLORS['right_leg'], (4, 5): BODY_PART_COLORS['right_leg'], (5, 6): BODY_PART_COLORS['right_leg'],
    (9, 14): BODY_PART_COLORS['left_arm'], (14, 15): BODY_PART_COLORS['left_arm'], (15, 16): BODY_PART_COLORS['left_arm'],
    (9, 11): BODY_PART_COLORS['right_arm'], (11, 12): BODY_PART_COLORS['right_arm'], (12, 13): BODY_PART_COLORS['right_arm']
}

def visualize_pose(keypoints, title="人体姿态", figsize=(10, 12), show_labels=True, show_legend=True):
    """可视化单帧人体姿态"""
    fig, ax = plt.subplots(figsize=figsize)

    for connection in SKELETON_CONNECTIONS:
        start_idx, end_idx = connection
        color = CONNECTION_COLORS.get(connection, 'gray')
        ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                color=color, linewidth=3)

    for part_name, indices in KEYPOINT_GROUPS.items():
        color = BODY_PART_COLORS[part_name]
        part_keypoints = keypoints[indices]
        ax.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                   c=color, s=50, edgecolors='black', label=safe_text(part_name))

    if show_labels:
        for i, (x, y) in enumerate(keypoints):
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center', color='white',
                    bbox=dict(boxstyle="circle,pad=0.1", fc='black', ec='none', alpha=0.7))

    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=safe_text(name))
            for name, color in BODY_PART_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right', title=safe_text("身体部位"))

    ax.set_title(safe_text(title), fontsize=14)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig, ax

def create_pose_animation_opencv(all_keypoints, output_file="pose_animation.mp4", fps=30,
                                 figsize=(800, 960), keypoint_scale=1.0,
                                 show_labels=True, show_legend=True):
    """使用OpenCV和tqdm创建人体姿态动画"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from tqdm import tqdm

    width, height = figsize
    frames = len(all_keypoints)
    reshaped_keypoints = all_keypoints.reshape(frames, -1, 2)

    if keypoint_scale != 1.0:
        reshaped_keypoints *= keypoint_scale

    all_x = reshaped_keypoints[:, :, 0].flatten()
    all_y = reshaped_keypoints[:, :, 1].flatten()

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    if show_legend:
        legend_fig = plt.figure(figsize=(width / 100, 1))
        legend_ax = legend_fig.add_subplot(111)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=safe_text(name))
            for name, color in BODY_PART_COLORS.items()
        ]
        legend_ax.legend(handles=legend_elements, loc='center', ncol=len(BODY_PART_COLORS), title=safe_text("身体部位"))
        legend_ax.axis('off')

        canvas = FigureCanvas(legend_fig)
        canvas.draw()
        legend_arr = np.array(canvas.renderer.buffer_rgba())
        legend_arr = cv2.cvtColor(legend_arr, cv2.COLOR_RGBA2BGR)
        legend_height = legend_arr.shape[0]
        plt.close(legend_fig)
    else:
        legend_arr = None
        legend_height = 0

    total_height = height + legend_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, total_height))

    print(safe_text(f"开始生成视频: {output_file}，共 {frames} 帧"))

    with tqdm(total=frames, desc="生成视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            frame_img = np.ones((total_height, width, 3), dtype=np.uint8) * 255

            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

            keypoints = reshaped_keypoints[frame_idx]

            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                        [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                        color=color, linewidth=3)

            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = keypoints[indices]
                ax.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                           c=color, s=50, edgecolors='black')

            if show_labels:
                for i, (x, y) in enumerate(keypoints):
                    ax.text(x, y, str(i), fontsize=10, ha='center', va='center', color='white',
                            bbox=dict(boxstyle="circle,pad=0.1", fc='black', ec='none', alpha=0.7))

            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)
            ax.set_title(safe_text(f"姿态 - 帧 {frame_idx + 1}/{frames}"), fontsize=14)
            ax.set_aspect('equal')
            ax.axis('off')

            plt.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.draw()
            mat_img = np.array(canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            h, w = mat_img.shape[:2]
            frame_img[:h, :w] = mat_img

            if show_legend and legend_arr is not None:
                lh, lw = legend_arr.shape[:2]
                y_offset = h
                frame_img[y_offset:y_offset + lh, :lw] = legend_arr

            video_writer.write(frame_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(safe_text(f"视频生成完成: {output_file}"))
    return output_file

def create_side_by_side_video_opencv(true_keypoints, pred_keypoints, output_file="comparison.mp4",
                                     keypoint_scale=1.0, fps=30):
    """创建对比视频"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from tqdm import tqdm

    frames = min(len(true_keypoints), len(pred_keypoints))
    true_reshaped = true_keypoints[:frames].reshape(frames, -1, 2)
    pred_reshaped = pred_keypoints[:frames].reshape(frames, -1, 2)

    if keypoint_scale != 1.0:
        true_reshaped *= keypoint_scale
        pred_reshaped *= keypoint_scale

    # 计算全局范围
    all_x = np.concatenate([true_reshaped[:, :, 0].flatten(), pred_reshaped[:, :, 0].flatten()])
    all_y = np.concatenate([true_reshaped[:, :, 1].flatten(), pred_reshaped[:, :, 1].flatten()])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    width, height = 1600, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"开始生成对比视频: {output_file}，共 {frames} 帧")

    with tqdm(total=frames, desc="生成对比视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # 真实姿态
            true_kp = true_reshaped[frame_idx]
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax1.plot([true_kp[start_idx, 0], true_kp[end_idx, 0]],
                        [true_kp[start_idx, 1], true_kp[end_idx, 1]],
                        color=color, linewidth=3)

            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = true_kp[indices]
                ax1.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                           c=color, s=50, edgecolors='black')

            ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            ax1.set_ylim(y_max + y_margin, y_min - y_margin)
            ax1.set_title(f"True Pose - Frame {frame_idx + 1}", fontsize=14)
            ax1.set_aspect('equal')
            ax1.axis('off')

            # 预测姿态
            pred_kp = pred_reshaped[frame_idx]
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax2.plot([pred_kp[start_idx, 0], pred_kp[end_idx, 0]],
                        [pred_kp[start_idx, 1], pred_kp[end_idx, 1]],
                        color=color, linewidth=3)

            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = pred_kp[indices]
                ax2.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                           c=color, s=50, edgecolors='black')

            ax2.set_xlim(x_min - x_margin, x_max + x_margin)
            ax2.set_ylim(y_max + y_margin, y_min - y_margin)
            ax2.set_title(f"Predicted Pose - Frame {frame_idx + 1}", fontsize=14)
            ax2.set_aspect('equal')
            ax2.axis('off')

            plt.tight_layout()

            canvas = FigureCanvas(fig)
            canvas.draw()
            mat_img = np.array(canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            video_writer.write(mat_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(f"对比视频生成完成: {output_file}")
    return output_file

def save_all_predictions(true_keypoints, pred_keypoints, output_file="predictions.csv", keypoint_scale=1000.0):
    """保存所有预测结果与真实值到CSV文件"""
    import pandas as pd
    import numpy as np

    n_samples = min(len(true_keypoints), len(pred_keypoints))

    columns = []
    for i in range(17):
        columns.extend([f"true_kp{i}_x", f"true_kp{i}_y", f"pred_kp{i}_x", f"pred_kp{i}_y"])

    data = []
    for i in range(n_samples):
        row = []
        true_kp = true_keypoints[i].reshape(17, 3) * keypoint_scale
        pred_kp = pred_keypoints[i].reshape(17, 3) * keypoint_scale

        for j in range(17):
            row.extend([true_kp[j, 0], true_kp[j, 1], pred_kp[j, 0], pred_kp[j, 1]])

        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=True, index_label="sample_id")

    print(f"已保存所有预测结果到: {output_file}")
    return output_file


def calculate_keypoint_errors(true_keypoints, pred_keypoints, keypoint_scale=1000.0):
    """计算每个关键点的误差统计信息 - 修复版本"""
    import pandas as pd
    import numpy as np

    n_samples = min(len(true_keypoints), len(pred_keypoints))

    # 修复：使用reshape确保正确的形状转换
    true_kp = np.array(true_keypoints[:n_samples]).reshape(n_samples, 17, 3) * keypoint_scale
    pred_kp = np.array(pred_keypoints[:n_samples]).reshape(n_samples, 17, 3) * keypoint_scale

    distances = np.sqrt(np.sum((true_kp - pred_kp) ** 2, axis=2))

    keypoint_stats = []
    for i in range(17):
        kp_distances = distances[:, i]
        stats = {
            'keypoint_id': i,
            'keypoint_name': KEYPOINT_NAMES.get(i, f"关键点 {i}"),
            'body_part': next((part for part, ids in KEYPOINT_GROUPS.items() if i in ids), "未知"),
            'mean_error': np.mean(kp_distances),
            'median_error': np.median(kp_distances),
            'std_error': np.std(kp_distances),
            'min_error': np.min(kp_distances),
            'max_error': np.max(kp_distances)
        }
        keypoint_stats.append(stats)

    df = pd.DataFrame(keypoint_stats)
    return df

def plot_training_history(history, output_dir="vis_results"):
    """绘制训练历史曲线图"""
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(20, 12))

    # 损失曲线
    plt.subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss', linewidth=2.5, marker='o', markersize=3)
    plt.plot(epochs, history['val_loss'], label='Val Total Loss', linewidth=2.5, marker='s', markersize=3)
    plt.title('Total Loss', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 分解损失
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_position_loss'], label='Position Loss', linewidth=2, marker='o', markersize=2)
    plt.plot(epochs, history['train_bone_loss'], label='Bone Loss', linewidth=2, marker='s', markersize=2)
    plt.title('Loss Components', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # MPE曲线
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_mpe'], label='Train MPE', linewidth=2.5, marker='o', markersize=3)
    plt.plot(epochs, history['val_mpe'], label='Val MPE', linewidth=2.5, marker='s', markersize=3)
    plt.title('Mean Pose Error', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MPE', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # PCK曲线
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_pck'], label='Train PCK@0.2', linewidth=2.5, marker='o', markersize=3)
    plt.plot(epochs, history['val_pck'], label='Val PCK@0.2', linewidth=2.5, marker='s', markersize=3)
    plt.title('PCK@0.2 Accuracy', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PCK@0.2', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 学习率曲线
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['lr'], label='Learning Rate', linewidth=2.5, marker='^', markersize=3, color='green')
    plt.title('Learning Rate', fontsize=15, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 损失比例饼图（最后一个epoch）
    plt.subplot(2, 3, 6)
    if len(history['train_position_loss']) > 0:
        last_losses = [
            history['train_position_loss'][-1],
            history['train_bone_loss'][-1]
        ]
        labels = ['Position', 'Bone']
        colors = ['#ff9999', '#66b3ff']
        plt.pie(last_losses, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Final Loss Composition', fontsize=15, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 已保存训练历史曲线图到: {output_path}")

    # 保存CSV数据
    history_csv_path = os.path.join(output_dir, 'training_history.csv')
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, len(history_df) + 1)
    history_df.to_csv(history_csv_path, index=False)
    print(f"📊 已保存训练历史数据到: {history_csv_path}")

    return output_path

# ============================== #
# 模型训练函数
# ============================== #
def get_gpu_memory_map():
    """获取所有GPU的显存信息"""
    result = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            result[i] = gpu_memory
    return result

def calculate_optimal_batch_size(gpu_id):
    """根据GPU显存计算最优批量大小"""
    if not torch.cuda.is_available():
        return 32

    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3

    if gpu_memory > 40:
        return 64
    elif gpu_memory > 20:
        return 64
    elif gpu_memory > 10:
        return 128
    else:
        return 128

def train_pose_model(train_loader, val_loader, test_loader,
                     batch_size=32, n_epochs=100, patience=5,
                     lr=1e-4, weight_decay=1e-5, keypoint_scale=1000.0,
                     gpu_config='auto', output_dir="vis_results", use_augmentation=False):
    """使用简化损失函数的训练函数"""
    os.makedirs(output_dir, exist_ok=True)

    # GPU配置
    if gpu_config == 'auto':
        gpu_memory_map = get_gpu_memory_map()
        rtx4090_ids = [i for i, mem in gpu_memory_map.items() if mem > 40]
        if rtx4090_ids:
            gpu_ids = rtx4090_ids[:1]
            print(f"自动选择: 使用RTX 4090 (GPU {gpu_ids[0]})")
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
            print(f"自动选择: 使用所有GPU {gpu_ids}")
    else:
        gpu_ids = [int(x) for x in gpu_config.split(',')]

    print(f"使用GPU: {gpu_ids}")
    print(f"数据增强: {'启用' if use_augmentation else '禁用'}")
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    # 批量大小配置
    if torch.cuda.is_available():
        gpu_batch_sizes = [calculate_optimal_batch_size(gpu_id) for gpu_id in gpu_ids]
        physical_batch_size = min(gpu_batch_sizes)
        if len(gpu_ids) == 1 and gpu_ids[0] == 1:
            physical_batch_size = 64
    else:
        physical_batch_size = 64

    gradient_accumulation_steps = max(1, batch_size // (physical_batch_size * len(gpu_ids)))
    effective_batch_size = physical_batch_size * len(gpu_ids) * gradient_accumulation_steps

    print(f"批量配置: 物理批量={physical_batch_size}, GPU数量={len(gpu_ids)}, "
          f"梯度累积={gradient_accumulation_steps}, 有效批量={effective_batch_size}")

    # 初始化模型
    model = CSIPoseEstimationModel(dropout=0.3).to(device)

    # ========== 极简模型统计（只要这几行） ==========
    # 1. 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. 计算FLOPs
    try:
        from thop import profile
        model_copy = copy.deepcopy(model)
        with torch.no_grad():
            flops, _ = profile(model_copy, inputs=(input_tensor,), verbose=False)
        print(f"💻 模型计算量: {flops / 1e6:.2f}M FLOPs")
        del model_copy
    except:
        print("💻 FLOPs计算需要安装: pip install thop")


    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print("使用DataParallel")

    # 混合精度训练
    scaler = GradScaler()

    # 使用新的简化损失函数
    criterion = SimplePoseLoss(
        position_weight=1.0,
        bone_length_weight=0.2,  # 增加骨骼长度权重
        loss_type='smooth_l1'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,  # 增加权重衰减
        betas=(0.9, 0.999)
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=lr / 1000,
        cooldown=1,
        threshold=1e-4
    )

    # 保存训练历史 - 移除时序相关的项
    history = {
        'train_loss': [], 'val_loss': [],
        'train_position_loss': [], 'train_bone_loss': [],
        'train_mpe': [], 'val_mpe': [],
        'train_pck': [], 'val_pck': [],
        'train_pck50': [], 'val_pck50': [],
        'lr': []
    }

    # 早停参数
    # best_val_mpe = float('inf')  # MPE越小越好，所以初始化为无穷大
    best_val_pck = 0.0
    patience_counter = 0
    best_model = None
    best_epoch = 0
    best_val_metrics = {'loss': float('inf'), 'mpe': float('inf'), 'pck': 0.0}

    # ============================== #
    # 新增：断点续训加载逻辑
    # ============================== #
    start_epoch = 0
    checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] 发现断点文件 {checkpoint_path}，正在恢复训练...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 兼容 DataParallel 的模型加载
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])  # 恢复 AMP scaler 状态

        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_pck = checkpoint.get('best_val_pck', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        best_val_metrics = checkpoint.get('best_val_metrics', best_val_metrics)
        best_model = checkpoint.get('best_model_state', None)

        print(f"[INFO] 成功恢复！将从 Epoch {start_epoch + 1} 继续训练...\n")
    else:
        print("\n[INFO] 未找到断点文件，将从头开始训练...")

    # 重新创建数据加载器
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    val_batch_size = physical_batch_size // 2

    train_loader_optimized = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader_optimized = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    print(f"开始训练，共{n_epochs}个epoch...")

    for epoch in range(start_epoch, n_epochs):
        # ====== 训练阶段 ======
        model.train()

        train_total_loss = 0.0
        train_total_position_loss = 0.0
        train_total_bone_loss = 0.0
        train_total_mpe = 0.0
        train_total_pck = 0.0
        train_total_pck50 = 0.0
        train_samples = 0
        optimizer.zero_grad()

        train_loop = tqdm(train_loader_optimized, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        current_step = 0

        for batch_idx, batch_data in enumerate(train_loop):
            try:
                # 提取 WiFi-CSI 并转换为 FloatTensor
                batch_x = batch_data["input_wifi-csi"].float().to(device, non_blocking=True)
                # 直接提取完整的输出，形状为 [batch_size, 17, 3]
                batch_y = batch_data["output"].float().to(device, non_blocking=True)

                # 增强的数据增强
                if use_augmentation and epoch > 0:
                    if torch.rand(1).item() < 0.6:
                        batch_x = time_masking(batch_x.permute(0, 2, 1), mask_ratio=0.3).permute(0, 2, 1)
                    if torch.rand(1).item() < 0.6:
                        batch_x = add_noise(batch_x, noise_level=0.02)
                    if torch.rand(1).item() < 0.5:
                        batch_x = random_scaling(batch_x, scale_range=(0.9, 1.1))

                # 使用混合精度训练
                with autocast():
                    outputs = model(batch_x)
                    loss, loss_dict = criterion(outputs, batch_y)
                    loss = loss / gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                # 计算指标
                with torch.no_grad():
                    mpe = mean_keypoint_error(outputs.detach(), batch_y)
                    pck_results = percentage_correct_keypoints(outputs.detach(), batch_y, thresholds=[0.2, 0.5])
                    pck = pck_results[0.2]
                    pck50 = pck_results[0.5]

                # 累积统计信息
                current_batch_size = batch_y.size(0)
                train_total_loss += (loss.item() * gradient_accumulation_steps) * current_batch_size
                train_total_position_loss += loss_dict['position'] * current_batch_size
                train_total_bone_loss += loss_dict['bone'] * current_batch_size
                train_total_mpe += mpe * current_batch_size
                train_total_pck += pck * current_batch_size
                train_total_pck50 += pck50 * current_batch_size
                train_samples += current_batch_size

                # 更新进度条
                cur_loss = loss.item() * gradient_accumulation_steps
                train_loop.set_postfix(
                    loss=f"{cur_loss:.4f}",
                    mpe=f"{mpe:.4f}",
                    pck20=f"{pck:.4f}",
                    pck50=f"{pck50:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}"
                )

                # 梯度累积
                current_step += 1
                if current_step >= gradient_accumulation_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    current_step = 0

                # 内存清理
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                    print(f"批次 {batch_idx}: 张量大小不匹配，跳过该批次")
                    continue
                else:
                    print(f"训练中遇到错误: {e}")
                    torch.cuda.empty_cache()
                    continue

        # 处理最后一个不完整的累积批次
        if current_step > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

        # 计算训练指标
        if train_samples > 0:
            train_loss = train_total_loss / train_samples
            train_position_loss = train_total_position_loss / train_samples
            train_bone_loss = train_total_bone_loss / train_samples
            train_mpe = train_total_mpe / train_samples
            train_pck = train_total_pck / train_samples
            train_pck50 = train_total_pck50 / train_samples
        else:
            train_loss = float('inf')
            train_position_loss = train_bone_loss = float('inf')
            train_mpe = float('inf')
            train_pck = train_pck50 = 0.0

        # ====== 验证阶段 ======
        torch.cuda.empty_cache()
        model.eval()

        val_total_loss = 0.0
        val_total_mpe = 0.0
        val_total_pck = 0.0
        val_total_pck50 = 0.0
        val_samples = 0

        val_loop = tqdm(val_loader_optimized, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loop):
                try:
                    batch_x = batch_data["input_wifi-csi"].float().to(device, non_blocking=True)
                    batch_y = batch_data["output"].float().to(device, non_blocking=True)

                    outputs = model(batch_x)
                    loss, loss_dict = criterion(outputs, batch_y)

                    mpe = mean_keypoint_error(outputs, batch_y)
                    pck_results = percentage_correct_keypoints(outputs, batch_y, thresholds=[0.2, 0.5])
                    pck = pck_results[0.2]
                    pck50 = pck_results[0.5]

                    current_batch_size = batch_y.size(0)
                    val_total_loss += loss.item() * current_batch_size
                    val_total_mpe += mpe * current_batch_size
                    val_total_pck += pck * current_batch_size
                    val_total_pck50 += pck50 * current_batch_size
                    val_samples += current_batch_size

                    val_loop.set_postfix(
                        loss=f"{loss.item():.4f}",
                        mpe=f"{mpe:.4f}",
                        pck20=f"{pck:.4f}",
                        pck50=f"{pck50:.4f}"
                    )

                except RuntimeError as e:
                    if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                        print(f"验证批次 {batch_idx}: 张量大小不匹配，跳过该批次")
                        continue
                    else:
                        print(f"验证出错: {e}")
                        torch.cuda.empty_cache()
                        continue

        # 计算验证指标
        if val_samples > 0:
            val_loss = val_total_loss / val_samples
            val_mpe = val_total_mpe / val_samples
            val_pck = val_total_pck / val_samples
            val_pck50 = val_total_pck50 / val_samples
        else:
            val_loss = float('inf')
            val_mpe = float('inf')
            val_pck = val_pck50 = 0.0

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_position_loss'].append(train_position_loss)
        history['train_bone_loss'].append(train_bone_loss)
        history['train_mpe'].append(train_mpe)
        history['val_mpe'].append(val_mpe)
        history['train_pck'].append(train_pck)
        history['val_pck'].append(val_pck)
        history['train_pck50'].append(train_pck50)
        history['val_pck50'].append(val_pck50)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 打印详细的损失信息
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  Train - Total: {train_loss:.4f}, Position: {train_position_loss:.4f}, "
              f"Bone: {train_bone_loss:.4f}")
        print(f"  Train - MPE: {train_mpe:.4f}, PCK@0.2: {train_pck:.4f}, PCK@0.5: {train_pck50:.4f}")
        print(f"  Val - Loss: {val_loss:.4f}, MPE: {val_mpe:.4f}, PCK@0.2: {val_pck:.4f}, PCK@0.5: {val_pck50:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 基于验证损失调整学习率
        scheduler.step(val_pck)

        # 早停检查
        if val_pck > best_val_pck:
            best_val_pck = val_pck
            best_val_metrics['pck'] = val_pck
            best_val_metrics['mpe'] = val_mpe
            best_val_metrics['loss'] = val_loss

            if hasattr(model, 'module'):
                best_model = copy.deepcopy(model.module.state_dict())
            else:
                best_model = copy.deepcopy(model.state_dict())

            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(output_dir, "best_pose_model.pth")
            torch.save(best_model, model_path)
            print(
                f"  💾 发现更优模型！保存最佳模型 (Epoch {best_epoch + 1}, PCK@0.2={val_pck:.4f}, MPE={val_mpe:.4f}) 到 {model_path}")
        else:
            patience_counter += 1
            print(f"  验证pck未改善，耐心计数: {patience_counter}/{patience}")

        # ============================== #
        # 新增：每个 Epoch 结束后保存一次断点
        # ============================== #
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # 保存 AMP scaler 状态
            'history': history,
            'best_val_pck': best_val_pck,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'best_val_metrics': best_val_metrics,
            'best_model_state': best_model
        }, checkpoint_path)
        print(f"  [SAVE] 已保存 Epoch {epoch + 1} 的断点数据。")

        if patience_counter >= patience:
            print(f"⏹️ 早停在 {epoch + 1} 个epoch后触发。最佳epoch: {best_epoch + 1}")
            break

    # 加载最佳模型进行测试
    if best_model is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model)
        else:
            model.load_state_dict(best_model)
        print(f"✅ 加载最佳模型，来自 epoch {best_epoch + 1}")

    print(f"🎯 最佳验证指标 - Loss: {best_val_metrics['loss']:.4f}, "
          f"MPE: {best_val_metrics['mpe']:.4f}, PCK@0.2: {best_val_metrics['pck']:.4f}")

    # 保存训练历史图表
    print("📊 正在生成训练历史曲线图...")
    plot_training_history(history, output_dir)

    # ====== 测试阶段 ======
    test_loader_optimized = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    model.eval()

    test_total_loss = 0.0
    test_total_mpe = 0.0
    test_total_pck10 = 0.0
    test_total_pck20 = 0.0
    test_total_pck30 = 0.0
    test_total_pck40 = 0.0
    test_total_pck50 = 0.0
    test_samples = 0
    all_pred_keypoints = []
    all_true_keypoints = []

    test_loop = tqdm(test_loader_optimized, desc="测试中")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loop):
            try:
                batch_x = batch_data["input_wifi-csi"].float().to(device, non_blocking=True)
                batch_y = batch_data["output"].float().to(device, non_blocking=True)

                outputs = model(batch_x)
                loss, _ = criterion(outputs, batch_y)

                mpe = mean_keypoint_error(outputs, batch_y)
                pck_results = percentage_correct_keypoints(outputs, batch_y, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])
                pck10 = pck_results[0.1]
                pck20 = pck_results[0.2]
                pck30 = pck_results[0.3]
                pck40 = pck_results[0.4]
                pck50 = pck_results[0.5]

                current_batch_size = batch_y.size(0)
                test_total_loss += loss.item() * current_batch_size
                test_total_mpe += mpe * current_batch_size
                test_total_pck10 += pck10 * current_batch_size
                test_total_pck20 += pck20 * current_batch_size
                test_total_pck30 += pck30 * current_batch_size
                test_total_pck40 += pck40 * current_batch_size
                test_total_pck50 += pck50 * current_batch_size
                test_samples += current_batch_size

                all_pred_keypoints.append(outputs.cpu().numpy())
                all_true_keypoints.append(batch_y.cpu().numpy())

                test_loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mpe=f"{mpe:.4f}",
                    pck20=f"{pck20:.4f}",
                    pck50=f"{pck50:.4f}"
                )

            except RuntimeError as e:
                if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                    print(f"测试批次 {batch_idx}: 张量大小不匹配，跳过该批次")
                    continue
                else:
                    print(f"测试时出错: {e}")
                    torch.cuda.empty_cache()
                    continue

    # 计算测试指标
    if test_samples > 0:
        test_loss = test_total_loss / test_samples
        test_mpe = test_total_mpe / test_samples
        test_pck10 = test_total_pck10 / test_samples
        test_pck20 = test_total_pck20 / test_samples
        test_pck30 = test_total_pck30 / test_samples
        test_pck40 = test_total_pck40 / test_samples
        test_pck50 = test_total_pck50 / test_samples
    else:
        test_loss = float('inf')
        test_mpe = float('inf')
        test_pck10 = test_pck20 = test_pck30 = test_pck40 = test_pck50 = 0.0

    # 显示所有PCK阈值的测试结果
    print(f"🎯 测试结果:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   MPE: {test_mpe:.4f}")
    print(f"   PCK@0.1: {test_pck10:.4f}")
    print(f"   PCK@0.2: {test_pck20:.4f}")
    print(f"   PCK@0.3: {test_pck30:.4f}")
    print(f"   PCK@0.4: {test_pck40:.4f}")
    print(f"   PCK@0.5: {test_pck50:.4f}")

    # 保存预测结果和生成视频
    if all_pred_keypoints and all_true_keypoints:
        all_preds = np.vstack(all_pred_keypoints)
        all_trues = np.vstack(all_true_keypoints)

        # 保存预测结果
        predictions_file = os.path.join(output_dir, "test_predictions.csv")
        save_all_predictions(all_trues, all_preds, predictions_file, keypoint_scale)
        print(f"💾 已保存测试预测结果到: {predictions_file}")

        # 计算关键点误差统计
        error_stats = calculate_keypoint_errors(
            all_trues[:min(1000, len(all_trues))],
            all_preds[:min(1000, len(all_preds))],
            keypoint_scale=keypoint_scale
        )
        error_stats_file = os.path.join(output_dir, "keypoint_error_stats.csv")
        error_stats.to_csv(error_stats_file)
        print(f"📊 已保存关键点误差统计到: {error_stats_file}")

        # 保存详细的测试结果到CSV
        test_results_file = os.path.join(output_dir, "test_results_summary.csv")
        test_results_data = {
            'Metric': ['Loss', 'MPE', 'PCK@0.1', 'PCK@0.2', 'PCK@0.3', 'PCK@0.4', 'PCK@0.5'],
            'Value': [test_loss, test_mpe, test_pck10, test_pck20, test_pck30, test_pck40, test_pck50]
        }
        import pandas as pd
        test_results_df = pd.DataFrame(test_results_data)
        test_results_df.to_csv(test_results_file, index=False)
        print(f"📊 已保存测试结果汇总到: {test_results_file}")

        # 生成视频
        try:
            videos_dir = os.path.join(output_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)

            frames_to_animate = min(720, len(all_preds))
            print(f"正在为前{frames_to_animate}帧生成视频...")

            # 1. 创建真实姿态视频
            print("正在生成真实姿态视频...")
            true_subset = all_trues[:frames_to_animate].copy()
            true_animation = create_pose_animation_opencv(
                true_subset,
                output_file=os.path.join(videos_dir, "true_poses.mp4"),
                keypoint_scale=keypoint_scale,
                show_labels=True
            )
            print(f"已生成真实姿态视频: {true_animation}")

            # 2. 创建预测姿态视频
            print("正在生成预测姿态视频...")
            pred_subset = all_preds[:frames_to_animate].copy()
            pred_animation = create_pose_animation_opencv(
                pred_subset,
                output_file=os.path.join(videos_dir, "predicted_poses.mp4"),
                keypoint_scale=keypoint_scale,
                show_labels=True
            )
            print(f"已生成预测姿态视频: {pred_animation}")

            # 3. 创建对比视频
            print("正在生成对比视频...")
            comparison_video = create_side_by_side_video_opencv(
                true_subset,
                pred_subset,
                output_file=os.path.join(videos_dir, "comparison_poses.mp4"),
                keypoint_scale=keypoint_scale,
                fps=30
            )
            print(f"已生成对比视频: {comparison_video}")

            print(f"已完成所有视频的生成，保存在 {videos_dir} 目录下")

        except Exception as e:
            print(f"生成视频时出错: {e}")
            import traceback
            traceback.print_exc()

    return model, history, test_loss, test_pck20, test_mpe, {
        'pck10': test_pck10,
        'pck20': test_pck20,
        'pck30': test_pck30,
        'pck40': test_pck40,
        'pck50': test_pck50
    }


def main():
    # 定义统一的输出目录
    output_dir = "wiflowm4"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有结果将保存到目录: {output_dir}")

    # 添加命令行参数
    parser = argparse.ArgumentParser(description='训练CSI姿态估计模型')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU配置: auto(自动选择), 1,2(使用GPU0和GPU2), 1(仅使用GPU1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='逻辑批量大小（梯度累积后的有效批量）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help=f'输出目录 (默认: {output_dir})')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='是否使用数据增强 (默认: 不使用)')
    args = parser.parse_args()

    # 如果用户通过命令行指定了输出目录，则使用用户指定的
    if args.output_dir != output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用用户指定的输出目录: {output_dir}")

    # 设置随机种子确保可重复性
    set_seed(42)

    # 显示系统信息
    print(f"系统内存使用情况: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, 显存: {props.total_memory / 1024 ** 3:.1f}GB")

    # 关键点归一化因子
    keypoint_scale = 1000.0

    # ========================================== #
    # 核心：MMFi 数据集加载逻辑
    # ========================================== #

    # 【注意！】运行前请务必确认这两个路径正确
    dataset_root = "/home/aip/DATA/MMFi_Clean"
    yaml_config_path = "config.yaml"

    try:
        print(f"正在读取 MMFi 配置文件并扫描数据集 (路径: {dataset_root})...")
        with open(yaml_config_path, "r", encoding="utf-8") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)

        # 1. 制作基础 Dataset (此过程可能需要几秒钟扫描文件夹)
        train_dataset, test_dataset = make_dataset(dataset_root, config)

        # 2. 生成 Train Loader
        rng_generator = torch.manual_seed(config["init_rand_seed"])
        train_loader = make_dataloader(
            train_dataset,
            is_training=True,
            generator=rng_generator,
            batch_size=args.batch_size  # 优先使用命令行传入的 batch_size
        )

        # 3. 使用 sklearn 将 test_dataset 一分为二 (50% 验证，50% 测试)
        val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)

        val_loader = make_dataloader(
            val_data, is_training=False, generator=rng_generator, batch_size=args.batch_size // 2
        )
        test_loader = make_dataloader(
            test_data, is_training=False, generator=rng_generator, batch_size=args.batch_size // 2
        )

        print(f"✅ 数据加载成功！")
        print(f"   - 训练集: {len(train_loader.dataset)} 样本")
        print(f"   - 验证集: {len(val_loader.dataset)} 样本")
        print(f"   - 测试集: {len(test_loader.dataset)} 样本")

        # ========================================== #
        # 数据维度检测 (仅在第一轮打印)
        # ========================================== #
        print("\n" + "=" * 50)
        print("🔍 首次数据维度检查 (Dimension Check)")
        for batch_data in train_loader:
            csi_shape = batch_data["input_wifi-csi"].shape
            gt_shape = batch_data["output"].shape
            print(f"📥 [Input] WiFi-CSI 形状: {csi_shape}")
            print(f"🎯 [Output] 真实关键点形状: {gt_shape}")
            print("=" * 50 + "\n")
            break  # 检测完第一个 batch 即退出

    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 训练参数
    n_epochs = args.epochs
    patience = 5
    lr = args.lr
    weight_decay = 1e-5

    # 训练模型
    print(f"开始训练模型...")
    print(f"GPU配置: {args.gpu}")
    print(f"批量大小: {args.batch_size}")
    print(f"训练轮数: {n_epochs}")
    print(f"学习率: {lr}")
    print(f"输出目录: {output_dir}")

    try:
        model, history, test_loss, test_pck, test_mpe, pck_details = train_pose_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            batch_size=args.batch_size,
            n_epochs=n_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            keypoint_scale=keypoint_scale,
            gpu_config=args.gpu,
            output_dir=output_dir,
            use_augmentation=args.use_augmentation
        )

        print(f"训练完成，测试损失: {test_loss:.4f}, 测试PCK@0.2: {test_pck:.4f}")
        print(f"详细PCK结果: {pck_details}")
        print(f"所有结果已保存到: {output_dir}")

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置多进程启动方法
    if torch.cuda.is_available():
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    main()