import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import pickle
import yaml
import seaborn
from sklearn.model_selection import train_test_split
from mmfi import make_dataset, make_dataloader  # 确保 mmfi.py 在同目录下

# ============================== #
# CSI数据格式转换函数
# ============================== #

import torch

def convert_csi_format(csi_data_code2):
    """
    将代码2的CSI数据格式转换为代码1的格式
    输入: [batch_size, 3, 114, 10]  (对应: batch, rx_antennas, subcarriers, time_steps)
    输出: [batch_size, 1140, 1, 3] (对应: batch, time_steps*subcarriers, tx_antennas, rx_antennas)
    """
    batch_size = csi_data_code2.shape[0]

    # 步骤1: 调整维度顺序
    # 输入维度索引: 0:batch, 1:antennas(3), 2:subcarriers(114), 3:time_steps(10)
    # 我们将时间步长和子载波移到中间，将天线移到最后
    # permute 顺序 -> [batch_size, 10, 114, 3]
    csi_reordered = csi_data_code2.permute(0, 3, 2, 1)

    # 步骤2: 重塑为最终格式
    # 融合 时间步长(10) * 子载波(114) = 1140
    # 并在天线前增加一个维度 1 (代表单发射天线 Tx=1)
    csi_final = csi_reordered.contiguous().view(
        batch_size,
        1140,  # 10 * 114
        1,     # tx_antennas
        3      # rx_antennas
    )

    return csi_final


# ============================== #
# ResNet模型（修改输出为2通道的PAM）
# ============================== #

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=1140):
        super(ResNet, self).__init__()

        self.in_channels = 150
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, 150, layers[1], 2)
        self.layer3 = self.make_layer(block, 300, layers[2], 2)
        self.layer4 = self.make_layer(block, 300, layers[3], 2)

        self.decode = nn.Sequential(
            nn.Conv2d(300, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False),  # 输出2通道(x', y')
        )

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels * block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 上采样到对称的分辨率
        x = F.interpolate(x, size=(136, 136), mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.mean(x, dim=-1, keepdim=True)  # [B, 300, 17, 1]

        x = self.decode(x) # [batch_size, 3, 17, 1]
        x = x.squeeze(-1)  # [B, 3, 17]
        x = x.transpose(1, 2)  # [B, 17, 3]

        return x

# ============================== #
# 评估函数
# ============================== #

# 在评估函数之前添加：
def mean_keypoint_error(pred, target, keypoint_scale=1.0):
    """
    计算平均关键点误差 (Mean Pose Error)

    Args:
        pred: 预测的关键点 [batch_size, 17, 3]
        target: 真实的关键点 [batch_size, 17, 3]
        keypoint_scale: 缩放因子

    Returns:
        mpe: 平均关键点误差
    """

    # 计算欧氏距离
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))  # [batch_size, 15]

    # 计算平均误差
    mpe = torch.mean(distances) * keypoint_scale

    return mpe.item()


def calculate_pck_metrics(pred, target, thr):
    """
    计算纯 3D 空间的 PCK (无根节点对齐)
    完全适配原代码的数据流结构，返回包含每个关节点和整体平均值的 numpy 数组
    Args:
        pred: [batch_size, 17, 3] 预测关键点
        target: [batch_size, 17, 3] 真实关键点
        thr: 单一 PCK 阈值 (例如 0.5)
    Returns:
        pck: shape 为 [18,] 的 numpy 数组。前 17 个是各个关节点的 PCK，最后一个(索引17)是整体平均 PCK。
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    assert pred.shape[0] == target.shape[0]
    batch_size = target.shape[0]
    kpts_num = target.shape[1]  # 17 个关键点

    # 1. 计算 3D 空间欧氏距离
    distances = np.sqrt(np.sum(np.square(pred - target), axis=2))  # [batch_size, 17]

    # 2. 计算归一化基准 (以 Left Hip(1) 到 Right Shoulder(11) 的距离为基准)
    IDX1, IDX2 = 1, 11
    scale = np.sqrt(np.sum(np.square(target[:, IDX1, :] - target[:, IDX2, :]), axis=1))

    # 防止除以0
    scale = np.maximum(scale, 1e-5)

    # 3. 归一化距离
    # 将 scale 扩展为 [batch_size, 17] 以便广播相除
    normalized_distances = distances / np.tile(scale, (kpts_num, 1)).T

    # 4. 计算 PCK 分数
    # 创建一个长度为 18 的数组，存放 17个点的分数 + 1个整体平均分
    pck = np.zeros(kpts_num + 1)

    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = np.mean(normalized_distances[:, kpt_idx] <= thr)

    # 最后一个位置存放整个 batch 所有点的平均正确率
    pck[kpts_num] = np.mean(normalized_distances <= thr)

    return pck


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

    # 骨骼连接和颜色
    SKELETON_CONNECTIONS = [
        (0, 1), (1, 8),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (8, 9), (8, 12),
        (9, 10), (10, 11),
        (12, 13), (13, 14)
    ]

    BODY_PART_COLORS = {
        'head': 'magenta',
        'torso': 'red',
        'right_arm': 'orange',  # 右臂
        'left_arm': 'green',  # 左臂
        'right_leg': 'blue',  # 右腿
        'left_leg': 'cyan'  # 左腿
    }

    KEYPOINT_GROUPS = {
        'head': [0],
        'torso': [1, 8],
        'left_arm': [2, 3, 4],
        'right_arm': [5, 6, 7],
        'left_leg': [9, 10, 11],
        'right_leg': [12, 13, 14]
    }

    CONNECTION_COLORS = {
        # 躯干
        (0, 1): BODY_PART_COLORS['torso'],  # Nose to Neck
        (1, 8): BODY_PART_COLORS['torso'],  # Neck to MidHip

        # 右臂（R开头的）
        (1, 2): BODY_PART_COLORS['right_arm'],  # Neck to RShoulder
        (2, 3): BODY_PART_COLORS['right_arm'],  # RShoulder to RElbow
        (3, 4): BODY_PART_COLORS['right_arm'],  # RElbow to RWrist

        # 左臂（L开头的）
        (1, 5): BODY_PART_COLORS['left_arm'],  # Neck to LShoulder
        (5, 6): BODY_PART_COLORS['left_arm'],  # LShoulder to LElbow
        (6, 7): BODY_PART_COLORS['left_arm'],  # LElbow to LWrist

        # 髋部连接
        (8, 9): BODY_PART_COLORS['right_leg'],  # MidHip to RHip
        (8, 12): BODY_PART_COLORS['left_leg'],  # MidHip to LHip

        # 右腿
        (9, 10): BODY_PART_COLORS['right_leg'],  # RHip to RKnee
        (10, 11): BODY_PART_COLORS['right_leg'],  # RKnee to RAnkle

        # 左腿
        (12, 13): BODY_PART_COLORS['left_leg'],  # LHip to LKnee
        (13, 14): BODY_PART_COLORS['left_leg']  # LKnee to LAnkle
    }

    if show_legend:
        legend_fig = plt.figure(figsize=(width / 100, 1))
        legend_ax = legend_fig.add_subplot(111)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=name)
            for name, color in BODY_PART_COLORS.items()
        ]
        legend_ax.legend(handles=legend_elements, loc='center', ncol=len(BODY_PART_COLORS), title="Body Parts")
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

    print(f"开始生成视频: {output_file}，共 {frames} 帧")

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
            ax.set_title(f"Pose - Frame {frame_idx + 1}/{frames}", fontsize=14)
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
    print(f"视频生成完成: {output_file}")
    return output_file


def create_side_by_side_video_opencv(true_keypoints, pred_keypoints, output_file="comparison.mp4",
                                     keypoint_scale=1000.0, fps=30):
    """创建对比视频"""
    frames = min(len(true_keypoints), len(pred_keypoints))

    # 转换为numpy数组
    if torch.is_tensor(true_keypoints):
        true_keypoints = true_keypoints.cpu().numpy()
    if torch.is_tensor(pred_keypoints):
        pred_keypoints = pred_keypoints.cpu().numpy()

    # 缩放关键点
    true_keypoints = true_keypoints[:frames] * keypoint_scale
    pred_keypoints = pred_keypoints[:frames] * keypoint_scale

    # 计算全局范围
    all_x = np.concatenate([true_keypoints[:, :, 0].flatten(), pred_keypoints[:, :, 0].flatten()])
    all_y = np.concatenate([true_keypoints[:, :, 1].flatten(), pred_keypoints[:, :, 1].flatten()])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    width, height = 1600, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"生成对比视频: {output_file}，共 {frames} 帧")

    with tqdm(total=frames, desc="生成对比视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # 真实姿态
            true_kp = true_keypoints[frame_idx]
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
            pred_kp = pred_keypoints[frame_idx]
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

            # 转换为OpenCV格式
            fig.canvas.draw()
            mat_img = np.array(fig.canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            video_writer.write(mat_img)
            plt.close(fig)
            pbar.update(1)

    video_writer.release()
    print(f"对比视频生成完成: {output_file}")
    return output_file

# ============================== #
# 主训练函数（使用PAM格式）
# ============================== #
def train_wisppn_pam_model(train_loader, val_loader, test_loader,
                           batch_size=32, num_epochs=20, learning_rate=0.001,
                           keypoint_scale=1000.0, output_dir="wisppn_pam_results",
                           resume=True):
    """训练使用PAM格式标签的WISPPN模型（已添加断点续训功能）"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

    # 初始化模型
    print("初始化WISPPN模型（PAM版本）...")
    wisppn = ResNet(ResidualBlock, [2, 2, 2, 2], input_channels=1140)
    wisppn = wisppn.to(device)

    # ========== 极简模型统计（只要这几行） ==========
    # 1. 计算总参数量
    total_params = sum(p.numel() for p in wisppn.parameters() if p.requires_grad)
    print(f"\n📊 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. 计算FLOPs
    try:
        from thop import profile
        import copy

        # 创建正确大小的测试输入
        model_copy = copy.deepcopy(wisppn)
        model_copy.eval()

        # 注意：输入应该是转换后的格式 [1, 600, 3, 6]
        test_input = torch.randn(1, 1140, 1, 3).to(device)

        with torch.no_grad():
            flops, params = profile(model_copy, inputs=(test_input,), verbose=False)
        print(f"💻 模型计算量: {flops / 1e6:.2f}M FLOPs")
        print(f"📊 THOP参数量: {params:,} ({params / 1e6:.2f}M)")

        del model_copy, test_input

    except ImportError:
        print("💻 FLOPs计算需要安装: pip install thop")
    except Exception as e:
        print(f"💻 FLOPs计算出错: {e}")
        print("💻 跳过FLOPs计算，继续训练...")

    # 损失函数和优化器
    criterion_L2 = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(wisppn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 30], gamma=0.5)

    # 训练历史记录（只记录训练损失）
    train_losses = []

    # ============================== #
    # 断点续训加载逻辑 (新增)
    # ============================== #
    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"🔄 发现检查点: {checkpoint_path}，正在恢复训练状态...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            wisppn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            train_losses = checkpoint.get('train_losses', [])
            print(f"✅ 成功恢复! 将从 Epoch {start_epoch + 1} 继续训练。")
        except Exception as e:
            print(f"⚠️ 恢复检查点失败: {e}，将重新开始训练。")
            start_epoch = 0
            train_losses = []
    else:
        print("🆕 未发现检查点或不进行恢复，开始新的训练。")

    # ============================== #
    # 训练循环（与原始代码一致，无验证）
    # ============================== #
    print("开始训练...")
    wisppn.train()

    accumulation_steps = 4
    optimizer.zero_grad()  # 循环开始前先清零

    for epoch_index in range(start_epoch, num_epochs):
        start = time.time()
        epoch_losses = []

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch_index + 1}/{num_epochs}")

        # 加上 enumerate，不然下面找不到 batch_index
        for batch_index, batch_data in enumerate(train_loop):

            # 1. 获取原始 WiFi 数据并转换格式
            batch_x_raw = batch_data["input_wifi-csi"].float()
            batch_x = convert_csi_format(batch_x_raw).to(device, non_blocking=True)

            # 获取真实输出 [B, 17, 3]
            batch_y = batch_data["output"].float().to(device, non_blocking=True)

            # 前向传播
            pred_keypoint = wisppn(batch_x)

            # 2：把用来记录的真实 loss 和用来反向传播的 loss_scaled 分开，不然你打印出来的 loss 会小 4 倍
            loss = criterion_L2(pred_keypoint, batch_y)
            epoch_losses.append(loss.item())

            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            # 3：加上 "or (batch_index + 1) == len(train_loop)"，防止最后一个批次漏更新
            if (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(train_loop):
                optimizer.step()
                optimizer.zero_grad()

            train_loop.set_postfix(loss=loss.item())

        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)

        endl = time.time()
        print(f'Epoch {epoch_index + 1} Avg Loss: {avg_epoch_loss:.6f}, 耗时: {(endl - start) / 60:.2f} 分钟')

        scheduler.step()

        # 保存检查点
        torch.save({
            'epoch': epoch_index + 1,
            'model_state_dict': wisppn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: Epoch {epoch_index + 1}")

    # 保存模型（与原始代码一致）
    os.makedirs('weights', exist_ok=True)
    model_path = f'weights/wisppn-{num_epochs}epochs.pkl'
    torch.save(wisppn, model_path)
    print(f"模型已保存到 {model_path}")

    # ============================== #
    # 测试阶段（类似原始代码，但添加评估指标）
    # ============================== #
    print("\n开始测试...")
    wisppn = wisppn.to(device).eval()

    # 用于保存所有预测和真实坐标
    all_pred_coords = []
    all_true_coords = []

    # 评估指标（额外添加的，原始代码没有）
    test_mpes = []
    test_pcks = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: []}

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing")

        for batch_idx, batch_data in enumerate(test_loop):
            # 转换 CSI 格式
            batch_x_raw = batch_data["input_wifi-csi"].float()
            csi_data = convert_csi_format(batch_x_raw).to(device)

            # 真实标签 [B, 17, 3]
            true_keypoints = batch_data["output"].float().to(device)

            # 预测 [B, 17, 3]
            pred_keypoints = wisppn(csi_data)

            # 提取结果用于视频
            all_pred_coords.extend(pred_keypoints.cpu().numpy())
            all_true_coords.extend(true_keypoints.cpu().numpy())

            mpe = mean_keypoint_error(pred_keypoints, true_keypoints, keypoint_scale)
            test_mpes.append(mpe)

            # �� 修复 2：正确调用 PCK 函数（分别计算不同的阈值，取返回数组的最后一个值作为整体平均）
            pck_10 = calculate_pck_metrics(pred_keypoints, true_keypoints, 0.1)[-1] * 100
            pck_20 = calculate_pck_metrics(pred_keypoints, true_keypoints, 0.2)[-1] * 100
            pck_30 = calculate_pck_metrics(pred_keypoints, true_keypoints, 0.3)[-1] * 100
            pck_40 = calculate_pck_metrics(pred_keypoints, true_keypoints, 0.4)[-1] * 100
            pck_50 = calculate_pck_metrics(pred_keypoints, true_keypoints, 0.5)[-1] * 100

            test_pcks[0.1].append(pck_10)
            test_pcks[0.2].append(pck_20)
            test_pcks[0.3].append(pck_30)
            test_pcks[0.4].append(pck_40)
            test_pcks[0.5].append(pck_50)

            # 更新测试进度条
            test_loop.set_postfix(mpe=f"{mpe:.2f}", pck20=f"{pck_20:.2f}")

            # 可选：显示第一个批次的预测（类似原始代码的可视化）
            if batch_idx == 0:
                print(f"\n第一个批次的预测示例:")
                print(f"  预测坐标范围 - X: [{pred_keypoints[0, :, 0].min():.2f}, "
                      f"{pred_keypoints[0, :, 0].max():.2f}]")
                print(f"  预测坐标范围 - Y: [{pred_keypoints[0, :, 1].min():.2f}, "
                      f"{pred_keypoints[0, :, 1].max():.2f}]")

    # 计算平均评估指标
    avg_test_mpe = np.mean(test_mpes) if test_mpes else float('inf')
    avg_test_pcks = {k: np.mean(v) if v else 0.0 for k, v in test_pcks.items()}

    print(f"\n🎯 测试结果:")
    print(f"   MPE: {avg_test_mpe:.4f}")
    print(f"   PCK@0.1: {avg_test_pcks[0.1]:.4f}")
    print(f"   PCK@0.2: {avg_test_pcks[0.2]:.4f}")
    print(f"   PCK@0.3: {avg_test_pcks[0.3]:.4f}")
    print(f"   PCK@0.4: {avg_test_pcks[0.4]:.4f}")
    print(f"   PCK@0.5: {avg_test_pcks[0.5]:.4f}")

    # ============================== #
    # 生成可视化视频
    # ============================== #
    if all_pred_coords and all_true_coords:
        print("\n生成姿态视频...")

        # 转换为numpy数组
        all_pred_coords_np = np.array(all_pred_coords)
        all_true_coords_np = np.array(all_true_coords)

        # 限制视频长度
        max_frames = min(720, len(all_pred_coords_np))

        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        print(f"生成前{max_frames}帧的视频...")

        try:
            # 生成真实姿态视频
            true_video = create_pose_animation_opencv(
                all_true_coords_np[:max_frames],
                output_file=os.path.join(videos_dir, "true_poses.mp4"),
                keypoint_scale=keypoint_scale,
                show_labels=False
            )

            # 生成预测姿态视频
            pred_video = create_pose_animation_opencv(
                all_pred_coords_np[:max_frames],
                output_file=os.path.join(videos_dir, "predicted_poses.mp4"),
                keypoint_scale=keypoint_scale,
                show_labels=False
            )

            # 生成对比视频
            comparison_video = create_side_by_side_video_opencv(
                all_true_coords_np[:max_frames],
                all_pred_coords_np[:max_frames],
                output_file=os.path.join(videos_dir, "comparison.mp4"),
                keypoint_scale=keypoint_scale,
                fps=30
            )

            print(f"\n视频生成完成!")
            print(f"  真实姿态视频: {true_video}")
            print(f"  预测姿态视频: {pred_video}")
            print(f"  对比视频: {comparison_video}")

        except Exception as e:
            print(f"生成视频时出错: {e}")
            import traceback
            traceback.print_exc()

    # ============================== #
    # 保存结果
    # ============================== #
    os.makedirs(output_dir, exist_ok=True)

    # 保存测试结果
    test_results = {
        'MPE': avg_test_mpe,
        'PCK@0.1': avg_test_pcks[0.1],
        'PCK@0.2': avg_test_pcks[0.2],
        'PCK@0.3': avg_test_pcks[0.3],
        'PCK@0.4': avg_test_pcks[0.4],
        'PCK@0.5': avg_test_pcks[0.5]
    }

    import pandas as pd
    results_df = pd.DataFrame([test_results])
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.show()

    print(f"\n所有结果已保存到: {output_dir}")

    return wisppn, test_results


def main():
    # ============================== #
    # 1. 基础配置
    # ============================== #
    # 训练参数
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.001

    # 关键点缩放因子 (虽然Dataset不需要了，但后面计算MPE误差和画图时还需要用到它)
    keypoint_scale = 1000.0

    # 数据目录
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
            batch_size=batch_size  # 优先使用命令行传入的 batch_size
        )

        # 3. 使用 sklearn 将 test_dataset 一分为二 (50% 验证，50% 测试)
        val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)

        val_loader = make_dataloader(
            val_data, is_training=False, generator=rng_generator, batch_size=batch_size // 2
        )
        test_loader = make_dataloader(
            test_data, is_training=False, generator=rng_generator, batch_size=batch_size // 2
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

    # 训练模型
    model, test_results = train_wisppn_pam_model(
        train_loader, val_loader, test_loader,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        keypoint_scale=1000.0,
        output_dir="wisppn",
        resume = True
    )

    print("\n训练完成！")


if __name__ == "__main__":
    main()