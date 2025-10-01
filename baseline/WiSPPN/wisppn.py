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

# ============================== #
# 数据加载和转换相关代码（修改为支持PAM格式）
# ============================== #

KEEP_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
KEYPOINT_MAPPING = {old_idx: new_idx for new_idx, old_idx in enumerate(KEEP_KEYPOINTS)}


class PreprocessedCSIKeypointsDataset(Dataset):
    """使用预处理后的CSI数据和PAM格式标签的数据集（优化版本）"""

    def __init__(self, csi_data_dir, pam_label_dir, keypoint_scale=1000.0, transform=None, enable_zero_clean=True):
        self.csi_windows = np.load(os.path.join(csi_data_dir, "csi_windows.npy"))

        window_info = np.load(os.path.join(csi_data_dir, "window_info.npz"))
        self.window_to_file = window_info['window_to_file']
        self.window_to_frame = window_info['window_to_frame']

        file_info = np.load(os.path.join(csi_data_dir, "file_info.npz"), allow_pickle=True)
        self.file_ids = file_info['file_ids']
        self.window_ranges = file_info['window_ranges']

        config = np.load(os.path.join(csi_data_dir, "config.npz"))
        self.window_size = config['window_size']
        self.stride = config['stride']

        self.keypoint_scale = keypoint_scale
        self.transform = transform
        self.pam_label_dir = pam_label_dir
        self.enable_zero_clean = enable_zero_clean

        # 缓存单个PAM帧（而不是整个序列）
        self._pam_cache = {}
        self._cache_size = 100  # 增加缓存大小

        print(f"加载了 {len(self.csi_windows)} 个CSI窗口")
        print(f"PAM标签目录: {pam_label_dir}")
        print(f"文件ID数量: {len(self.file_ids)}")
        print(f"零值清理: {'启用(单帧均值法)' if enable_zero_clean else '禁用'}")

    def _get_pam_file_path(self, file_idx, frame_idx):
        """根据文件索引和帧索引获取PAM文件路径"""
        file_id = self.file_ids[file_idx]
        pam_filename = f"{file_id}_dual_cropped_frame_{frame_idx:06d}.mat"

        for person_idx in range(1, 6):
            person_dir = os.path.join(self.pam_label_dir, f"wisppn_labels{person_idx}")
            pam_path = os.path.join(person_dir, pam_filename)
            if os.path.exists(pam_path):
                return pam_path
        return None

    def _clean_single_frame_zeros(self, keypoint):
        """清理单帧中的零值关键点（单帧均值法）"""
        cleaned = keypoint.copy()

        # 找到非零关键点的掩码
        non_zero_mask = (keypoint[:, 0] != 0) | (keypoint[:, 1] != 0)

        # 只有当存在至少一个非零点时才进行处理
        if non_zero_mask.any():
            # 计算非零关键点的平均位置
            mean_pos = keypoint[non_zero_mask].mean(axis=0)
            # 找到所有零值点的索引
            zero_indices = np.where(~non_zero_mask)[0]

            # 用均值替换零值点
            for idx in zero_indices:
                cleaned[idx] = mean_pos

        return cleaned

    def _load_pam_frame(self, file_idx, frame_idx):
        """加载特定帧的PAM数据（按需加载）"""
        cache_key = f"{file_idx}_{frame_idx}"

        if cache_key in self._pam_cache:
            return self._pam_cache[cache_key]

        # 缓存管理 - LRU策略
        if len(self._pam_cache) >= self._cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self._pam_cache))
            del self._pam_cache[oldest_key]

        # 获取PAM文件路径
        pam_path = self._get_pam_file_path(file_idx, frame_idx)

        if pam_path is None:
            # 文件不存在，返回零矩阵
            pam_data = np.zeros((3, 15, 15), dtype=np.float32)
        else:
            try:
                # 加载.mat文件
                data = hdf5storage.loadmat(pam_path)
                pam_data = data['jointsMatrix'][:3, :, :].astype(np.float32)

                # 如果启用清理，应用单帧均值清理
                if self.enable_zero_clean:
                    # 提取关键点坐标
                    keypoints = np.zeros((15, 2), dtype=np.float32)
                    for kp_idx in range(15):
                        keypoints[kp_idx, 0] = pam_data[0, kp_idx, kp_idx]
                        keypoints[kp_idx, 1] = pam_data[1, kp_idx, kp_idx]

                    # 清理零值
                    cleaned_keypoints = self._clean_single_frame_zeros(keypoints)

                    # 更新PAM矩阵
                    for kp_idx in range(15):
                        # 更新对角线（绝对坐标）
                        pam_data[0, kp_idx, kp_idx] = cleaned_keypoints[kp_idx, 0]
                        pam_data[1, kp_idx, kp_idx] = cleaned_keypoints[kp_idx, 1]

                    # 更新非对角线（相对坐标）
                    for i in range(15):
                        for j in range(15):
                            if i != j:
                                pam_data[0, i, j] = cleaned_keypoints[i, 0] - cleaned_keypoints[j, 0]
                                pam_data[1, i, j] = cleaned_keypoints[i, 1] - cleaned_keypoints[j, 1]

                # 归一化坐标
                pam_data[0:2, :, :] = pam_data[0:2, :, :] / self.keypoint_scale
                # 置信度保持不变

            except Exception as e:
                print(f"加载PAM文件失败 {pam_path}: {e}")
                pam_data = np.zeros((3, 15, 15), dtype=np.float32)

        # 缓存结果
        self._pam_cache[cache_key] = pam_data
        return pam_data

    def __len__(self):
        return len(self.csi_windows)

    def __getitem__(self, idx):
        # 获取CSI窗口
        csi_window = self.csi_windows[idx]

        # 获取对应的文件和帧索引
        file_idx = self.window_to_file[idx]
        frame_idx = self.window_to_frame[idx]

        # 按需加载单个PAM帧
        pam_label = self._load_pam_frame(file_idx, frame_idx)

        # 转换为张量
        csi_tensor = torch.from_numpy(csi_window).float()
        pam_tensor = torch.from_numpy(pam_label).float()

        if self.transform:
            csi_tensor = self.transform(csi_tensor)

        return csi_tensor, pam_tensor

    def get_file_indices(self):
        """获取所有文件索引"""
        return list(range(len(self.file_ids)))

    def get_samples_from_file(self, file_idx):
        """获取指定文件的所有样本索引"""
        start_idx, end_idx = self.window_ranges[file_idx]
        return list(range(start_idx, end_idx))

def create_train_val_test_loaders(dataset, batch_size=32, num_workers=0, random_seed=42):
    """按文件级别划分预处理后的数据集"""
    # 设置随机种子
    np.random.seed(random_seed)

    # 获取所有文件索引
    file_indices = dataset.get_file_indices()
    total_files = len(file_indices)

    # 随机打乱文件顺序
    np.random.shuffle(file_indices)

    # 按比例划分文件
    train_ratio, val_ratio = 0.7, 0.15
    train_split = int(np.floor(train_ratio * total_files))
    val_split = int(np.floor(val_ratio * total_files))

    # 获取每个集合的文件索引
    train_file_indices = file_indices[:train_split]
    val_file_indices = file_indices[train_split:train_split + val_split]
    test_file_indices = file_indices[train_split + val_split:]

    # 获取每个集合的样本索引
    train_indices = []
    val_indices = []
    test_indices = []

    for file_idx in train_file_indices:
        train_indices.extend(dataset.get_samples_from_file(file_idx))

    for file_idx in val_file_indices:
        val_indices.extend(dataset.get_samples_from_file(file_idx))

    for file_idx in test_file_indices:
        test_indices.extend(dataset.get_samples_from_file(file_idx))

    print(f"训练集: {len(train_indices)} 样本 (来自 {len(train_file_indices)} 个文件)")
    print(f"验证集: {len(val_indices)} 样本 (来自 {len(val_file_indices)} 个文件)")
    print(f"测试集: {len(test_indices)} 样本 (来自 {len(test_file_indices)} 个文件)")

    # 创建子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

    return train_loader, val_loader, test_loader


# ============================== #
# CSI数据格式转换函数
# ============================== #

def convert_csi_format(csi_data_code2):
    """
    将代码2的CSI数据格式转换为代码1的格式
    输入: [batch_size, 540, 20]
    输出: [batch_size, 600, 3, 6]
    """
    batch_size = csi_data_code2.shape[0]

    # 步骤1: 分离两个接收端
    csi_split = csi_data_code2.view(batch_size, 2, 270, 20)

    # 步骤2: 每个接收端重塑为 (30子载波, 3发送天线, 3接收天线)
    csi_reshaped = csi_split.view(batch_size, 2, 30, 3, 3, 20)

    # 步骤3: 调整维度顺序，将时间维度移到前面
    csi_reordered = csi_reshaped.permute(0, 1, 5, 2, 3, 4)

    # 步骤4: 合并时间和子载波维度，同时合并两个接收端的接收天线
    batch_size, num_receivers, time_steps, num_subcarriers, tx_antennas, rx_antennas = csi_reordered.shape

    # 重塑为最终格式
    csi_final = csi_reordered.contiguous().view(
        batch_size,
        time_steps * num_subcarriers,  # 600 = 20 × 30
        tx_antennas,  # 3
        num_receivers * rx_antennas  # 6 = 2 × 3
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
    def __init__(self, block, layers, input_channels=600):
        super(ResNet, self).__init__()

        self.in_channels = input_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, 600, layers[1], 2)
        self.layer3 = self.make_layer(block, 1024, layers[2], 2)
        self.layer4 = self.make_layer(block, 1024, layers[3], 2)

        self.decode = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 输出2通道(x', y')
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
        # 输入: [batch_size, 600, 3, 6]
        # 上采样到对称的分辨率
        x = F.interpolate(x, size=(120, 120), mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.decode(x)
        # 输出: [batch_size, 2, height, width]

        # 最终输出: [batch_size, 2, 15, 15]
        return x


# ============================== #
# 关键点提取函数（从PAM对角线提取）
# ============================== #

def extract_keypoints_from_pam(pam_data, num_keypoints=15):
    """
    从PAM矩阵中提取关键点坐标（用于测试）
    输入: pam_data [batch_size, 2, 15, 15] - PAM的x'和y'通道
    输出: keypoints [batch_size, num_keypoints, 2] - 提取的关键点坐标
    """
    batch_size = pam_data.shape[0]

    # 初始化关键点坐标
    keypoints = torch.zeros(batch_size, num_keypoints, 2, device=pam_data.device)

    # 从对角线元素提取坐标
    for b in range(batch_size):
        for k in range(num_keypoints):
            keypoints[b, k, 0] = pam_data[b, 0, k, k]  # x坐标从第0个通道的对角线
            keypoints[b, k, 1] = pam_data[b, 1, k, k]  # y坐标从第1个通道的对角线

    return keypoints


# ============================== #
# 评估函数
# ============================== #

def mean_keypoint_error(pred, target, keypoint_scale=1000.0):
    """计算平均关键点误差"""
    # 确保输入维度正确
    if len(pred.shape) == 3 and pred.shape[1] == 2 and pred.shape[2] == 15:
        pred = pred.transpose(1, 2)
    if len(target.shape) == 3 and target.shape[1] == 2 and target.shape[2] == 15:
        target = target.transpose(1, 2)

    # 计算欧氏距离
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))

    # 计算平均误差
    mpe = torch.mean(distances) * keypoint_scale

    return mpe.item()


def calculate_pck_metrics(pred_coords, true_coords, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """计算PCK指标 - 修正版本"""
    batch_size = pred_coords.shape[0]

    # 使用正确的躯干长度作为参考：Neck(1) to MidHip(8)
    NECK_IDX = 2  # 右肩
    MIDHIP_IDX = 12  # 左髋

    # 计算躯干长度作为归一化参考
    torso_length = torch.sqrt(
        torch.sum((true_coords[:, NECK_IDX] - true_coords[:, MIDHIP_IDX]) ** 2, dim=1)
    )
    torso_length = torch.clamp(torso_length, min=0.01)  # 避免除零

    # 计算所有关键点的欧氏距离
    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=2))

    # 归一化距离（相对于躯干长度）
    normalized_distances = distances / torso_length.unsqueeze(1)

    pck_results = {}
    for threshold in thresholds:
        correct = (normalized_distances < threshold).float()
        pck = torch.mean(correct).item()
        pck_results[threshold] = pck

    return pck_results


# ============================== #
# 姿态可视化相关设置
# ============================== #

SKELETON_CONNECTIONS = [
    (0, 1), (1, 8),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (8, 9), (8, 12),
    (9, 10), (10, 11),
    (12, 13), (13, 14)
]

# 定义关键点的身体部位名称
KEYPOINT_NAMES = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle"
}
# ============================== #
# 视频可视化函数
# ============================== #

# ============================== #
# 姿态可视化相关设置
# ============================== #

# 根据图片中的关键点连接定义骨架连接
SKELETON_CONNECTIONS = [
    # 躯干
    (0, 1), (1, 8),
    # 左臂
    (1, 2), (2, 3), (3, 4),
    # 右臂
    (1, 5), (5, 6), (6, 7),
    # 下半身
    (8, 9), (8, 12),
    # 左腿
    (9, 10), (10, 11),
    # 右腿
    (12, 13), (13, 14)
]

# 定义关键点的身体部位名称 - 简化为15个点
# 定义关键点的身体部位名称
KEYPOINT_NAMES = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle"
}

# 定义身体部位颜色
BODY_PART_COLORS = {
    'head': 'magenta',
    'torso': 'red',
    'right_arm': 'orange',  # 右臂
    'left_arm': 'green',     # 左臂
    'right_leg': 'blue',     # 右腿
    'left_leg': 'cyan'       # 左腿
}

# 将关键点分组到不同的身体部位 - 简化为15个点
KEYPOINT_GROUPS = {
    'head': [0],
    'torso': [1, 8],
    'left_arm': [2, 3, 4],
    'right_arm': [5, 6, 7],
    'left_leg': [9, 10, 11],
    'right_leg': [12, 13, 14]
}

# 为每个连接分配颜色
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

# ============================== #
# 主训练函数（使用PAM格式）
# ============================== #
def train_wisppn_pam_model(train_loader, val_loader, test_loader,
                           batch_size=32, num_epochs=20, learning_rate=0.001,
                           keypoint_scale=1000.0, output_dir="wisppn_pam_results"):
    """训练使用PAM格式标签的WISPPN模型（与原始代码保持一致）"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    print("初始化WISPPN模型（PAM版本）...")
    wisppn = ResNet(ResidualBlock, [2, 2, 2, 2], input_channels=600)
    wisppn = wisppn.cuda()

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
        test_input = torch.randn(1, 600, 3, 6).to(device)

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
    criterion_L2 = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(wisppn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 30], gamma=0.5)

    # 训练历史记录（只记录训练损失）
    train_losses = []

    # ============================== #
    # 训练循环（与原始代码一致，无验证）
    # ============================== #
    print("开始训练（使用PAM标签）...")
    wisppn.train()

    for epoch_index in range(num_epochs):
        start = time.time()

        # 打乱数据（原始代码中的shuffle(mats)）
        # DataLoader已经设置了shuffle=True，所以这里不需要额外操作

        epoch_losses = []

        # 训练批次循环
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch_index + 1}/{num_epochs}")

        for batch_index, (csi_batch, pam_batch) in enumerate(train_loop):
            # 转换CSI格式
            csi_data = convert_csi_format(csi_batch).cuda()

            # PAM标签处理（与原始代码一致）
            xy = pam_batch[:, 0:2, :, :].cuda()  # x'和y'通道
            confidence = pam_batch[:, 2:4, :, :].cuda()  # 置信度通道（原始代码用2个通道）

            # 如果只有3个通道，复制置信度通道
            if pam_batch.shape[1] == 3:
                confidence = pam_batch[:, 2:3, :, :].cuda()
                confidence = confidence.repeat(1, 2, 1, 1)  # 扩展为2个通道

            # 前向传播
            pred_xy = wisppn(csi_data)

            # 计算损失（与原始代码完全一致）
            loss = criterion_L2(torch.mul(confidence, pred_xy), torch.mul(confidence, xy))

            # 打印损失（原始代码的print(loss.item())）
            if batch_index % 10 == 0:  # 每10个批次打印一次
                print(f"Batch {batch_index}: {loss.item():.6f}")

            # 记录损失
            epoch_losses.append(loss.item())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条
            train_loop.set_postfix(loss=loss.item())

        # 计算epoch平均损失
        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)

        endl = time.time()
        print(f'Epoch {epoch_index + 1}: Avg Loss: {avg_epoch_loss:.6f}, '
              f'Costing time: {(endl - start) / 60:.2f} minutes')

        scheduler.step()  # 移到这里

    # 保存模型（与原始代码一致）
    os.makedirs('weights', exist_ok=True)
    model_path = f'weights/wisppn-{num_epochs}epochs.pkl'
    torch.save(wisppn, model_path)
    print(f"模型已保存到 {model_path}")

    # ============================== #
    # 测试阶段（类似原始代码，但添加评估指标）
    # ============================== #
    print("\n开始测试...")
    wisppn = wisppn.cuda().eval()

    # 用于保存所有预测和真实坐标
    all_pred_coords = []
    all_true_coords = []

    # 评估指标（额外添加的，原始代码没有）
    test_mpes = []
    test_pcks = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: []}

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing")

        for batch_idx, (csi_batch, pam_batch) in enumerate(test_loop):
            # 转换CSI格式
            csi_data = convert_csi_format(csi_batch).cuda()

            # 前向传播
            pred_xy = wisppn(csi_data)  # 输出 [batch_size, 2, 15, 15]

            # 从PAM对角线提取关键点坐标（与原始测试代码一致）
            batch_size = pred_xy.shape[0]
            pred_keypoints = torch.zeros(batch_size, 15, 2).cuda()
            true_keypoints = torch.zeros(batch_size, 15, 2).cuda()

            for b in range(batch_size):
                for index in range(15):
                    # 从预测PAM提取坐标
                    pred_keypoints[b, index, 0] = pred_xy[b, 0, index, index]  # x坐标
                    pred_keypoints[b, index, 1] = pred_xy[b, 1, index, index]  # y坐标

                    # 从真实PAM提取坐标
                    true_keypoints[b, index, 0] = pam_batch[b, 0, index, index].cuda()
                    true_keypoints[b, index, 1] = pam_batch[b, 1, index, index].cuda()

            # 保存坐标用于视频生成
            all_pred_coords.extend(pred_keypoints.cpu().numpy())
            all_true_coords.extend(true_keypoints.cpu().numpy())

            # 计算评估指标（额外添加的功能）
            mpe = mean_keypoint_error(pred_keypoints, true_keypoints, keypoint_scale)
            test_mpes.append(mpe)

            pck_results = calculate_pck_metrics(pred_keypoints, true_keypoints,
                                                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])
            for threshold, value in pck_results.items():
                test_pcks[threshold].append(value)

            # 更新进度条
            test_loop.set_postfix(mpe=mpe, pck20=pck_results[0.2])

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
    # 训练参数
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # 数据目录
    csi_data_dir = "preprocessed_csi_data"  # CSI预处理数据
    pam_label_dir = "keypoints_pam_data"  # PAM格式标签目录

    # 检查数据是否存在
    if not os.path.exists(csi_data_dir) or not os.path.exists(os.path.join(csi_data_dir, "csi_windows.npy")):
        print(f"错误: 未找到CSI预处理数据 {csi_data_dir}")
        return

    if not os.path.exists(pam_label_dir):
        print(f"错误: 未找到PAM标签目录 {pam_label_dir}")
        return

    # 创建数据集
    print("正在加载数据...")
    full_dataset = PreprocessedCSIKeypointsDataset(
        csi_data_dir=csi_data_dir,
        pam_label_dir=pam_label_dir,
        keypoint_scale=1000.0,  # 添加这一行
        enable_zero_clean=True  # 启用零值清理
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        dataset=full_dataset,
        batch_size=batch_size,
        num_workers=0
    )

    # 测试数据加载
    print("\n测试数据加载...")
    for csi_batch, pam_batch in train_loader:
        print(f"CSI数据形状: {csi_batch.shape}")  # [batch_size, 540, 20]
        print(f"PAM标签形状: {pam_batch.shape}")  # [batch_size, 4, 15, 15]
        print(f"  - x'通道形状: {pam_batch[:, 0, :, :].shape}")
        print(f"  - y'通道形状: {pam_batch[:, 1, :, :].shape}")
        print(f"  - c'通道形状: {pam_batch[:, 2, :, :].shape}")
        break

    # 训练模型
    model, test_results = train_wisppn_pam_model(
        train_loader, val_loader, test_loader,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        keypoint_scale=1000.0,
        output_dir="wisppn_pam_results3"
    )

    print("\n训练完成！")


if __name__ == "__main__":
    main()