import numpy as np
import csv
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os.path
import torch.nn as nn
from torch.autograd import Variable
import time
from torchvision.transforms import Resize
from ChannelTrans import ChannelTransformer
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import random
import os
import torchvision
from torchvision.models import ResNet34_Weights  # Add this import
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import glob
import hdf5storage
import re
import pickle
import yaml
import seaborn
from sklearn.model_selection import train_test_split
from mmfi import make_dataset, make_dataloader  # 确保 mmfi.py 在同目录下

# ============================== #
# 原始的posenet模型 - 保持不变，只做输入输出适配
# ============================== #
class posenet(nn.Module):
    """
    WiFi-based Human Pose Estimation Network (WPFormer)

    Input: WiFi CSI data from 6 antenna pairs
    Output: 2D pose landmarks (y coordinates for 15 keypoints)

    Modified Architecture:
    - Processes 6 CSI streams from antenna pairs
    - Uses shared ResNet34 encoder for feature extraction
    - Applies Channel Transformer for feature integration
    - Outputs 2×15 pose coordinates via decoder and average pooling
    """

    def __init__(self):
        super(posenet, self).__init__()

        # Create ResNet34 encoder (shared weights for all 6 antenna pairs)
        # FIX: Use weights parameter instead of deprecated pretrained parameter
        # resnet_raw_model1 = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # 项目目录下的权重文件路径
        project_weight_path = "./resnet34-b627a593.pth"  # 相对路径
        # 或者使用绝对路径：
        # project_weight_path = "/home/DY/resnet34-b627a593.pth"

        try:
            if os.path.exists(project_weight_path):
                print(f"发现项目目录下的权重文件: {project_weight_path}")

                # 创建不带权重的ResNet34模型
                resnet_raw_model1 = torchvision.models.resnet34(weights=None)

                # 手动加载权重
                print("正在加载权重...")
                state_dict = torch.load(project_weight_path, map_location='cpu')
                resnet_raw_model1.load_state_dict(state_dict)
                print("✅ 成功从项目目录加载预训练权重！")

            else:
                print(f"❌ 未找到权重文件: {project_weight_path}")
                print("请确保将 resnet34-b627a593.pth 文件放到项目目录下")
                print("使用随机初始化权重...")
                resnet_raw_model1 = torchvision.models.resnet34(weights=None)

        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            print("使用随机初始化权重...")
            resnet_raw_model1 = torchvision.models.resnet34(weights=None)

        # Expected feature map sizes at each ResNet layer
        filters = [64, 64, 128, 256, 512]

        # Encoder components from ResNet34 (shared across all 6 antenna pairs)
        # Input: 1-channel CSI data -> 64-channel feature maps
        self.encoder_conv1_p1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=False)
        self.encoder_bn1_p1 = resnet_raw_model1.bn1
        self.encoder_relu_p1 = resnet_raw_model1.relu

        # ResNet34 layers (Block 1-5 in paper's Table I)
        self.encoder_layer1_p1 = resnet_raw_model1.layer1  # Block 1: 64×136×32
        self.encoder_layer2_p1 = resnet_raw_model1.layer2  # Block 2: 128×68×16
        self.encoder_layer3_p1 = resnet_raw_model1.layer3  # Block 3: 256×34×8
        self.encoder_layer4_p1 = resnet_raw_model1.layer4  # Block 4: 512×17×4

        # Channel Transformer for feature integration
        # Input: 512×15×24 (concatenated features from 6 antenna pairs)
        # Output: 512×15×24 (same size, with attention weights)
        self.tf = ChannelTransformer(vis=False, img_size=[17, 12], channel_num=512, num_layers=1, num_heads=3)

        # Decoder: 512×15×24 -> 2×15×24 (pose coordinates)
        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),  # 2 channels for (x,y) coordinates
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(3)  # For final 2D pose output
        self.bn2 = nn.BatchNorm2d(512)  # For concatenated features before transformer
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through WPFormer network

        Args:
            x: Input CSI data, shape [batch_size, 540, 20]
               Modified to accept the new data format and convert it internally

        Returns:
            x: Pose landmarks, shape [batch_size, 15, 2] (15 keypoints with x,y coordinates)
            time_sum: Forward pass execution time
        """

        # 创建resize函数
        # x 的输入形状: [batch_size, 3, 114, 10]
        torch_resize = Resize([136, 32], antialias=True)

        x_resized = []
        # 直接遍历第 1 维（即 3 个天线）
        for i in range(3):
            # 提取单根天线的数据 -> [batch_size, 114, 10]
            x_ant = x[:, i, :, :]
            # 进行缩放 -> [batch_size, 136, 32]
            x_part = torch_resize(x_ant)
            # 添加channel维度
            x_part = x_part.unsqueeze(1)  # [batch_size, 1, 136, 32]
            x_resized.append(x_part)

        time_start = time.time()

        # 对所有6个天线对进行编码
        encoded_features = []

        for x_input in x_resized:
            # Initial convolution layer
            x_feat = self.encoder_conv1_p1(x_input)  # [batch_size, 64, 136, 32]
            x_feat = self.encoder_bn1_p1(x_feat)
            x_feat = self.encoder_relu_p1(x_feat)

            # ResNet34 layers
            x_feat = self.encoder_layer1_p1(x_feat)  # [batch_size, 64, 136, 32]
            x_feat = self.encoder_layer2_p1(x_feat)  # [batch_size, 128, 68, 16]
            x_feat = self.encoder_layer3_p1(x_feat)  # [batch_size, 256, 34, 8]
            x_feat = self.encoder_layer4_p1(x_feat)  # [batch_size, 512, 17, 4]

            encoded_features.append(x_feat)

        # Concatenation step
        # Concatenate features from 6 antenna pairs along width dimension
        # Each x_i: [batch_size, 512, 17, 4]
        # Result: [batch_size, 512, 17, 12] (3*4=12 in width dimension)
        x = torch.cat(encoded_features, dim=3)

        # Batch normalization before transformer
        x = self.bn2(x)  # [batch_size, 512, 17, 12]

        # Channel Transformer
        # Input: [batch_size, 512, 17, 12] -> Output: [batch_size, 512, 17, 12]
        x, weight = self.tf(x)

        # Decoder
        # Input: [batch_size, 512, 15, 24] -> Output: [batch_size, 3, 17, 12]
        x = self.decode(x)

        # Average pooling
        # Pool across width dimension: [batch_size, 3, 17, 12] -> [batch_size, 3, 17, 1]
        m = torch.nn.AvgPool2d((1, 12), stride=(1, 1))
        x = m(x).squeeze(dim=3)  # [batch_size, 3, 17]

        # Final batch normalization
        x = self.bn1(x)

        time_end = time.time()
        time_sum = time_end - time_start

        # Transpose to get final pose format
        # [batch_size, 3, 17] -> [batch_size, 17, 3]
        x = torch.transpose(x, 1, 2)

        # 输出适配：只保留前15个关键点
        # x = x[:, :17, :]  # [batch_size, 17, 3]

        return x, time_sum


def weights_init(m):
    """
    Initialize network weights

    Args:
        m: Neural network module to initialize
    """
    if isinstance(m, nn.Conv2d):
        # Xavier normal initialization for convolutional layers
        nn.init.xavier_normal_(m.weight.data)
        # Note: bias initialization is commented out
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        # Constant initialization for 2D batch normalization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        # Constant initialization for 1D batch normalization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


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
                                 figsize=(800, 960), keypoint_scale=1000.0):
    """使用OpenCV创建人体姿态动画"""
    width, height = figsize
    frames = len(all_keypoints)

    # 确保关键点是numpy数组
    if torch.is_tensor(all_keypoints):
        all_keypoints = all_keypoints.cpu().numpy()

    # 缩放关键点
    all_keypoints = all_keypoints * keypoint_scale

    # 计算全局边界
    all_x = all_keypoints[:, :, 0].flatten()
    all_y = all_keypoints[:, :, 1].flatten()

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    margin = 0.1
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"生成姿态动画: {output_file}，共 {frames} 帧")

    with tqdm(total=frames, desc="生成视频", unit="帧") as pbar:
        for frame_idx in range(frames):
            # 创建白色背景
            frame_img = np.ones((height, width, 3), dtype=np.uint8) * 255

            # 创建matplotlib图
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

            keypoints = all_keypoints[frame_idx]

            # 绘制骨骼连接
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                color = CONNECTION_COLORS.get(connection, 'gray')
                ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                        [keypoints[start_idx, 1], keypoints[end_idx, 1]],
                        color=color, linewidth=3)

            # 绘制关键点
            for part_name, indices in KEYPOINT_GROUPS.items():
                color = BODY_PART_COLORS[part_name]
                part_keypoints = keypoints[indices]
                ax.scatter(part_keypoints[:, 0], part_keypoints[:, 1],
                           c=color, s=50, edgecolors='black')

            # 设置坐标轴
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_max + y_margin, y_min - y_margin)
            ax.set_title(f"Frame {frame_idx + 1}/{frames}", fontsize=14)
            ax.set_aspect('equal')
            ax.axis('off')

            plt.tight_layout()

            # 转换为OpenCV格式
            fig.canvas.draw()
            mat_img = np.array(fig.canvas.renderer.buffer_rgba())
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)

            # 写入视频
            video_writer.write(mat_img)
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


def compute_3d_pck(pred, target, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    计算纯 3D 空间的 PCK (无根节点对齐) - 终极修复版
    完全兼容原代码中返回字典的调用方式
    """
    batch_size = pred.shape[0]

    # 计算 3D 空间欧氏距离
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2)) # [batch_size, 17]

    # 基准距离：Left Hip(1) 到 Right Shoulder(11)
    IDX1, IDX2 = 1, 11
    scale = torch.sqrt(torch.sum((target[:, IDX1, :] - target[:, IDX2, :]) ** 2, dim=1))
    scale = torch.clamp(scale, min=1e-5) # 防止除以 0

    # 计算归一化距离
    normalized_distances = distances / scale.unsqueeze(1) # [batch_size, 17]

    pck_results = {}
    pck_per_joint = {} # 占位返回值，以兼容 "pck_results, _ = ..." 的调用

    for thr in thresholds:
        correct_mask = (normalized_distances <= thr).float()
        # 计算整个 batch 的平均正确率，转为百分制 numpy
        pck_overall = correct_mask.mean().item() * 100
        pck_results[thr] = np.array(pck_overall)  # 包装成 numpy，防止 test_pcks[thr].append 报错

    return pck_results, pck_per_joint


# ============================== #
# 训练函数
# ============================== #
def get_gpu_memory_map():
    """获取所有GPU的显存信息"""
    result = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # 转换为GB
            result[i] = gpu_memory
    return result


def calculate_optimal_batch_size(gpu_id):
    """根据GPU显存计算最优批量大小"""
    if not torch.cuda.is_available():
        return 32

    # 获取GPU显存（GB）
    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3

    # 根据显存大小设置批量大小
    # 经验公式：每GB显存大约可以处理64个样本
    if gpu_memory > 40:  # 4090 (49GB)
        return 128  # 更大的批量
    elif gpu_memory > 20:  # 2080Ti (22GB)
        return 128
    elif gpu_memory > 10:
        return 32
    else:
        return 32


def setup_distributed(gpu_ids):
    """设置分布式训练环境"""
    if len(gpu_ids) > 1:
        # 初始化分布式训练
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=len(gpu_ids), rank=0)
        return True
    return False


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_posenet():
    # ============================== #
    # 统一输出目录配置
    # ============================== #
    output_dir = "metafi2"
    dataset_root = "/home/aip/DATA/MMFi_Clean"
    yaml_config_path = "config.yaml"

    model_dir = os.path.join(output_dir, "models")
    video_dir = os.path.join(output_dir, "videos")
    log_dir = os.path.join(output_dir, "logs")
    result_dir = os.path.join(output_dir, "results")

    for directory in [output_dir, model_dir, video_dir, log_dir, result_dir]:
        os.makedirs(directory, exist_ok=True)

    # 文件路径
    best_model_path = os.path.join(model_dir, 'posenet_best.pth')
    latest_model_path = os.path.join(model_dir, 'posenet_latest.pth')

    # 【新增】断点存档路径
    resume_checkpoint_path = os.path.join(model_dir, 'resume_checkpoint.pth')

    test_results_file = os.path.join(result_dir, 'test_results.json')
    training_curves_file = os.path.join(result_dir, 'training_curves.png')

    true_poses_video = os.path.join(video_dir, 'posenet_true_poses.mp4')
    predicted_poses_video = os.path.join(video_dir, 'posenet_predicted_poses.mp4')
    comparison_video = os.path.join(video_dir, 'posenet_comparison.mp4')

    # ============================== #
    # 设备配置 (保留你的原逻辑)
    # ============================== #
    gpu_config = '0'
    if gpu_config == 'auto':
        gpu_ids = [0]
    else:
        gpu_ids = [int(x) for x in gpu_config.split(',')]

    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4090 防崩配置
    torch.backends.cudnn.enabled = False

    keypoint_scale = 1000.0
    batch_size = 64

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
            batch_size=batch_size # 优先使用命令行传入的 batch_size
        )

        # 3. 使用 sklearn 将 test_dataset 一分为二 (50% 验证，50% 测试)
        val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)

        val_loader = make_dataloader(
            val_data, is_training=False, generator=rng_generator, batch_size= batch_size // 2
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

    # ============================== #
    # 模型初始化
    # ============================== #
    model = posenet()
    # model.apply(weights_init) # 断点续训时会覆盖权重，放后面判断
    model = model.to(device)

    # ========== 极简模型统计（只要这几行） ==========
    # 1. 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. 计算FLOPs
    try:
        from thop import profile
        import copy

        # 创建正确大小的测试输入
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        test_input = torch.randn(1, 3, 114, 10).to(device)

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

    criterion_L2 = nn.MSELoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + 1 - 20) / float(30 + 1)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.5)

    # ============================== #
    # 【新增】断点续训加载
    # ============================== #
    start_epoch = 0
    pck_50_overall_max = 0
    train_mean_loss_iter = []

    if os.path.exists(resume_checkpoint_path):
        print(f"🔄 检测到训练存档，正在恢复: {resume_checkpoint_path}")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            pck_50_overall_max = checkpoint.get('best_pck50', 0)
            train_mean_loss_iter = checkpoint.get('loss_history', [])
            print(f"✅ 成功恢复！从 Epoch {start_epoch + 1} 继续训练")
        except Exception as e:
            print(f"⚠️ 存档读取失败 ({e})，重新初始化模型")
            model.apply(weights_init)
    else:
        print("✨ 未发现存档，开始新训练")
        model.apply(weights_init)

    # ============================== #
    # 训练循环
    # ============================== #
    num_epochs = 50

    if start_epoch < num_epochs:
        print("Starting training...")

        for epoch_index in range(start_epoch, num_epochs):
            model.train()
            train_loss_iter = []

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch_index + 1}/{num_epochs} [Train]', leave=False)

            for batch_data in train_pbar:
                # 1. 提取输入输出，直接强制转换为 [Batch, 17, 3] 形状
                batch_x = batch_data["input_wifi-csi"].float().to(device, non_blocking=True)
                batch_y = batch_data["output"].float().to(device, non_blocking=True)

                # 2. 前向传播
                pred_keypoint, _ = model(batch_x)

                # 3. 直接计算纯净 MSE 损失
                # loss = criterion_L2(pred_keypoint, batch_y)
                valid_mask = (torch.sum(torch.abs(batch_y), dim=-1, keepdim=True) > 1e-5).float()

                # 3. 计算带 Mask 的 MSE 损失 (只惩罚真实存在的点)
                loss = criterion_L2(pred_keypoint * valid_mask, batch_y * valid_mask)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_iter.append(loss.item())

                lr = scheduler.get_last_lr()[0]
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.5f}'})

            scheduler.step()
            train_mean_loss = np.mean(train_loss_iter) if train_loss_iter else 0
            train_mean_loss_iter.append(train_mean_loss)
            print('end of the epoch: %d, with loss: %.4f' % (epoch_index, train_mean_loss))

            # --- Validation Phase ---
            # --- Validation Phase ---
            model.eval()
            valid_loss_iter = []
            pck_50_iter = []
            pck_20_iter = []
            mpe_iter = []

            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch_index + 1}/{num_epochs} [Valid]', leave=False)

            with torch.no_grad():
                for batch_data in val_pbar:
                    # 提取数据
                    csi_data = batch_data["input_wifi-csi"].float().to(device)
                    gt_keypoints = batch_data["output"].float().to(device)

                    # 预测
                    pred_keypoint, _ = model(csi_data)

                    # MSE 损失
                    # loss = criterion_L2(pred_keypoint, gt_keypoints)
                    valid_mask = (torch.sum(torch.abs(gt_keypoints), dim=-1, keepdim=True) > 1e-5).float()

                    # 3. 计算带 Mask 的 MSE 损失 (只惩罚真实存在的点)
                    loss = criterion_L2(pred_keypoint * valid_mask, gt_keypoints * valid_mask)
                    valid_loss_iter.append(loss.item())

                    # MPE 误差计算
                    mpe = mean_keypoint_error(pred_keypoint, gt_keypoints, keypoint_scale)
                    mpe_iter.append(mpe)

                    # �� 纯 3D PCK 评估
                    pck_results, _ = compute_3d_pck(pred_keypoint, gt_keypoints, thresholds=[0.2, 0.5])
                    pck_50_iter.append(pck_results[0.5])
                    pck_20_iter.append(pck_results[0.2])

            valid_mean_loss = np.mean(valid_loss_iter)
            valid_mean_mpe = np.mean(mpe_iter)
            pck_50_overall = np.mean(pck_50_iter)
            pck_20_overall = np.mean(pck_20_iter)

            print('validation result with loss: %.3f, mpe: %.3f, pck_50: %.3f, pck_20: %.3f' %
                  (valid_mean_loss, valid_mean_mpe, pck_50_overall, pck_20_overall))

            # ============================== #
            # 【新增】保存断点 (覆盖式保存)
            # ============================== #
            checkpoint = {
                'epoch': epoch_index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_pck50': max(pck_50_overall, pck_50_overall_max),
                'loss_history': train_mean_loss_iter
            }
            torch.save(checkpoint, resume_checkpoint_path)
            print(f"💾 进度已保存至: {resume_checkpoint_path}")

            # 保存最佳模型
            if pck_50_overall > pck_50_overall_max:
                print('saving the model at the end of epoch %d with pck_50: %.3f' %
                      (epoch_index, pck_50_overall))
                torch.save(model, best_model_path)
                pck_50_overall_max = pck_50_overall

            if (epoch_index + 1) % 10 == 0:
                torch.save(model, latest_model_path)

            if (epoch_index + 1) % 50 == 0:
                print('the train loss for the first %.1f epoch is' % (epoch_index))
                print(train_mean_loss_iter)

        print("\nTraining completed!")
    else:
        print("✅ 检测到训练已完成，直接进入测试阶段。")

    # ============================== #
    # 测试阶段
    # ============================== #
    print("\nStarting testing...")

    model = torch.load(best_model_path, map_location=device, weights_only=False)
    model = model.to(device).eval()

    test_loss_iter = []
    pck_50_iter = []
    pck_40_iter = []
    pck_30_iter = []
    pck_20_iter = []
    pck_10_iter = []
    pck_5_iter = []
    mpe_iter = []

    # 保存所有预测和真实关键点用于视频生成
    all_pred_keypoints = []
    all_true_keypoints = []

    test_pbar = tqdm(test_loader, desc='Testing', unit='batch')

    with torch.no_grad():
        for batch_data in test_pbar:
            # 提取数据
            csi_data = batch_data["input_wifi-csi"].float().to(device)
            keypoints = batch_data["output"].float().to(device)

            # 预测
            pred_keypoint, _ = model(csi_data)

            # 损失与误差
            # loss = criterion_L2(pred_keypoint, keypoints)
            valid_mask = (torch.sum(torch.abs(keypoints), dim=-1, keepdim=True) > 1e-5).float()

            # 3. 计算带 Mask 的 MSE 损失 (只惩罚真实存在的点)
            loss = criterion_L2(pred_keypoint * valid_mask, keypoints * valid_mask)
            test_loss_iter.append(loss.item())

            mpe = mean_keypoint_error(pred_keypoint, keypoints, keypoint_scale)
            mpe_iter.append(mpe)

            # 保存供后续视频使用
            all_pred_keypoints.append(pred_keypoint.cpu().numpy())
            all_true_keypoints.append(keypoints.cpu().numpy())

            # �� 纯 3D PCK 评估 (所有阈值)
            pck_res, _ = compute_3d_pck(pred_keypoint, keypoints, thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
            pck_50_iter.append(pck_res[0.5])
            pck_40_iter.append(pck_res[0.4])
            pck_30_iter.append(pck_res[0.3])
            pck_20_iter.append(pck_res[0.2])
            pck_10_iter.append(pck_res[0.1])
            pck_5_iter.append(pck_res[0.05])

            test_pbar.set_postfix({'test_loss': f'{loss.item():.4f}'})

    test_pbar.close()

    test_mean_loss = np.mean(test_loss_iter)
    test_mean_mpe = np.mean(mpe_iter)
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)

    pck_50_overall = pck_50[-1] if len(pck_50.shape) > 0 else pck_50
    pck_40_overall = pck_40[-1] if len(pck_40.shape) > 0 else pck_40
    pck_30_overall = pck_30[-1] if len(pck_30.shape) > 0 else pck_30
    pck_20_overall = pck_20[-1] if len(pck_20.shape) > 0 else pck_20
    pck_10_overall = pck_10[-1] if len(pck_10.shape) > 0 else pck_10
    pck_5_overall = pck_5[-1] if len(pck_5.shape) > 0 else pck_5

    print('test result with loss: %.3f, mpe: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, '
          'pck_20: %.3f, pck_10: %.3f, pck_5: %.3f' %
          (test_mean_loss, test_mean_mpe, pck_50_overall, pck_40_overall, pck_30_overall,
           pck_20_overall, pck_10_overall, pck_5_overall))

    # 保存测试结果
    test_results = {
        'test_loss': float(test_mean_loss),
        'test_mpe': float(test_mean_mpe),
        'pck_50': float(pck_50_overall),
        'pck_40': float(pck_40_overall),
        'pck_30': float(pck_30_overall),
        'pck_20': float(pck_20_overall),
        'pck_10': float(pck_10_overall),
        'pck_5': float(pck_5_overall),
        'detailed_pck': {
            'pck_50_per_joint': pck_50.tolist() if hasattr(pck_50, 'tolist') else [float(pck_50)],
            'pck_40_per_joint': pck_40.tolist() if hasattr(pck_40, 'tolist') else [float(pck_40)],
            'pck_30_per_joint': pck_30.tolist() if hasattr(pck_30, 'tolist') else [float(pck_30)],
            'pck_20_per_joint': pck_20.tolist() if hasattr(pck_20, 'tolist') else [float(pck_20)],
            'pck_10_per_joint': pck_10.tolist() if hasattr(pck_10, 'tolist') else [float(pck_10)],
            'pck_5_per_joint': pck_5.tolist() if hasattr(pck_5, 'tolist') else [float(pck_5)]
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'keypoint_scale': keypoint_scale,
            'learning_rate': 0.001
        }
    }

    import json
    with open(test_results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n测试结果已保存到: {test_results_file}")

    # ============================== #
    # 生成视频 (完整保留你的逻辑)
    # ============================== #
    if len(all_pred_keypoints) > 0:
        print("\n生成可视化视频...")

        # 转换为numpy数组
        all_pred_keypoints = np.vstack(all_pred_keypoints)
        all_true_keypoints = np.vstack(all_true_keypoints)

        # 选择一部分帧来生成视频（例如前720帧）
        frames_to_animate = min(720, len(all_pred_keypoints))

        # 生成真实姿态视频
        print("生成真实姿态视频...")
        create_pose_animation_opencv(
            all_true_keypoints[:frames_to_animate],
            output_file=true_poses_video,
            keypoint_scale=keypoint_scale
        )

        # 生成预测姿态视频
        print("生成预测姿态视频...")
        create_pose_animation_opencv(
            all_pred_keypoints[:frames_to_animate],
            output_file=predicted_poses_video,
            keypoint_scale=keypoint_scale
        )

        # 生成对比视频
        print("生成对比视频...")
        create_side_by_side_video_opencv(
            all_true_keypoints[:frames_to_animate],
            all_pred_keypoints[:frames_to_animate],
            output_file=comparison_video,
            keypoint_scale=keypoint_scale,
            fps=30
        )

        print("\n视频生成完成！")
        print(f"真实姿态视频: {true_poses_video}")
        print(f"预测姿态视频: {predicted_poses_video}")
        print(f"对比视频: {comparison_video}")

    # 生成训练曲线图
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_mean_loss_iter, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(training_curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存到: {training_curves_file}")
    except Exception as e:
        print(f"保存训练曲线失败: {e}")

    print(f"\n所有输出文件已保存到: {output_dir}")
    print("训练完成！")


if __name__ == "__main__":
    train_posenet()