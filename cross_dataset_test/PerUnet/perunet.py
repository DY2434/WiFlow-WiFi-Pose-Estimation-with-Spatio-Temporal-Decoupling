import torch
import torch.nn as nn
import torch.nn.functional as F
# 别忘了 pip install performer-pytorch
from performer_pytorch import Performer
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hdf5storage
import pickle
import yaml
import seaborn
from sklearn.model_selection import train_test_split
from mmfi import make_dataset, make_dataloader  # 确保 mmfi.py 在同目录下

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
# 1. 定义要保留的关键点
# ============================== #
KEEP_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
KEYPOINT_MAPPING = {old_idx: new_idx for new_idx, old_idx in enumerate(KEEP_KEYPOINTS)}

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

class DoubleConv(nn.Module):
    """Unet 基础双层卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class PerUnet_Baseline(nn.Module):
    """
    遵循严密物理映射的修改版 PerUnet
    完全符合 [B, 3, 114, 10] -> [B, 1140, 24, 24] -> [B, 17, 3] 的重塑逻辑
    """

    def __init__(self):
        super(PerUnet_Baseline, self).__init__()

        # === 1. Unet 编码器 (Encoder) ===
        # 输入变成了 1140 维，所以 inc 的输入也要是 1140
        self.inc = DoubleConv(1140, 600)  # 这里把 1140 降维/整合到 600
        self.pool1 = nn.MaxPool2d(2)  # 24x24 -> 12x12

        self.down1 = DoubleConv(600, 1200)
        self.pool2 = nn.MaxPool2d(2)  # 12x12 -> 6x6

        self.down2 = DoubleConv(1200, 2400)
        self.pool3 = nn.MaxPool2d(2)  # 6x6 -> 3x3

        # 底部瓶颈层
        self.bot = DoubleConv(2400, 2400)

        # === 2. 跳跃连接 1 中的 Performer ===
        # SC1 处理浅层特征，进入 inc 后通道数变成了 600
        self.performer_sc1 = Performer(
            dim=600,
            depth=3,
            heads=4,
            dim_head=64,
            causal=False
        )

        # === 3. Unet 解码器 (Decoder) ===
        # Up 1: 瓶颈层 2400 上采样 -> 1200, 拼接 down2 的 2400 -> 3600
        # �� 修正：ConvTranspose2d 的输入通道必须和 bot 的输出 2400 一致
        self.up1 = nn.ConvTranspose2d(2400, 1200, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(3600, 1200)

        # Up 2: 1200 上采样 -> 600, 拼接 down1 的 1200 -> 1800
        # �� 修正：ConvTranspose2d 的输入通道必须和 up_conv1 的输出 1200 一致
        self.up2 = nn.ConvTranspose2d(1200, 600, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(1800, 600)

        # Up 3: 600 上采样 -> 300, 拼接 Performer 出来的 600 -> 900
        # �� 修正：这里调成 300 方便后续处理，拼接后变 900
        self.up3 = nn.ConvTranspose2d(600, 300, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(900, 285)

        # 4. Scale Matching (尺度匹配)
        # self.scale_match = nn.Sequential(
        #     nn.Conv2d(285, 150, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # �� 确保输出通道为3 (x, y, z)
        #     nn.Conv2d(150, 3, kernel_size=3, padding=1)
        # )
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((17, 1))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 将 24x24 压缩成全局特征向量

        self.coordinate_regressor = nn.Sequential(
            nn.Linear(285, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(512, 17 * 3)  # 17个点 * 3个坐标(x,y,z) = 51 个输出
        )

    def forward(self, x):
        b = x.shape[0]

        # 如果你在外面没做维度转换，这段解开注释：
        x = x.permute(0, 3, 2, 1)
        x = x.contiguous().view(b, 1140, 1, 3)

        # --- 1. Patch Magnification ---
        # �� 如果这里报错 "Input size must have spatial dimensions"，
        # 请确保传进来的 x 是 4D 张量，比如上面转换好的 [B, 1140, 1, 3]
        x = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)

        # --- 2. Encoder ---
        x1 = self.inc(x)  # [B, 600, 24, 24]
        x2 = self.down1(self.pool1(x1))  # [B, 1200, 12, 12]
        x3 = self.down2(self.pool2(x2))  # [B, 2400, 6, 6]
        bot = self.bot(self.pool3(x3))   # [B, 2400, 3, 3]

        # --- 3. Attention-Based Denoising (SC 1) ---
        _, c, h, w = x1.shape  # c = 600
        x1_flat = x1.view(b, c, -1).permute(0, 2, 1)  # [B, 576, 600]
        x1_att = self.performer_sc1(x1_flat)
        x1_att = x1_att.permute(0, 2, 1).view(b, c, h, w)  # [B, 600, 24, 24]

        # --- 4. Decoder & Skip Connections ---
        u3 = self.up1(bot)  # [B, 1200, 6, 6]
        u3 = torch.cat([u3, x3], dim=1)  # [B, 3600, 6, 6]
        u3 = self.up_conv1(u3)  # [B, 1200, 6, 6]

        u2 = self.up2(u3)  # [B, 600, 12, 12]
        u2 = torch.cat([u2, x2], dim=1)  # [B, 1800, 12, 12]
        u2 = self.up_conv2(u2)  # [B, 600, 12, 12]

        u1 = self.up3(u2)  # [B, 300, 24, 24]
        u1 = torch.cat([u1, x1_att], dim=1)  # [B, 900, 24, 24]
        u1 = self.up_conv3(u1)  # [B, 285, 24, 24]

        # --- 5. 尺度匹配与坐标回归 ---
        # out = self.scale_match(u1)     # [B, 3, 24, 24]
        # out = self.adaptive_pool(out)  # [B, 3, 17, 1]
        #
        # # �� 最终重塑维度以对齐 3D 坐标标签 [B, 17, 3]
        # out = out.squeeze(-1)          # [B, 3, 17]
        # out = out.transpose(1, 2)      # [B, 17, 3]
        out = self.global_pool(u1)  # [B, 285, 1, 1]
        out = torch.flatten(out, 1)  # [B, 285]
        out = self.coordinate_regressor(out)  # [B, 51]
        out = out.view(b, 17, 3)  # [B, 17, 3]

        return out

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


def calculate_pck_metrics(pred, target, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    计算纯 3D 空间的 PCK (无根节点对齐) - 终极修复版
    完全兼容原代码中返回字典的调用方式
    """
    batch_size = pred.shape[0]

    # 计算 3D 空间欧氏距离
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))  # [batch_size, 17]

    # 基准距离：Left Hip(1) 到 Right Shoulder(11)
    IDX1, IDX2 = 1, 11
    scale = torch.sqrt(torch.sum((target[:, IDX1, :] - target[:, IDX2, :]) ** 2, dim=1))
    scale = torch.clamp(scale, min=1e-5)  # 防止除以 0

    # 计算归一化距离
    normalized_distances = distances / scale.unsqueeze(1)  # [batch_size, 17]

    pck_results = {}
    pck_per_joint = {}  # 占位返回值，以兼容 "pck_results, _ = ..." 的调用

    for thr in thresholds:
        correct_mask = (normalized_distances <= thr).float()
        # 计算整个 batch 的平均正确率，转为百分制 numpy
        pck_overall = correct_mask.mean().item() * 100
        pck_results[thr] = np.array(pck_overall)  # 包装成 numpy，防止 test_pcks[thr].append 报错

    return pck_results, pck_per_joint

# ============================== #
# 姿态可视化相关设置
# ============================== #

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

# ============================== #
# 主训练函数（使用PAM格式）
# ============================== #
def train(train_loader, val_loader, test_loader,
                           batch_size=32, num_epochs=50, learning_rate=0.001,
                           keypoint_scale=1000.0, output_dir="perunet2"):
    """训练使用PAM格式标签的perunet 模型（与原始代码保持一致）"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    print("初始化perunet模型（PAM版本）...")
    perunet = PerUnet_Baseline().to(device)

    # ========== 极简模型统计（只要这几行） ==========
    # 1. 计算总参数量
    total_params = sum(p.numel() for p in perunet.parameters() if p.requires_grad)
    print(f"\n📊 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. 计算FLOPs
    try:
        from thop import profile
        import copy

        # 创建正确大小的测试输入
        model_copy = copy.deepcopy(perunet)
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

    # 损失函数和优化器
    criterion_L2 = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(perunet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    # ============================== #
    # 新增：断点续训加载逻辑
    # ============================== #
    start_epoch = 0
    train_losses = []
    checkpoint_path = os.path.join('weights2', 'latest_checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] 发现断点文件 {checkpoint_path}，正在恢复训练...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        perunet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        print(f"[INFO] 成功恢复！将从 Epoch {start_epoch + 1} 继续训练...\n")
    else:
        print("\n[INFO] 未找到断点文件，将从头开始训练...")

    # ============================== #
    # 训练循环
    # ============================== #
    print("开始训练...")
    perunet.train()

    # �� 注意：tqdm 和内部循环必须放在 epoch 循环里面
    for epoch_index in range(start_epoch, num_epochs):
        start = time.time()
        epoch_losses = []

        # 训练批次循环
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch_index + 1}/{num_epochs}", leave=False)

        for batch_index, batch_data in enumerate(train_loop):
            # 1. 提取输入输出，处理 NaN 毒数据
            csi_data = batch_data["input_wifi-csi"].float().to(device, non_blocking=True)

            # 处理 NaN 毒数据
            if torch.isnan(csi_data).any() or torch.isinf(csi_data).any():
                print(f"\n[警告] Batch {batch_index}: CSI 数据异常，跳过！")
                continue

            # 真实标签 [B, 17, 3]
            true_keypoints = batch_data["output"].float().to(device, non_blocking=True)

            # 2. 前向传播
            pred_keypoints = perunet(csi_data)

            # 3. 纯 MSE 损失计算
            # loss = criterion_L2(pred_keypoints, true_keypoints)
            valid_mask = (torch.sum(torch.abs(true_keypoints), dim=-1, keepdim=True) > 1e-5).float()

            # 把预测值和真实值都乘上 mask，无效点相减就是 0，不会产生梯度惩罚
            loss = criterion_L2(pred_keypoints * valid_mask, true_keypoints * valid_mask)

            epoch_losses.append(loss.item())

            # 4. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        # 计算epoch平均损失
        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)

        endl = time.time()
        print(
            f'Epoch {epoch_index + 1}: Avg Loss: {avg_epoch_loss:.6f}, Costing time: {(endl - start) / 60:.2f} minutes')

        scheduler.step()

        # ============================== #
        # 每个 Epoch 结束后保存一次断点
        # ============================== #
        os.makedirs('weights2', exist_ok=True)
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': perunet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses
        }, checkpoint_path)
        print(f"[SAVE] 已保存 Epoch {epoch_index + 1} 的断点数据。")

    # 保存模型（与原始代码一致）
    os.makedirs('weights2', exist_ok=True)
    model_path = f'weights2/perunet2-{num_epochs}epochs.pkl'
    torch.save(perunet, model_path)
    print(f"模型已保存到 {model_path}")

    # ============================== #
    # 测试阶段（类似原始代码，但添加评估指标）
    # ============================== #
    print("\n开始测试...")
    perunet = perunet.to(device).eval()

    # 用于保存所有预测和真实坐标
    all_pred_coords = []
    all_true_coords = []

    # 评估指标（额外添加的，原始代码没有）
    test_mpes = []
    test_pcks = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: []}

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=False)

        for batch_index, batch_data in enumerate(test_loop):
            # 1. 提取原始数据，【直接送入设备】
            csi_data = batch_data["input_wifi-csi"].float().to(device, non_blocking=True)

            # 真实标签 [B, 17, 3]
            true_keypoints = batch_data["output"].float().to(device, non_blocking=True)

            # 2. 前向传播
            pred_keypoints = perunet(csi_data)

            # 3. 提取坐标用于视频生成
            all_pred_coords.extend(pred_keypoints.cpu().numpy())
            all_true_coords.extend(true_keypoints.cpu().numpy())

            # 4. 计算 MPE 误差
            mpe = mean_keypoint_error(pred_keypoints, true_keypoints, keypoint_scale)
            test_mpes.append(mpe)

            # 5. 计算纯 3D PCK
            pck_results, _ = calculate_pck_metrics(pred_keypoints, true_keypoints, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])

            for threshold, value in pck_results.items():
                test_pcks[threshold].append(value)

            test_loop.set_postfix(mpe=f"{mpe:.2f}", pck20=f"{pck_results[0.2]:.2f}")

            # 可选：显示第一个批次的预测
            if batch_index == 0:
                print(f"\n第一个批次的预测示例:")
                print(f"  预测坐标范围 - X: [{pred_keypoints[0, :, 0].min():.2f}, {pred_keypoints[0, :, 0].max():.2f}]")
                print(f"  预测坐标范围 - Y: [{pred_keypoints[0, :, 1].min():.2f}, {pred_keypoints[0, :, 1].max():.2f}]")
                print(f"  预测坐标范围 - Z: [{pred_keypoints[0, :, 2].min():.2f}, {pred_keypoints[0, :, 2].max():.2f}]")

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

    return perunet, test_results


def main():
    # 训练参数
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

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
    model, test_results = train(
        train_loader, val_loader, test_loader,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        keypoint_scale=1000.0,
        output_dir="perunet2"
    )

    print("\n训练完成！")


if __name__ == "__main__":
    main()

