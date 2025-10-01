import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
import pickle
import random
from typing import Optional, Tuple, List

# ============================== #
# 1. 定义要保留的关键点
# ============================== #
KEEP_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
KEYPOINT_MAPPING = {old_idx: new_idx for new_idx, old_idx in enumerate(KEEP_KEYPOINTS)}

class PreprocessedCSIKeypointsDataset(Dataset):
    """
    使用预处理后的CSI数据和关键点的数据集
    自动检测并使用NPY格式（如果存在）以提升速度
    """

    def __init__(self, data_dir, keypoint_scale=1000.0, transform=None, enable_temporal_clean=True):
        # 加载CSI窗口数据
        self.csi_windows = np.load(os.path.join(data_dir, "csi_windows.npy"))

        # 加载窗口元数据
        window_info = np.load(os.path.join(data_dir, "window_info.npz"))
        self.window_to_file = window_info['window_to_file']
        self.window_to_frame = window_info['window_to_frame']

        # 加载文件元数据
        file_info = np.load(os.path.join(data_dir, "file_info.npz"), allow_pickle=True)
        self.keypoints_files = file_info['keypoints_files']
        self.file_ids = file_info['file_ids']
        self.window_ranges = file_info['window_ranges']

        # 加载配置
        config = np.load(os.path.join(data_dir, "config.npz"))
        self.window_size = config['window_size']
        self.stride = config['stride']

        self.keypoint_scale = keypoint_scale
        self.transform = transform
        self.enable_temporal_clean = enable_temporal_clean

        # 检测并加载NPY格式的关键点（如果存在）
        self.use_npy_mode = False
        self.all_keypoints = None
        self.file_mappings = None

        # 检查是否存在处理好的NPY数据
        all_keypoints_path = os.path.join(data_dir, "all_keypoints.npy")
        file_mappings_path = os.path.join(data_dir, "file_mappings.pkl")

        if os.path.exists(all_keypoints_path) and os.path.exists(file_mappings_path):
            print("检测到NPY格式关键点，使用快速加载模式...")
            self.all_keypoints = np.load(all_keypoints_path)
            with open(file_mappings_path, 'rb') as f:
                self.file_mappings = pickle.load(f)
            self.use_npy_mode = True
            print(f"加载了 {len(self.csi_windows)} 个CSI窗口（NPY快速模式）")
            print(f"关键点数据形状: {self.all_keypoints.shape}")
        else:
            # 回退到CSV模式
            print("未检测到NPY格式关键点，使用CSV模式（较慢）...")
            print("建议运行 preprocess_keypoints_to_npy.py 以加速训练")

            # 缓存清理后的关键点序列
            self._cleaned_keypoints_cache = {}
            self._raw_keypoints_cache = {}
            self._cache_size = 10

            print(f"加载了 {len(self.csi_windows)} 个CSI窗口，来自 {len(self.keypoints_files)} 个文件")

        print(f"零值清理: {'启用' if enable_temporal_clean else '禁用'}")

    def __len__(self):
        return len(self.csi_windows)

    def _get_keypoint_npy(self, idx):
        """NPY模式：从预处理的NPY数据获取关键点"""
        file_idx = self.window_to_file[idx]
        frame_idx = self.window_to_frame[idx]

        # 获取对应的CSV文件路径
        csv_file = self.keypoints_files[file_idx]

        # 从映射中获取数据索引
        if csv_file in self.file_mappings:
            mapping = self.file_mappings[csv_file]
            global_frame_idx = mapping['start_idx'] + frame_idx

            if global_frame_idx < len(self.all_keypoints):
                keypoint = self.all_keypoints[global_frame_idx]

                # 如果需要清理零值
                if self.enable_temporal_clean:
                    keypoint = self._clean_single_frame_zeros(keypoint)

                return keypoint

        # 如果找不到，返回零
        return np.zeros((15, 2), dtype=np.float32)

    def _clean_single_frame_zeros(self, keypoint):
        """清理单帧中的零值关键点（简单处理）"""
        cleaned = keypoint.copy()

        # 找到非零关键点的平均位置
        non_zero_mask = (keypoint[:, 0] != 0) | (keypoint[:, 1] != 0)

        if non_zero_mask.any():
            # 用非零关键点的平均值替代零值
            mean_pos = keypoint[non_zero_mask].mean(axis=0)
            zero_indices = np.where(~non_zero_mask)[0]

            for idx in zero_indices:
                cleaned[idx] = mean_pos

        return cleaned

    def _load_raw_keypoints(self, file_idx):
        """CSV模式：加载原始关键点序列"""
        if file_idx in self._raw_keypoints_cache:
            return self._raw_keypoints_cache[file_idx]

        # 缓存管理
        if len(self._raw_keypoints_cache) >= self._cache_size:
            oldest_key = next(iter(self._raw_keypoints_cache))
            del self._raw_keypoints_cache[oldest_key]
            if oldest_key in self._cleaned_keypoints_cache:
                del self._cleaned_keypoints_cache[oldest_key]

        keypoints_file = self.keypoints_files[file_idx]

        # 读取关键点数据
        keypoints_data = pd.read_csv(keypoints_file, header=0).values

        # 处理数据格式
        if keypoints_data.shape[1] > 50:
            keypoints_data = keypoints_data[:, -50:]

        # 归一化
        keypoints_data = keypoints_data.astype(np.float32) / self.keypoint_scale

        # 重塑为 (num_frames, 25, 2)
        num_frames = len(keypoints_data)
        keypoints_reshaped = keypoints_data.reshape(num_frames, 25, 2)

        # 只保留需要的15个关键点
        filtered_keypoints = keypoints_reshaped[:, KEEP_KEYPOINTS, :]

        # 缓存
        self._raw_keypoints_cache[file_idx] = filtered_keypoints

        return filtered_keypoints

    def _clean_zero_keypoints(self, keypoints_sequence):
        """CSV模式：只处理零值异常点，通过插值修复"""
        num_frames, num_keypoints, _ = keypoints_sequence.shape
        coords = keypoints_sequence.copy()

        for kp_idx in range(num_keypoints):
            # 找到所有零值点
            zero_indices = []
            for t in range(num_frames):
                if coords[t, kp_idx, 0] == 0 and coords[t, kp_idx, 1] == 0:
                    zero_indices.append(t)

            # 修复零值点
            for t in zero_indices:
                # 找到前后最近的非零点
                valid_prev = None
                valid_next = None

                # 向前搜索非零点
                for prev_t in range(t - 1, -1, -1):
                    if not (coords[prev_t, kp_idx, 0] == 0 and coords[prev_t, kp_idx, 1] == 0):
                        valid_prev = prev_t
                        break

                # 向后搜索非零点
                for next_t in range(t + 1, num_frames):
                    if not (coords[next_t, kp_idx, 0] == 0 and coords[next_t, kp_idx, 1] == 0):
                        valid_next = next_t
                        break

                # 修复策略
                if valid_prev is not None and valid_next is not None:
                    # 线性插值
                    alpha = (t - valid_prev) / (valid_next - valid_prev)
                    coords[t, kp_idx] = (1 - alpha) * coords[valid_prev, kp_idx] + \
                                        alpha * coords[valid_next, kp_idx]
                elif valid_prev is not None:
                    # 使用前一个有效点
                    coords[t, kp_idx] = coords[valid_prev, kp_idx]
                elif valid_next is not None:
                    # 使用后一个有效点
                    coords[t, kp_idx] = coords[valid_next, kp_idx]

        return coords

    def _load_and_clean_keypoints(self, file_idx):
        """CSV模式：加载并清理关键点序列"""
        if not self.enable_temporal_clean:
            return self._load_raw_keypoints(file_idx)

        if file_idx in self._cleaned_keypoints_cache:
            return self._cleaned_keypoints_cache[file_idx]

        raw_keypoints = self._load_raw_keypoints(file_idx)
        cleaned_keypoints = self._clean_zero_keypoints(raw_keypoints)
        self._cleaned_keypoints_cache[file_idx] = cleaned_keypoints

        return cleaned_keypoints

    def __getitem__(self, idx):
        # 获取CSI窗口
        csi_window = self.csi_windows[idx]

        # 根据模式获取关键点
        if self.use_npy_mode:
            # NPY快速模式
            keypoint = self._get_keypoint_npy(idx)
        else:
            # CSV模式
            file_idx = self.window_to_file[idx]
            frame_idx = self.window_to_frame[idx]

            # 获取清理后的关键点序列
            keypoints_sequence = self._load_and_clean_keypoints(file_idx)

            # 提取对应帧的关键点
            keypoint = keypoints_sequence[frame_idx]

        # 转换为张量
        csi_tensor = torch.from_numpy(csi_window).float()
        keypoint_tensor = torch.from_numpy(keypoint).float()

        # 应用变换
        if self.transform:
            csi_tensor = self.transform(csi_tensor)

        return csi_tensor, keypoint_tensor

    def get_file_indices(self):
        """获取所有文件索引"""
        return list(range(len(self.keypoints_files)))

    def get_samples_from_file(self, file_idx):
        """获取指定文件的所有样本索引"""
        start_idx, end_idx = self.window_ranges[file_idx]
        return list(range(start_idx, end_idx))


def create_preprocessed_train_val_test_loaders(dataset, batch_size=64, num_workers=0, random_seed=42):
    """按文件级别划分预处理后的数据集"""
    # 设置随机种子
    random.seed(random_seed)

    # 获取所有文件索引
    file_indices = dataset.get_file_indices()
    total_files = len(file_indices)

    # 随机打乱文件顺序
    random.shuffle(file_indices)

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
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader