import torch
import os
import argparse
import psutil
import numpy as np
import gc
import random
import traceback
from torch.utils.data import DataLoader
from config import Config
from train import train_pose_model
from dataset import PreprocessedCSIKeypointsDataset, create_preprocessed_train_val_test_loaders

# 警告过滤器通常保留，因为它们设置了环境
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    output_dir = "test"

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='WiFi Pose Estimation Training')
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Output directory')
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='Use data augmentation')
    parser.add_argument('--data_dir', type=str, default='preprocessed_csi_data',
                        help='Data directory')
    args = parser.parse_args()

    # 创建配置
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.OUTPUT_DIR = args.output_dir
    config.DATA_DIR = args.data_dir

    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

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

    # 预处理数据目录
    preprocessed_dir = "../preprocessed_csi_data" # 数据路径

    # 检查预处理数据是否存在
    if not os.path.exists(preprocessed_dir) or not os.path.exists(os.path.join(preprocessed_dir, "csi_windows.npy")):
        print(f"错误: 未找到预处理数据 {preprocessed_dir}")
        print("请先运行 preprocess_csi_data.py 脚本生成预处理数据")
        return

    # 使用预处理数据创建数据集
    try:
        print(f"正在加载预处理数据...")
        dataset = PreprocessedCSIKeypointsDataset(
            data_dir=preprocessed_dir,
            keypoint_scale=1000.0,
            enable_temporal_clean=True
        )

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_preprocessed_train_val_test_loaders(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=0
        )

        # 查看一个批次的数据形状
        for csi_batch, keypoints_batch in train_loader:
            print(f"CSI数据形状: {csi_batch.shape}")
            print(f"关键点数据形状: {keypoints_batch.shape}")
            if torch.isnan(csi_batch).any() or torch.isinf(csi_batch).any():
                print("警告: CSI数据包含NaN或Inf")
            if torch.isnan(keypoints_batch).any() or torch.isinf(keypoints_batch).any():
                print("警告: 关键点数据包含NaN或Inf")
            break

    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 训练参数
    n_epochs = args.epochs
    # 改成3-5比较好，在epoch15之后过拟合精度下降
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
        # 修复：正确接收所有返回值
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

# 命令行入口点
if __name__ == "__main__":
    # 设置多进程启动方法
    if torch.cuda.is_available():
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    main()