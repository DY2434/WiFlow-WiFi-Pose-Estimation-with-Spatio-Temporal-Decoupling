import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import copy
import pandas as pd
import cv2
from tqdm import tqdm
import traceback # 用于错误报告
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
from models.pose_model import WiFlowPoseModel
from losses.pose_loss import PoseLoss
from utils.metrics import calculate_pck, calculate_mpjpe
from visualization import create_side_by_side_video_opencv, save_all_predictions, calculate_keypoint_errors

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
    model = WiFlowPoseModel(dropout=0.5).to(device)


    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print("使用DataParallel")

    # 混合精度训练
    scaler = GradScaler()

    # 使用新的简化损失函数
    criterion = PoseLoss(
        position_weight=1.0,
        bone_weight=0.2,  # 增加骨骼长度权重
        loss_type='smooth_l1'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-5,  # 增加权重衰减
        betas=(0.9, 0.999)
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        # verbose=True,
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
    best_val_mpe = float('inf')  # MPE越小越好，所以初始化为无穷大
    patience_counter = 0
    best_model = None
    best_epoch = 0
    best_val_metrics = {'loss': float('inf'), 'mpe': float('inf'), 'pck': 0.0}

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

    for epoch in range(n_epochs):
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

        for batch_idx, (batch_x, batch_y) in enumerate(train_loop):
            try:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

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
                    mpe = calculate_mpjpe(outputs.detach(), batch_y)
                    pck_results = calculate_pck(outputs.detach(), batch_y, thresholds=[0.2, 0.5])
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
            for batch_idx, (batch_x, batch_y) in enumerate(val_loop):
                try:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    outputs = model(batch_x)
                    loss, loss_dict = criterion(outputs, batch_y)

                    mpe = calculate_mpjpe(outputs, batch_y)
                    pck_results = calculate_pck(outputs, batch_y, thresholds=[0.2, 0.5])
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
        scheduler.step(val_mpe)

        # 早停检查
        if val_mpe < best_val_mpe:  # MPE越小越好
            best_val_mpe = val_mpe
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
            print(f"  💾 保存最佳模型 (Epoch {best_epoch + 1}, MPE={val_mpe:.4f}) 到 {model_path}")
        else:
            patience_counter += 1
            print(f"  验证MPE未改善，耐心计数: {patience_counter}/{patience}")

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
        for batch_idx, (batch_x, batch_y) in enumerate(test_loop):
            try:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                outputs = model(batch_x)
                loss, _ = criterion(outputs, batch_y)

                mpe = calculate_mpjpe(outputs, batch_y)
                pck_results = calculate_pck(outputs, batch_y, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])
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