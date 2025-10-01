import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from tqdm import tqdm

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
    for i in range(15):
        columns.extend([f"true_kp{i}_x", f"true_kp{i}_y", f"pred_kp{i}_x", f"pred_kp{i}_y"])

    data = []
    for i in range(n_samples):
        row = []
        true_kp = true_keypoints[i].reshape(15, 2) * keypoint_scale
        pred_kp = pred_keypoints[i].reshape(15, 2) * keypoint_scale

        for j in range(15):
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
    true_kp = np.array(true_keypoints[:n_samples]).reshape(n_samples, 15, 2) * keypoint_scale
    pred_kp = np.array(pred_keypoints[:n_samples]).reshape(n_samples, 15, 2) * keypoint_scale

    distances = np.sqrt(np.sum((true_kp - pred_kp) ** 2, axis=2))

    keypoint_stats = []
    for i in range(15):
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