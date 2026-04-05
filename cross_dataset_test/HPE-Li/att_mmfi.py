#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 08:59:20 2024

@author: jackson-devworks
"""
import os

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch import nn
from tqdm import tqdm

from constant import experiment_config
from dataset_lib import make_dataloader, make_dataset
from model import *
from utils import calulate_error, compute_pck_pckh

with open(
    experiment_config["mmfi_config"], "r"
) as fd:  # change the .yaml file in your code.
    config = yaml.load(fd, Loader=yaml.FullLoader)

dataset_root = experiment_config["dataset_root"]
train_dataset, test_dataset = make_dataset(dataset_root, config)
rng_generator = torch.manual_seed(config["init_rand_seed"])
train_loader = make_dataloader(
    train_dataset,
    is_training=True,
    generator=rng_generator,
    **config["train_loader"],
)
val_data, test_data = train_test_split(
    test_dataset, test_size=0.5, random_state=41
)
val_loader = make_dataloader(
    val_data, is_training=False, generator=rng_generator, **config["val_loader"]
)
test_loader = make_dataloader(
    test_data,
    is_training=False,
    generator=rng_generator,
    **config["test_loader"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

metafi = DSKNetTransMMFI().to(device)

# ========== 极简模型统计 ==========
# 1. 计算总参数量
total_params = sum(p.numel() for p in metafi.parameters() if p.requires_grad)
print(f"\n�� 模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

# 2. 计算FLOPs
try:
    from thop import profile
    import copy

    model_copy = copy.deepcopy(metafi)
    model_copy.eval()

    # 这里传入原始 MMFi 的标准输入维度 [B, 3, 114, 10]
    # 如果你的 DSKNetTransMMFI 内部修改过输入期望，请把这里的形状改成对应的即可
    test_input = torch.randn(1, 3, 114, 10).to(device)

    with torch.no_grad():
        flops, params = profile(model_copy, inputs=(test_input,), verbose=False)
    print(f"�� 模型计算量: {flops / 1e6:.2f}M FLOPs")
    print(f"�� THOP参数量: {params:,} ({params / 1e6:.2f}M)\n")

    del model_copy, test_input

except ImportError:
    print("�� FLOPs计算需要安装: pip install thop\n")
except Exception as e:
    print(f"�� FLOPs计算出错: {e}\n")
# ==================================

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(metafi.parameters(), lr=0.001)
n_epochs = 20
n_epochs_decay = 30
epoch_count = 1
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: 1.0
    - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1),
)

num_epochs = experiment_config["epoch"]
pck_20_overall_max = 0
train_mean_loss_iter = []
valid_mean_loss_iter = []
time_iter = []

print(metafi._get_name() + "\n")
torch.cuda.empty_cache()

# ========================================== #
# �� 新增：全自动断点续训加载逻辑
# ========================================== #
start_epoch = 0
state_path = os.path.join(experiment_config["checkpoint"], "att_mmfi", "auto_resume_state.pth")

if os.path.exists(state_path):
    print(f"\n[INFO]  发现中断的训练记录，正在自动恢复...")
    # 读取状态字典
    state = torch.load(state_path, map_location=device, weights_only=False)

    # 恢复所有状态
    metafi.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    start_epoch = state['epoch'] + 1
    pck_20_overall_max = state['pck_20_overall_max']
    train_mean_loss_iter = state['train_mean_loss_iter']
    print(f"[INFO] ✅ 成功恢复！将精确从 Epoch {start_epoch} 继续训练...\n")
else:
    print("\n[INFO] ✨ 未找到断点记录，将从头开始训练...\n")
# ========================================== #

# 注意：循环起点改成了 start_epoch
for epoch_index in range(start_epoch, num_epochs):
    torch.cuda.empty_cache()
    loss = 0
    train_loss_iter = []
    metric = []
    metafi.train()
    relation_mean = []
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch_index}/{num_epochs} [Train]")
    for idx, data in enumerate(train_loop):
    # for idx, data in enumerate(train_loader):
        torch.cuda.empty_cache()
        csi_data = data["input_wifi-csi"]
        csi_data = csi_data.numpy()
        csi_data = torch.tensor(csi_data)
        csi_data = csi_data.cuda()
        csi_data = csi_data.type(torch.cuda.FloatTensor)
        keypoint = data["output"]
        keypoint = keypoint.cuda()

        xy_keypoint = keypoint[:, :, 0:2].cuda()
        confidence = keypoint[:, :, 2:3].cuda()

        pred_xy_keypoint, time = metafi(csi_data)  # b,2,17,17
        pred_xy_keypoint = pred_xy_keypoint.squeeze()
        loss = (
            criterion_L2(
                torch.mul(confidence, pred_xy_keypoint),
                torch.mul(confidence, xy_keypoint),
            )
            / 32
        )
        train_loss_iter.append(loss.cpu().detach().numpy())
        time_iter.append(time)
        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

        train_loop.set_postfix(loss=loss.item())

        # �� 修改 1：直接提取列表里的第一个纯数字，不要转成 numpy 数组
        lr = scheduler.get_last_lr()[0]

        # �� 修改 2：给 loss 加上 .item()，提取纯净的标量数值
        message = "(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) " % (
            epoch_index,
            idx * 32,
            lr,
            loss.item(),
        )
        # print(message)
    scheduler.step()
    sum_time = np.mean(time_iter)
    train_mean_loss = np.mean(train_loss_iter)
    train_mean_loss_iter.append(train_mean_loss)

    metafi.eval()
    valid_loss_iter = []
    # metric = []
    pck_50_iter = []
    pck_20_iter = []
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch_index}/{num_epochs} [Val]")
        for idx, data in enumerate(val_loop):
            torch.cuda.empty_cache()
            csi_data = data["input_wifi-csi"]
            csi_data = csi_data.numpy()
            csi_data = torch.tensor(csi_data)
            csi_data = csi_data.cuda()
            csi_data = csi_data.type(torch.cuda.FloatTensor)

            keypoint = data["output"]  # 17,3
            keypoint = keypoint.cuda()

            xy_keypoint = keypoint[:, :, 0:2].cuda()
            confidence = keypoint[:, :, 2:3].cuda()

            pred_xy_keypoint, time = metafi(csi_data)  # 4,2,17,17
            loss = criterion_L2(
                torch.mul(confidence, pred_xy_keypoint),
                torch.mul(confidence, xy_keypoint),
            )

            valid_loss_iter.append(loss.cpu().detach().numpy())
            pred_xy_keypoint = pred_xy_keypoint.cpu()
            xy_keypoint = xy_keypoint.cpu()
            pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
            xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)
            pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)

            metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
            pck_50_iter.append(
                compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
            )
            pck_20_iter.append(
                compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2)
            )

        valid_mean_loss = np.mean(valid_loss_iter)
        valid_mean_loss_iter.append(valid_mean_loss)
        mean = np.mean(metric, 0) * 1000
        mpjpe_mean = mean[0]
        pa_mpjpe_mean = mean[1]
        pck_50 = np.mean(pck_50_iter, 0)
        pck_20 = np.mean(pck_20_iter, 0)
        pck_50_overall = pck_50[17]
        pck_20_overall = pck_20[17]
        print(
            "\nvalidation result with loss: %.3f, pck_50: %.3f, pck_20: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f"
            % (
                valid_mean_loss,
                pck_50_overall,
                pck_20_overall,
                mpjpe_mean,
                pa_mpjpe_mean,
            )
        )

        if pck_20_overall >= pck_20_overall_max:
            print(
                "saving the model at the end of epoch %d with pck_20: %.3f"
                % (epoch_index, pck_20_overall)
            )
            checkpoint_path = os.path.join(
                experiment_config["checkpoint"], "att_mmfi"
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(metafi, os.path.join(checkpoint_path, "best.pt"))
            pck_20_overall_max = pck_20_overall
        # ========================================== #
        # �� 新增：每个 Epoch 跑完后，自动备份一次完整进度
        # ========================================== #
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': metafi.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'pck_20_overall_max': pck_20_overall_max,
            'train_mean_loss_iter': train_mean_loss_iter
        }, state_path)
        print(f"[SAVE] �� Epoch {epoch_index} 的训练进度已自动备份。")
        # ========================================== #

        if (epoch_index + 1) % 50 == 0:
            print("the train loss for the first %.1f epoch is" % (epoch_index))
            print(train_mean_loss_iter)

epsilon = 0.4
criterion_L2 = nn.MSELoss()
loss = 0
test_loss_iter = []
metric = []
time_iter = []
pck_50_iter = []
pck_40_iter = []
pck_30_iter = []
pck_20_iter = []
pck_10_iter = []
pck_5_iter = []
metafi = torch.load(
    os.path.join(experiment_config["checkpoint"], "att_mmfi", "best.pt"),
    weights_only=False
)
with torch.no_grad():
    test_loop = tqdm(test_loader, desc="[Final Testing]")
    for i, data in enumerate(test_loop):
        torch.cuda.empty_cache()
        csi_data = data["input_wifi-csi"]
        # csi_data = torch.mean(csi_data, dim =2)
        scale = 6
        length_test = 27

        csi_data = csi_data.clone().detach().requires_grad_(True)
        csi_data = csi_data.cuda()
        csi_data = csi_data.type(torch.cuda.FloatTensor)
        # csi_dafeaturesta = csi_data.view(16,2,3,114,10)
        keypoint = data["output"]  # 17,3
        keypoint = keypoint.cuda()

        xy_keypoint = keypoint[:, :, 0:2].cuda()
        confidence = keypoint[:, :, 2:3].cuda()

        # add noise and denoise

        csi_data = csi_data.to("cpu").numpy()

        csi_data = torch.tensor(csi_data)

        csi_data = torch.tensor(csi_data)
        csi_data = csi_data.to("cuda")
        csi_data = csi_data.type(torch.cuda.FloatTensor)

        pred_xy_keypoint, time = metafi(csi_data)  # b,2,17,17

        loss = criterion_L2(
            torch.mul(confidence, pred_xy_keypoint),
            torch.mul(confidence, xy_keypoint),
        )
        test_loss_iter.append(loss.cpu().detach().numpy())

        pred_xy_keypoint = pred_xy_keypoint.cpu()
        xy_keypoint = xy_keypoint.cpu()
        pred_xy_keypoint_pck = torch.transpose(pred_xy_keypoint, 1, 2)
        xy_keypoint_pck = torch.transpose(xy_keypoint, 1, 2)

        pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
        metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))

        pck_50_iter.append(
            compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
        )
        pck_40_iter.append(
            compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.4)
        )
        pck_30_iter.append(
            compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.3)
        )
        pck_20_iter.append(
            compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2)
        )
        pck_10_iter.append(
            compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.1)
        )
        pck_5_iter.append(
            compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.05)
        )

    test_mean_loss = np.mean(test_loss_iter)
    sum_time = np.sum(time_iter)
    mean = np.mean(metric, 0) * 1000
    mpjpe_mean = mean[0]
    pa_mpjpe_mean = mean[1]
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)
    pck_50_overall = pck_50[17]
    pck_40_overall = pck_40[17]
    pck_30_overall = pck_30[17]
    pck_20_overall = pck_20[17]
    pck_10_overall = pck_10[17]
    pck_5_overall = pck_5[17]
    print(
        "test result with loss: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f"
        % (
            test_mean_loss,
            pck_50_overall,
            pck_40_overall,
            pck_30_overall,
            pck_20_overall,
            pck_10_overall,
            pck_5_overall,
            mpjpe_mean,
            pa_mpjpe_mean,
        )
    )

    label = [
        "Bot Torso",
        "L.Hip",
        "L.Knee",
        "L.Foot",
        "R.Hip",
        "R.Knee",
        "R.Foot",
        "Center Torso",
        "Upper Torso",
        "Neck Base",
        "Center Head",
        "R.Shoulder",
        "R.Elbow",
        "R.Hand",
        "L.Shoulder",
        "L.Elbow",
        "L.Hand",
    ]

    # Tạo danh sách dữ liệu
    data = []
    for i in range(len(label)):
        data.append([label[i], pck_20[i], pck_30[i], pck_40[i], pck_50[i]])

    # Thêm hàng "Average"
    data.append(
        [
            "Average",
            np.mean(pck_20),
            np.mean(pck_30),
            np.mean(pck_40),
            np.mean(pck_50),
        ]
    )

    # In bảng dạng tabular
    headers = ["Keypoint", "PCK@20", "PCK@30", "PCK@40", "PCK@50"]
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".2f"))
