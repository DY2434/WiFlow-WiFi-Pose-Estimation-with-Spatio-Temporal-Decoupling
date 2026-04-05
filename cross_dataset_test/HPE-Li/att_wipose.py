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
from torch.utils.data import DataLoader
from tqdm import tqdm

from constant import experiment_config
from model import *
from utils import calulate_error, compute_pck_pckh_18
from wipose import WiPoseDataset

train_dataset, test_dataset = WiPoseDataset(
    root_dir="/home/jackson-devworks/Desktop/HPE/Wi-Pose"
), WiPoseDataset(
    root_dir="/home/jackson-devworks/Desktop/HPE/Wi-Pose", split="Test"
)
val_data, test_data = train_test_split(
    test_dataset, test_size=0.5, random_state=41
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

metafi = HPEWiPoseModel().to(device)

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.AdamW(metafi.parameters(), lr=0.001)
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
for epoch_index in tqdm(range(num_epochs)):
    torch.cuda.empty_cache()
    loss = 0
    train_loss_iter = []
    metric = []
    metafi.train()
    relation_mean = []
    for idx, data in enumerate(train_loader):
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

        lr = np.array(scheduler.get_last_lr())
        message = "(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) " % (
            epoch_index,
            idx * 32,
            lr,
            loss,
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
        for idx, data in enumerate(val_loader):
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
            pck = compute_pck_pckh_18(
                pred_xy_keypoint_pck, xy_keypoint_pck, 0.5
            )

            metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
            pck_50_iter.append(
                compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
            )
            pck_20_iter.append(
                compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2)
            )

        valid_mean_loss = np.mean(valid_loss_iter)
        valid_mean_loss_iter.append(valid_mean_loss)
        mean = np.mean(metric, 0) * 1000
        mpjpe_mean = mean[0]
        pa_mpjpe_mean = mean[1]
        pck_50 = np.mean(pck_50_iter, 0)
        pck_20 = np.mean(pck_20_iter, 0)
        pck_50_overall = pck_50[18]
        pck_20_overall = pck_20[18]
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
                experiment_config["checkpoint"], "att_wipose"
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(metafi, os.path.join(checkpoint_path, "best.pt"))
            pck_20_overall_max = pck_20_overall

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
    os.path.join(experiment_config["checkpoint"], "att_wipose", "best.pt")
)
with torch.no_grad():
    for i, data in enumerate(test_loader):
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

        pck = compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)

        metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))

        pck_50_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
        )
        pck_40_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.4)
        )
        pck_30_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.3)
        )
        pck_20_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2)
        )
        pck_10_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.1)
        )
        pck_5_iter.append(
            compute_pck_pckh_18(pred_xy_keypoint_pck, xy_keypoint_pck, 0.05)
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
    pck_50_overall = pck_50[18]
    pck_40_overall = pck_40[18]
    pck_30_overall = pck_30[18]
    pck_20_overall = pck_20[18]
    pck_10_overall = pck_10[18]
    pck_5_overall = pck_5[18]
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
        "Nose",
        "Neck",
        "R.Shoulder",
        "R.Elbow",
        "R.Wrist",
        "L.Shoulder",
        "L.Elbow",
        "L.Wrist",
        "R.Hip",
        "R.Knee",
        "R.Ankle",
        "L.Hip",
        "L.Knee",
        "L.Ankle",
        "R.Eye",
        "L.Eye",
        "R.Ear",
        "L.Ear",
    ]

    # Tạo danh sách dữ liệu
    data = []
    for i in range(len(label)):
        data.append([label[i], pck_5[i], pck_10[i], pck_20[i], pck_30[i], pck_40[i], pck_50[i]])

    # Thêm hàng "Average"
    data.append(
        [
            "Average",
            np.mean(pck_5),
            np.mean(pck_10),
            np.mean(pck_20),
            np.mean(pck_30),
            np.mean(pck_40),
            np.mean(pck_50),
        ]
    )

    # In bảng dạng tabular
    headers = ["Keypoint", "PCK@5", "PCK@10", "PCK@20", "PCK@30", "PCK@40", "PCK@50"]
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".2f"))
