#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:30:24 2024

@author: jackson-devworks
"""
import numpy as np


def compute_pck_pckh_18(dt_kpts, gt_kpts, thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)
    assert dt.shape[0] == gt.shape[0]
    kpts_num = gt.shape[2]  # keypoints
    ped_num = gt.shape[0]  # batch_size
    # compute dist
    scale = np.sqrt(
        np.sum(np.square(gt[:, :, 5] - gt[:, :, 8]), 1)
    )  # right shoulder--left hip
    dist = (
        np.sqrt(np.sum(np.square(dt - gt), 1))
        / np.tile(scale, (gt.shape[2], 1)).T
    )
    # dist=np.sqrt(np.sum(np.square(dt-gt),1))
    # compute pck
    pck = np.zeros(gt.shape[2] + 1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= thr)
        # compute average pck
    pck[18] = 100 * np.mean(dist <= thr)
    return pck


def compute_pck_pckh(dt_kpts, gt_kpts, thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)
    assert dt.shape[0] == gt.shape[0]
    kpts_num = gt.shape[2]  # keypoints
    ped_num = gt.shape[0]  # batch_size
    # compute dist
    scale = np.sqrt(
        np.sum(np.square(gt[:, :, 1] - gt[:, :, 11]), 1)
    )  # right shoulder--left hip
    # dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    dist = (
        np.sqrt(np.sum(np.square(dt - gt), 1))
        / np.tile(scale, (gt.shape[2], 1)).T
    )
    # dist=np.sqrt(np.sum(np.square(dt-gt),1))
    # compute pck
    pck = np.zeros(gt.shape[2] + 1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= thr)
        # compute average pck
    pck[17] = 100 * np.mean(dist <= thr)
    return pck


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Args:
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    X0 /= normX
    Y0 /= normY

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    detT = np.linalg.det(T)
    if detT < 0:  # Ensure a proper rotation matrix
        V[:, -1] *= -1
        s[-1] *= -1
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def calulate_error(predicted_keypoints, ground_truth_keypoints):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.

    :param predicted_keypoints: Estimated results from the algorithm, shape=[n, h, w]
    :param ground_truth_keypoints: Ground truth marked results, shape=[n, h, w]
    :return: Mean Per Joint Position Error (MPJPE) and Procrustes Aligned MPJPE (PA-MPJPE)
    """
    # Convert inputs to numpy arrays
    predicted_keypoints = np.array(predicted_keypoints)
    ground_truth_keypoints = np.array(ground_truth_keypoints)

    # Validate input shapes
    assert (
        predicted_keypoints.shape == ground_truth_keypoints.shape
    ), "Input shapes must match"

    N = predicted_keypoints.shape[0]  # Number of samples
    num_joints = predicted_keypoints.shape[2]  # Number of keypoints

    # Calculate MPJPE
    mpjpe = np.mean(
        np.sqrt(
            np.sum(
                np.square(predicted_keypoints - ground_truth_keypoints), axis=2
            )
        )
    )

    # Calculate PA-MPJPE
    pampjpe = np.zeros(N)

    for n in range(N):
        frame_pred = predicted_keypoints[n]  # Shape [h, w]
        frame_gt = ground_truth_keypoints[n]  # Shape [h, w]

        # Compute similarity transform
        _, Z, T, b, c = compute_similarity_transform(
            frame_gt, frame_pred, compute_optimal_scale=True
        )

        # Apply the transformation to predictions
        frame_pred_transformed = (b * frame_pred @ T) + c
        pampjpe[n] = np.mean(
            np.sqrt(
                np.sum(np.square(frame_pred_transformed - frame_gt), axis=1)
            )
        )

    # Compute average PA-MPJPE
    pampjpe = np.mean(pampjpe)

    return mpjpe, pampjpe
