import torch

def calculate_pck(pred, target, thresholds=[0.2], use_torso_norm=True):
    """计算PCK (Percentage of Correct Keypoints)"""
    batch_size = pred.shape[0]

    if len(pred.shape) == 2 and pred.shape[1] == 30:
        pred = pred.reshape(batch_size, 15, 2)
        target = target.reshape(batch_size, 15, 2)

    # 归一化距离
    if use_torso_norm:
        NECK_IDX, PELVIS_IDX = 2, 12
        normalize_distances = torch.sqrt(
            torch.sum((target[:, NECK_IDX] - target[:, PELVIS_IDX]) ** 2, dim=1)
        )
    else:
        L_SHOULDER_IDX, R_SHOULDER_IDX = 2, 5
        normalize_distances = torch.sqrt(
            torch.sum((target[:, L_SHOULDER_IDX] - target[:, R_SHOULDER_IDX]) ** 2, dim=1)
        )

    normalize_distances = torch.clamp(normalize_distances, min=0.01)
    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))
    normalized_distances = distances / normalize_distances.unsqueeze(1)

    pck_results = {}
    for threshold in thresholds:
        correct_keypoints = (normalized_distances <= threshold).float()
        pck_overall = correct_keypoints.mean()
        pck_results[threshold] = pck_overall.item()

    return pck_results


def calculate_mpjpe(pred, target):
    """计算MPJPE (Mean Per Joint Position Error)"""
    batch_size = pred.shape[0]

    if len(pred.shape) == 2 and pred.shape[1] == 30:
        pred = pred.reshape(batch_size, 15, 2)
        target = target.reshape(batch_size, 15, 2)

    distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))
    mean_distance = torch.mean(distances)

    return mean_distance.item()