import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    """姿态估计损失函数"""

    def __init__(
            self,
            position_weight: float = 1.0,
            bone_weight: float = 0.2,
            loss_type: str = 'smooth_l1'
    ):
        super().__init__()
        self.position_weight = position_weight
        self.bone_weight = bone_weight
        self.loss_type = loss_type

        # 骨架连接
        self.bone_connections = [
            (0, 1), (1, 8), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7), (8, 9), (8, 12),
            (9, 10), (10, 11), (12, 13), (13, 14)
        ]

    def compute_bone_lengths(self, keypoints):
        """计算骨骼长度"""
        bone_lengths = []
        for start_idx, end_idx in self.bone_connections:
            bone_vec = keypoints[..., end_idx, :] - keypoints[..., start_idx, :]
            bone_length = torch.sqrt(torch.sum(bone_vec ** 2, dim=-1) + 1e-8)
            bone_lengths.append(bone_length)
        return torch.stack(bone_lengths, dim=-1)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 15, 2] 预测关键点
            target: [B, 15, 2] 真实关键点
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        batch_size = pred.shape[0]

        # 形状检查
        if pred.shape != target.shape:
            if len(pred.shape) == 2 and pred.shape[1] == 30:
                pred = pred.reshape(batch_size, 15, 2)
            if len(target.shape) == 2 and target.shape[1] == 30:
                target = target.reshape(batch_size, 15, 2)

        # 位置损失
        if self.loss_type == 'mse':
            position_loss = F.mse_loss(pred, target)
        elif self.loss_type == 'l1':
            position_loss = F.l1_loss(pred, target)
        elif self.loss_type == 'smooth_l1':
            position_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 骨骼长度约束
        try:
            pred_bone_lengths = self.compute_bone_lengths(pred)
            target_bone_lengths = self.compute_bone_lengths(target)

            if self.loss_type == 'mse':
                bone_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'l1':
                bone_loss = F.l1_loss(pred_bone_lengths, target_bone_lengths)
            elif self.loss_type == 'smooth_l1':
                bone_loss = F.smooth_l1_loss(pred_bone_lengths, target_bone_lengths, beta=0.05)
        except Exception:
            bone_loss = torch.tensor(0.0, device=pred.device)

        # 总损失
        total_loss = (
                self.position_weight * position_loss +
                self.bone_weight * bone_loss
        )

        loss_dict = {
            'position': position_loss.item(),
            'bone': bone_loss.item()
        }

        return total_loss, loss_dict