import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AxialAttention(nn.Module):
    """轴向自注意力机制"""

    def __init__(self, in_planes, out_planes, groups=8, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.stride = stride
        self.bias = bias
        self.width = width

        # QKV变换
        self.qkv_transform = nn.Conv1d(
            in_planes, out_planes * 3, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_qkv = nn.BatchNorm1d(out_planes * 3)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_transform.weight.data, 0, math.sqrt(1. / self.in_planes))

    def forward(self, x):
        B, C, H, W = x.shape

        # 调整维度顺序
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # QKV变换
        qkv = self.bn_qkv(self.qkv_transform(x))
        qkv = qkv.reshape(N * W, 3, self.out_planes, H)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 分组
        q = q.reshape(N * W, self.groups, self.group_planes, H)
        k = k.reshape(N * W, self.groups, self.group_planes, H)
        v = v.reshape(N * W, self.groups, self.group_planes, H)

        # 注意力计算
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        qk = self.bn_similarity(qk)
        similarity = F.softmax(qk, dim=-1)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        # 整理输出
        sv = sv.reshape(N * W, self.out_planes, H)
        out = self.bn_output(sv)
        out = out.view(N, W, self.out_planes, H)

        # 恢复维度
        if self.width:
            out = out.permute(0, 2, 1, 3)
        else:
            out = out.permute(0, 2, 3, 1)

        if self.stride > 1:
            out = self.pooling(out)

        return out


class DualAxialAttention(nn.Module):
    """双轴注意力：宽度+高度"""

    def __init__(self, in_planes, out_planes, groups=8, stride=1, bias=False):
        super().__init__()
        self.width_axis = AxialAttention(
            in_planes, out_planes, groups, stride, bias, width=True
        )
        self.height_axis = AxialAttention(
            out_planes, out_planes, groups, stride, bias, width=False
        )

    def forward(self, x):
        x = self.width_axis(x)
        x = self.height_axis(x)
        return x