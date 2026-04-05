import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

# from thop import profile
# from thop import clever_format


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branches.
            G: number of convolution groups.
            r: ratio to compute d, the length of z.
            stride: stride, default 1.
            L: minimum dimension of the vector z, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        # Define M convolution branches
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=stride,
                        padding=1 + i,
                        dilation=1 + i,
                        groups=G,
                        bias=False,
                    ),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                )
                for i in range(M)
            ]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        # Fully connected layers for each branch
        self.fcs = nn.ModuleList(
            [nn.Conv2d(d, features, kernel_size=1) for _ in range(M)]
        )
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm2d(features)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply each branch convolution and stack results
        feats = torch.stack(
            [conv(x) for conv in self.convs], dim=1
        )  # [batch, M, features, H, W]

        # Compute channel attention
        feats_U = feats.sum(dim=1)  # [batch, features, H, W]
        feats_S = self.gap(feats_U)  # [batch, features, 1, 1]
        feats_Z = self.fc(feats_S)  # [batch, d, 1, 1]
        attention_vectors = torch.stack(
            [fc(feats_Z) for fc in self.fcs], dim=1
        )  # [batch, M, features, 1, 1]
        attention_vectors = self.softmax(
            attention_vectors
        )  # [batch, M, features, 1, 1]

        # Weighted sum of features using channel attention
        feats_channel = (feats * attention_vectors).sum(
            dim=1
        )  # [batch, features, H, W]

        # Compute frequency attention
        feats_frequency = feats.sum(dim=2)  # [batch, M, H, W]
        feats_S_frequency = F.adaptive_avg_pool2d(
            feats_frequency, (feats_frequency.size(2), 1)
        )  # [batch, M, H, 1]
        attention_vectors_frequency = self.softmax(
            feats_S_frequency
        )  # [batch, M, H, 1]

        # Weighted sum of features using frequency attention
        feats_frequency = (
            feats * attention_vectors_frequency.unsqueeze(2)
        ).sum(
            dim=1
        ) 
        
        # Reshape and compute output
        return feats_channel + feats_frequency


class SKUnit(nn.Module):
    def __init__(
        self,
        in_features,
        mid_features,
        out_features,
        M=2,
        G=32,
        r=16,
        stride=1,
        L=32,
    ):
        """Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )

        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

        if (
            in_features == out_features
        ):  # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_features, out_features, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_features),
            )

        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_features)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_sk(out)
        residual = out
        out = self.conv3(out)
        return self.relu(self.norm(out + residual))
