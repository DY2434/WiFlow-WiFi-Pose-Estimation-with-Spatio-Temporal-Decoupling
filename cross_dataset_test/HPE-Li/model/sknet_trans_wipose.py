import time

import torch
import torch.nn.functional as F
from torch import nn

from .utils import ChannelTransformer, regression


class SKConv(nn.Module):
    def __init__(self, features, img_size, M=2, G=32, r=16, stride=1, L=32):
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
        self.tf = ChannelTransformer(
            vis=False,
            img_size=img_size,
            channel_num=features,
            num_layers=1,
            num_heads=3,
        )

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
        ).sum(dim=1)

        # Reshape and compute output
        feats_V = torch.cat([feats_channel, feats_frequency], dim=3)

        feats_V = self.norm(feats_V)
        feats_V, weight = self.tf(feats_V)
        feats_V = F.avg_pool2d(feats_V, kernel_size=(1, 2))
        # print(feats_V.shape, feats_channel.shape)
        return feats_V


class SKUnit(nn.Module):
    def __init__(
        self,
        in_features,
        mid_features,
        out_features,
        img_size,
        M=2,
        G=32,
        r=16,
        stride=1,
        L=32,
    ):
        super(SKUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )

        self.conv2_sk = SKConv(
            mid_features, img_size=img_size, M=M, G=G, r=r, stride=stride, L=L
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.pooling = torch.nn.AvgPool2d((2, 2))
        self.norm = nn.BatchNorm2d(mid_features)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.pooling(out)
        out = self.conv2_sk(out)
        out = self.norm(out)
        out = self.conv3(out)
        return out


class DSKNetTransWipose(nn.Module):
    def __init__(self):
        super(DSKNetTransWipose, self).__init__()
        num_lay = 64
        hidden_reg = 32
        self.skunit1 = SKUnit(
            in_features=9,
            mid_features=num_lay,
            out_features=num_lay,
            img_size=[30, 10],
            M=2,
            G=64,
            r=4,
            stride=1,
            L=32,
        )
        self.skunit2 = SKUnit(
            in_features=num_lay,
            mid_features=num_lay * 2,
            out_features=num_lay * 2,
            img_size=[15,4],
            M=2,
            G=64,
            r=4,
            stride=1,
            L=32,
        )

        self.regression = regression(
            input_dim=3840, output_dim=36, hidden_dim=hidden_reg
        )
        self.norm = nn.BatchNorm2d(num_lay)

        # self._init_weights()

    def forward(self, x):
        batch = x.shape[0]

        time_start = time.time()
        m = torch.nn.AvgPool2d((2, 2))
        out = self.skunit1(x)
        out = self.norm(out)
        out = m(out)
        out = self.skunit2(out)
        

        out = self.regression(out)
        out = out.reshape(batch, 18, 2)

        time_end = time.time()
        time_sum = time_end - time_start

        return out, time_sum

    def _init_weights(self):
        """
        Hàm khởi tạo trọng số cho các layer trong model.
        """

        def init_fn(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(
                layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            ):
                if layer.weight is not None:
                    nn.init.ones_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                nn.init.uniform_(layer.weight, a=-0.05, b=0.05)
            elif isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(layer, nn.TransformerEncoderLayer) or isinstance(
                layer, nn.TransformerDecoderLayer
            ):
                for name, param in layer.named_parameters():
                    if param.dim() > 1:  # Trọng số
                        nn.init.xavier_uniform_(param)
                    else:  # Bias
                        nn.init.zeros_(param)

        # Áp dụng hàm khởi tạo cho tất cả các layer
        self.apply(init_fn)
