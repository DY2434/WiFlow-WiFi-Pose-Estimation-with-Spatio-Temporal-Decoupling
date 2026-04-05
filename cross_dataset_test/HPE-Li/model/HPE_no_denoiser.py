import time

import torch
from torch import nn

from .utils import SKUnit, regression


class OriginalHPE(nn.Module):
    def __init__(self):
        super(OriginalHPE, self).__init__()
        num_lay = 64  # numer hidden dim of DyConv1
        hidden_reg = 32  # number hidden dim of Regression

        self.skunit1 = SKUnit(
            in_features=3,
            mid_features=num_lay,
            out_features=num_lay,
            dim1=114,
            dim2=10,
            pool_dim="freq-chan",
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
            dim1=57,
            dim2=8,
            pool_dim="freq-chan",
            M=2,
            G=64,
            r=4,
            stride=1,
            L=32,
        )
        self.regression = regression(
            input_dim=7168, output_dim=34, hidden_dim=hidden_reg
        )

    def forward(self, x):  # 16,2,3,114,32
        batch = x.shape[0]

        time_start = time.time()

        " CNN-spatio"

        " Selective-Dynamic_convolution "
        m = torch.nn.AvgPool2d((2, 2))
        x = self.skunit1(x)
        x = m(x)  # [32, 64, 57, 5]

        out1 = self.skunit2(x)
        # out1 = self.bn1(out1)
        # out1 = self.relu(out1)

        " Max-Pooling"

        out1 = m(out1)

        x = self.regression(out1)
        x = x.reshape(batch, 17, 2)

        time_end = time.time()
        time_sum = time_end - time_start

        # x = torch.transpose(x, 1, 2)

        return x, time_sum
