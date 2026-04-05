import time

import torch
from torch import nn

from .utils import SKUnit, regression


class BasicCnnHPE(nn.Module):
    def __init__(self):
        super(BasicCnnHPE, self).__init__()
        num_lay = 64#numer hidden dim of DyConv1
        hidden_reg = 32 #number hidden dim of Regression

        self.CNN1 = nn.Conv2d(in_channels=3, out_channels=num_lay, kernel_size=7, stride=1)
        self.bn = nn.BatchNorm2d(num_lay)
        self.relu = nn.ReLU(inplace=True)

        #self.CNN2 = nn.Conv2d(in_channels=num_lay, out_channels=num_lay*2, kernel_size=3, stride=1)
        self.regression = regression(input_dim = 1728,output_dim = 34, hidden_dim= hidden_reg)

    def forward(self,x): #16,2,3,114,32
        batch = x.shape[0]
        
        
 
        time_start = time.time()
        
        " CNN-spatio"
        
       
        
        
        " Selective-Dynamic_convolution "
        m = torch.nn.AvgPool2d((2, 2))
        x = self.CNN1(x)
        x = m(x)#[32, 64, 57, 5]
        
        #out1 = self.CNN2(x)
        out1 = self.bn(x)
        out1 = self.relu(out1)
        
        " Max-Pooling"
        
        out1 = m(out1) 

    
        x = self.regression(out1)
        x = x.reshape(batch,17,2)
        
        
        time_end = time.time()
        time_sum = time_end - time_start

        #x = torch.transpose(x, 1, 2)

        return x,time_sum