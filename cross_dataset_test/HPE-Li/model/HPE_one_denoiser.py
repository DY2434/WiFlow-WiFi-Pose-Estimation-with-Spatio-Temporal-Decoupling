import time

import torch
import torch.nn as nn
from torch import nn

from .utils import SKUnit, regression


class OneStageAE(nn.Module):
    def __init__(self, pre_encoder=None):
        super(OneStageAE, self).__init__()
        self.pre_encoder = pre_encoder
        # Encoder with a single convolutional layer and max pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # Convolution
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling
        )
    
        # Decoder with corresponding layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # Upsampling
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)  # Final layer
        )

    def getProcessingInput(self, x):
        if self.pre_encoder == None:
            return x
        return self.pre_encoder(x)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def getEncoder(self):
        return nn.Sequential(
                self.encoder
            )

class OneLayerDenoiserHPE(nn.Module):
    def __init__(self, encoder):
        super(OneLayerDenoiserHPE, self).__init__()
        num_lay = 64#numer hidden dim of DyConv1
        hidden_reg = 32 #number hidden dim of Regression
        self.encoder = encoder
        self.skunit1 = SKUnit(in_features=16, mid_features=num_lay, out_features=num_lay, dim1 = 57,dim2 = 10,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = SKUnit(in_features=num_lay, mid_features=num_lay*2, out_features=num_lay*2, dim1 = 28,dim2 = 8,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.regression = regression(input_dim =1792,output_dim = 34, hidden_dim= hidden_reg)

    def forward(self,x): #16,2,3,114,32
        batch = x.shape[0]
        
        
 
        time_start = time.time()
        
        " CNN-spatio"
        
        x = self.encoder(x)
        
        
        " Selective-Dynamic_convolution "
        m = torch.nn.AvgPool2d((2, 2))
        x = self.skunit1(x)
        x = m(x)#[32, 64, 57, 5]
        
        
        out1 = self.skunit2(x)
        #out1 = self.bn1(out1)
        #out1 = self.relu(out1)
        
        " Max-Pooling"
        
        out1 = m(out1) 

    
        x = self.regression(out1)
        x = x.reshape(batch,17,2)
        
        
        time_end = time.time()
        time_sum = time_end - time_start

        #x = torch.transpose(x, 1, 2)

        return x,time_sum