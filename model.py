import torch
import torch.nn as nn
import numpy as np

from utils import conv2d, upconv2d
 
class Generator(nn.Module):
    def __init__(self, z_dim, dim):
        super(Generator, self).__init__()

        # z_dim = 100, dim = 1024
        cfg = [
            [z_dim, dim, 4, 1, 0, 'ReLU'], 
            [dim, dim//2, 4, 2, 1, 'ReLU'], 
            [dim//2, dim//4, 4, 2, 1, 'ReLU'], 
            [dim//4, dim//8, 4, 2, 1, 'ReLU'], 
            [dim//8, 3, 4, 2, 1, 'Tanh']
        ]

        self.conv1 = upconv2d(cfg[0])
        self.conv2 = upconv2d(cfg[1])
        self.conv3 = upconv2d(cfg[2])
        self.conv4 = upconv2d(cfg[3])
        self.conv5 = upconv2d(cfg[4], batch_norm = False) 

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        return out


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        # dim = 1024
        cfg_d = [
            [3, dim//8, 4, 2, 1, 'LeakyReLU'], 
            [dim//8, dim//4, 4, 2, 1, 'LeakyReLU'], 
            [dim//4, dim//2, 4, 2, 1, 'LeakyReLU'], 
            [dim//2, dim, 4, 2, 1, 'LeakyReLU'], 
            [dim, 1, 4, 1, 0, 'Sigmoid']
        ] 
        
        self.conv1 = conv2d(cfg_d[0])
        self.conv2 = conv2d(cfg_d[1])
        self.conv3 = conv2d(cfg_d[2])
        self.conv4 = conv2d(cfg_d[3])
        self.conv5 = conv2d(cfg_d[4], batch_norm = False)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out