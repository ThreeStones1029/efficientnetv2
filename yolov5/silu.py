'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-06-03 06:17:19
LastEditors: ShuaiLei
LastEditTime: 2024-06-06 08:30:24
'''
import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def silu_derivative(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 + x * (1 - sigmoid_x))
