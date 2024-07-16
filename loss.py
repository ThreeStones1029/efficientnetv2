'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-07-16 03:00:22
LastEditors: ShuaiLei
LastEditTime: 2024-07-16 15:04:51
'''
import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


if __name__ == "__main__":
    loss = FocalLoss()