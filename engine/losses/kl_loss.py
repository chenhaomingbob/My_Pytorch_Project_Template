# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
        KL Loss
"""
import torch.nn as nn


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        pass

    def forward(self, mu, logvar):
        kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
        if len(kl_loss.shape) == 4:
            B, C, H, W = kl_loss.shape
            kl_loss = kl_loss.view(B, C * H * W)
            kl_loss = kl_loss.sum() / (B * C * H * W)
        elif len(kl_loss.shape) == 3:
            B, C, D = kl_loss.shape
            kl_loss = kl_loss.view(B, C * D)
            kl_loss = kl_loss.sum() / (B * C * D)
        else:
            kl_loss = kl_loss.sum(dim=1).mean()

        return kl_loss
