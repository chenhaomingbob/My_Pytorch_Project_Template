# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
        from DEKR
"""
import torch
import torch.nn as nn


class OffsetsLoss(nn.Module):
    def __init__(self, beta=1 / .9, norm_cal=True):
        super().__init__()
        self.beta = beta
        self.width = 72
        self.height = 96
        self.norm_cal = norm_cal

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)

        return loss

    def forward(self, norm_pred, norm_gt, weights):
        assert norm_pred.size() == norm_gt.size()
        pred, gt = torch.zeros_like(norm_pred), torch.zeros_like(norm_gt)
        #
        if self.norm_cal:
            pred = norm_pred
            gt = norm_gt
        else:
            pred[:, 0::2] = norm_pred[:, 0::2] * self.width  # x
            pred[:, 1::2] = norm_pred[:, 1::2] * self.height  # y
            gt[:, 0::2] = norm_gt[:, 0::2] * self.width
            gt[:, 1::2] = norm_gt[:, 1::2] * self.height
        #
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred * weights, gt * weights, self.beta)
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class OffsetsVarLoss(nn.Module):
    """
        logvar : log(/sigmoid^2)
    """

    def __init__(self):
        super().__init__()

    def var_loss(self, pred, logvar, gt, beta=1):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        # xy_logvar = torch.repeat_interleave(logvar, 2, dim=1)
        smooth_loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)

        error_loss = torch.exp(-logvar) * smooth_loss
        regular_loss = 0.5 * logvar
        return error_loss, regular_loss

    def forward(self, pred, logvar, gt, mask):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(mask > 0).size()[0]
        # regular_weight = (mask.reshape(mask.shape[0], 34, -1) > 0).sum(2) > 0

        error_loss, var_loss = self.var_loss(pred, logvar, gt)
        error_loss *= mask
        var_loss *= mask
        # regular_loss *= regular_weight
        if num_pos == 0:
            num_pos = 1.
        error_loss = error_loss.sum() / num_pos
        regular_loss = var_loss.sum() / num_pos

        loss = error_loss + regular_loss
        return loss
