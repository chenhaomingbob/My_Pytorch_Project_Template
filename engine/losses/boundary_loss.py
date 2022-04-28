# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description: 
"""
import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    def __init__(self, threshold=20):
        super(BoundaryLoss, self).__init__()
        self.threshold = threshold

    def forward(self, output, target_weight, **kwargs):
        """
            output:
                heatmaps shape (B, K, H, W)
                    (B, 17, 96, 72)
            target_weight: (B, K, 1)
                    (B,17,1)

        """
        batch_size, num_joints = output.shape[0:2]
        output = output.reshape(batch_size, num_joints, -1)
        output = torch.sum(output, dim=2)
        ph_sup_heatmaps = kwargs.get('ph_sup_heatmaps', None)
        ph_sup_heatmaps = ph_sup_heatmaps.reshape(batch_size, num_joints, -1)
        ph_sup_heatmaps = torch.sum(ph_sup_heatmaps, dim=2)
        target = kwargs.get('target', None)

        if target is not None:
            target = target.reshape(batch_size, num_joints, -1)
            target = torch.sum(target, dim=2)
            target_mask = torch.le(target, self.threshold)
        output_mask_1 = torch.le(output, self.threshold)
        output_mask_2 = torch.gt(ph_sup_heatmaps, self.threshold)
        output_mask = output_mask_1 * output_mask_2
        loss = output * output_mask
        loss = self.threshold - loss  # handle with boundary loss
        loss = loss * target_weight.squeeze(-1)
        loss = torch.sum(loss, dim=1).mean()
        return loss
