#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import torch
import torch.nn as nn


class DiversityLoss(nn.Module):
    def __init__(self, criterion="MSE"):
        super(DiversityLoss, self).__init__()
        # LOSS: MSE, NLL, CrossEntropy, KLDiv
        Candidate_Loss_Criterion = ["MSE", "KLDiv"]
        assert criterion in Candidate_Loss_Criterion, f"Unexpected criterion '{criterion}' of DiversityLoss"
        if criterion == "MSE":
            self.criterion = nn.MSELoss(reduction='none')
        elif criterion == "KLDiv":
            self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, sample_a, sample_b, target_weight=None):
        batch_size, num_joints = sample_a.size(0), sample_a.size(1)
        sample_a = sample_a.reshape(batch_size, num_joints, -1)
        sample_b = sample_b.reshape(batch_size, num_joints, -1)

        loss = self.criterion(sample_a, sample_b)
        loss = torch.mean(loss, dim=2)
        if target_weight is not None:
            loss = loss * target_weight.squeeze(-1)
        loss = torch.sum(loss, dim=1).mean()

        return loss / num_joints
