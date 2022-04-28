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


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight: bool = True, divided_num_joints=True):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.use_target_weight = use_target_weight
        self.divided_num_joints = divided_num_joints

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                # loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
                # loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        if self.divided_num_joints:
            loss = loss / num_joints

        return loss


class JointHeatMapSupplementLoss(nn.Module):
    def __init__(self, use_keypoint_weight: bool = True, divided_num_joints=True):
        super(JointHeatMapSupplementLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

        self.use_keypoint_weight = use_keypoint_weight
        self.divided_num_joints = divided_num_joints

    def forward(self, sf_output, kf_output, gt, keypoint_weight):
        """
            sf_output:
                sup frame output [B,N,H,W]
            kp_output:
                key frame output [B,N,H,W]
            gt: [B,N,H,W]
            keypoint_weight: [B,N]
        """
        b, n, h, w = sf_output.shape
        sf_hm_pred = sf_output.reshape(b, n, -1)
        kf_hm_pred = kf_output.reshape(b, n, -1)
        hm_gt = gt.reshape(b, n, -1)

        hard_mask = (hm_gt > 0.01) * (kf_hm_pred <= 0.05)
        easy_maks = (hm_gt > 0.01) * (kf_hm_pred > 0.05)
        hard_weight, easy_weight = 1.5, 1

        loss = 0
        for k in range(n):
            joint_k_hm_pred = sf_hm_pred[:, k]
            joint_k_hm_gt = hm_gt[:, k]
            joint_k_hard_mask = hard_mask[:, k]
            joint_k_easy_mask = easy_maks[:, k]
            if self.use_keypoint_weight:
                joint_k_loss_map = self.criterion(joint_k_hm_pred.mul(keypoint_weight[:, k]), joint_k_hm_gt.mul(keypoint_weight[:, k]))
                joint_k_hard_loss_map = joint_k_loss_map * joint_k_hard_mask * hard_weight  # [B,N,-1]
                joint_k_easy_loss_map = joint_k_loss_map * joint_k_easy_mask * easy_weight  # [B,N,-1]

                loss += torch.sum(joint_k_hard_loss_map) / (torch.sum(joint_k_hard_mask) + 1) \
                        + torch.sum(joint_k_easy_loss_map) / (torch.sum(joint_k_easy_mask) + 1)
            else:

                joint_k_loss_map = self.criterion(joint_k_hm_pred, joint_k_hm_gt)
                joint_k_hard_loss_map = joint_k_loss_map * joint_k_hard_mask * hard_weight  # [B,N,-1]
                joint_k_easy_loss_map = joint_k_loss_map * joint_k_easy_mask * easy_weight  # [B,N,-1]

                loss += torch.sum(joint_k_hard_loss_map) / (torch.sum(joint_k_hard_mask) + 1) \
                        + torch.sum(joint_k_easy_loss_map) / (torch.sum(joint_k_easy_mask) + 1)
        if self.divided_num_joints:
            loss = loss / n
        return loss
