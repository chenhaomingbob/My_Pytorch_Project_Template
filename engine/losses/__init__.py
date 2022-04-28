#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
from .build import build_loss
from .integral_loss import IntegralL1Loss
from .mse_loss import JointMSELoss, JointHeatMapSupplementLoss
from .perceptual_loss import PerceptualLoss
