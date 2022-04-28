#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import logging

import torch.optim


def build_lr_scheduler(cfg, optimizer, **kwargs):
    lr_scheduler_name = cfg.train.lr_scheduler.lower()

    if lr_scheduler_name == 'multisteplr':
        last_epoch = kwargs["last_epoch"] if 'last_epoch' in kwargs else -1

        if not isinstance(optimizer, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.train.lr_step, cfg.train.lr_factor, last_epoch=last_epoch)
        else:
            lr_scheduler = []
            for op in optimizer:
                lr_scheduler.append(
                    torch.optim.lr_scheduler.MultiStepLR(op, cfg.train.lr_step, cfg.train.lr_factor, last_epoch=last_epoch))

    else:
        logger = logging.getLogger(__name__)
        logger.error("Please Check if LR_SCHEDULER is valid")
        raise Exception("Please Check if LR_SCHEDULER is valid")

    return lr_scheduler
