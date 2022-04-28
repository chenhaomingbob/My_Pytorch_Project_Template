#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import logging

import numpy as np
from tabulate import tabulate
from termcolor import colored
from torch.utils.data import Dataset

from utils.utils_constant import TRAIN_PHASE
from datasets.build import get_dataset_name


class BaseDataset(Dataset):
    def __init__(self, cfg, phase='train', **kwargs):
        self.dataset_name = get_dataset_name(cfg)
        self.phase = phase

        # common init
        self.is_train = True if self.phase == 'train' else False
        self.pixel_std = 200
        self.num_joints = cfg.model.num_joints
        self.output_dir = cfg.output_dir
        self.color_rgb = cfg.dataset.color_rgb

        self.image_size = np.array(cfg.model.image_size)
        self.image_width = self.image_size[0]
        self.image_height = self.image_size[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = np.array(cfg.model.heatmap_size)

        # normal data augmentation
        self.scale_factor = cfg.train.scale_factor
        self.rotation_factor = cfg.train.rot_factor
        self.flip = cfg.train.flip
        self.num_joints_half_body = cfg.train.num_joints_half_body
        self.prob_half_body = cfg.train.prob_half_body

        # Loss
        self.use_different_joints_weight = cfg.loss.use_different_joints_weight

        self.data = []

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class VideoDataset(BaseDataset):
    """
        A base class representing VideoDataset.
        All other video datasets should subclass it.
    """

    def __init__(self, cfg, phase, **kwargs):
        super(VideoDataset, self).__init__(cfg, phase, **kwargs)

    def __getitem__(self, item):
        raise NotImplementedError

    def show_samples(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset_Name", "Num of samples"]
        table_data = [[self.dataset_name, len(self.data)]]

        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Samples Info : \n" + colored(table, "magenta"))

    def show_data_parameters(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset parameters", "Value"]
        table_data = [
            ["BBOX_ENLARGE_FACTOR", self.bbox_enlarge_factor],
            ["NUM_JOINTS", self.num_joints]
        ]
        if self.phase != TRAIN_PHASE:
            table_extend_data = [
                []
            ]
            table_data.extend(table_extend_data)
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Parameters Info : \n" + colored(table, "magenta"))


class ImageDataset(BaseDataset):
    """
        A base class representing ImageDataset.
        All other image datasets should subclass it.
    """

    def __getitem__(self, item):
        raise NotImplementedError

    def show_samples(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset_Name", "Num of samples"]
        table_data = [[self.dataset_name, len(self.data)]]

        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Samples Info : \n" + colored(table, "magenta"))
