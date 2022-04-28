#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import os

from yacs.config import CfgNode as _CfgNode
import os.path as osp
from engine.models.zoo import get_model_hyperparameter
from datasets import get_dataset_name
from utils.utils_folder import create_folder

BASE_KEY = '_BASE_'


class CfgNode(_CfgNode):

    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = osp.expanduser(base_cfg_file)
            else:
                base_cfg_file = osp.join(osp.dirname(cfg_filename), base_cfg_file)
            with open(base_cfg_file, "r") as base_f:
                base_cfg = self.load_cfg(base_f)
            self.merge_from_other_cfg(base_cfg)
            del cfg[BASE_KEY]
        self.merge_from_other_cfg(cfg)


def update_config(cfg: CfgNode, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # distributed parameters
    cfg.dist.dist_url = args.dist_url
    cfg.dist.world_size = args.world_size
    cfg.dist.rank = args.rank

    # update dir
    if args.rootDir:
        cfg.root_dir = args.rootDir

    cfg.output_dir = osp.abspath(osp.join(cfg.root_dir, cfg.output_dir))

    cfg.dataset.json_dir = osp.abspath(osp.join(cfg.root_dir, cfg.dataset.json_dir))
    cfg.dataset.img_dir = osp.abspath(osp.join(cfg.root_dir, cfg.dataset.img_dir))
    cfg.dataset.test_img_dir = osp.abspath(osp.join(cfg.root_dir, cfg.dataset.test_img_dir))

    if len(cfg.model.pretrained) > 0:
        cfg.model.pretrained = osp.abspath(osp.join(cfg.root_dir, cfg.model.pretrained))
    cfg.model.backbone_pretrained = osp.abspath(osp.join(cfg.root_dir, cfg.model.backbone_pretrained))

    cfg.val.annot_dir = osp.abspath(osp.join(cfg.root_dir, cfg.val.annot_dir))
    cfg.val.coco_bbox_file = osp.abspath(osp.join(cfg.root_dir, cfg.val.coco_bbox_file))

    cfg.test.annot_dir = osp.abspath(osp.join(cfg.root_dir, cfg.test.annot_dir))
    cfg.test.coco_bbox_file = osp.abspath(osp.join(cfg.root_dir, cfg.test.coco_bbox_file))

    hyper_parameters_setting = get_model_hyperparameter(cfg)
    dataset_name = get_dataset_name(cfg)
    cfg.output_dir = osp.join(cfg.output_dir, cfg.experiment_name, dataset_name, hyper_parameters_setting)

    cfg.checkpoint_dir = osp.join(cfg.output_dir, "checkpoint")
    cfg.log_dir = osp.join(cfg.output_dir, "log")
    cfg.tensorboard_dir = osp.join(cfg.output_dir, 'tensorboard')

    create_folder(cfg.checkpoint_dir)
    create_folder(cfg.log_dir)
    create_folder(cfg.tensorboard_dir)

    # update Weights and Biases
    cfg.wandb.start = args.enable_wandb
    cfg.wandb.project = args.project
    cfg.wandb.ckpt = args.wandb_ckpt

    cfg.freeze()


def get_base_cfg(args) -> CfgNode:
    """
        Get a copy of the default config.
        Returns:
            a fastreid CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()
