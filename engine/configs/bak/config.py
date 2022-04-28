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

    if args.rootDir:
        cfg.ROOT_DIR = args.rootDir

    cfg.OUTPUT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.OUTPUT_DIR))

    cfg.DATASET.JSON_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.JSON_DIR))
    cfg.DATASET.IMG_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.IMG_DIR))
    cfg.DATASET.TEST_IMG_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.DATASET.TEST_IMG_DIR))

    if len(cfg.MODEL.PRETRAINED) > 0:
        cfg.MODEL.PRETRAINED = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.MODEL.PRETRAINED))
    cfg.MODEL.BACKBONE_PRETRAINED = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.MODEL.BACKBONE_PRETRAINED))

    cfg.VAL.ANNOT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.VAL.ANNOT_DIR))
    cfg.VAL.COCO_BBOX_FILE = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.VAL.COCO_BBOX_FILE))

    cfg.TEST.ANNOT_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.TEST.ANNOT_DIR))
    cfg.TEST.COCO_BBOX_FILE = os.path.abspath(os.path.join(cfg.ROOT_DIR, cfg.TEST.COCO_BBOX_FILE))

    cfg.freeze()


def get_cfg(args) -> CfgNode:
    """
        Get a copy of the default config.
        Returns:
            a fastreid CfgNode instance.
    """
    from .defaults import _C
    from .mppe_config import _C as MPPE_C

    if args.use_mppe_config:
        return MPPE_C.clone()
    else:
        return _C.clone()
