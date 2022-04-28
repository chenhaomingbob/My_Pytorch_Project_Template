#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import os
import sys

import random
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('../../'))
from engine import get_cfg, update_config, DefaultRunner
from argument_parser import default_parse_args

world_size = 3


def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)
    return cfg


def set_random_seed():
    random.seed(19970808)
    np.random.seed(19970808)
    torch.random.manual_seed(19970808)


def main():
    args = default_parse_args()
    cfg = setup(args)
    set_random_seed()
    runner = DefaultRunner(cfg, args)
    runner.launch()


if __name__ == '__main__':
    main()
