#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import argparse
import os.path as osp


def default_parse_args():
    parser = argparse.ArgumentParser(description='FAMI-Pose training and evaluation script for human pose estimation')
    parser.add_argument('--cfg', help='name of configure file', type=str, required=True)

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train_with_val', action='store_true', default=True)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    # Following parameters will be merged into to the config
    # directory parameters
    parser.add_argument('--root_dir', type=str, default='../')

    # distributed parameters
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:55555', type=str, help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node/rank id for distributed training')


    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', action='store_true', default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', action='store_true', default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    #
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    args.rootDir = osp.abspath(args.root_dir)
    args.cfg = osp.join(args.rootDir, osp.abspath(args.cfg))

    return args


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
