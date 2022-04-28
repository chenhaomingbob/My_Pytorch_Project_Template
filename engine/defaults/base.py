#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import logging
import os.path as osp
import time
from utils.utils_folder import create_folder
from utils.utils_logger import reset_logger
from tabulate import tabulate
from termcolor import colored


class BaseExecutor:
    def __init__(self, cfg, output_folders: dict, phase: str, **kwargs):
        self._hooks = []
        self.output_path_dict = {}
        self.cfg = cfg
        self.phase = phase
        self.checkpoints_save_folder, self.tb_save_folder, self.log_file = None, None, None
        self.local_rank = kwargs.get("local_rank")
        self.update_output_paths(output_folders, phase)

    def update_output_paths(self, output_paths, phase):
        self.checkpoints_save_folder = output_paths["checkpoints_save_folder"]
        self.tb_save_folder = output_paths["tb_save_folder"]
        self.log_save_folder = output_paths.get("log_save_folder", "./log")
        # log
        create_folder(self.log_save_folder)
        self.log_file = osp.join(self.log_save_folder, "{}-{}-R{}.log".format(phase, time.strftime("%Y_%m_%d_%H_%M"), self.local_rank))
        reset_logger(self.log_file, self.local_rank)

        self.show_info()

    def show_info(self):
        logger = logging.getLogger(__name__)
        table_header = ["Key", "Value"]
        table_data = [
            ["Phase", self.phase],
            ["Log Folder", self.log_file],
            ["Checkpoint Folder", self.checkpoints_save_folder],
            ["Tensorboard_save_folder", self.tb_save_folder],
        ]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Executor Operating Parameter Table: \n" + colored(table, "red"))

    def exec(self):
        raise NotImplementedError

    def __del__(self):
        pass
