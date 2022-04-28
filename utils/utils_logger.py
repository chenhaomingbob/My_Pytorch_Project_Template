#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import logging
from sheen import ColoredHandler


def reset_logger(log_file, dist_rank=0, logger=None, logger_level=logging.DEBUG):
    if logger is None:
        logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    logger.setLevel(logger_level)

    # create console handlers for master process
    if dist_rank == 0:
        logger.addHandler(ColoredHandler())


    # create file handlers for all processes
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
