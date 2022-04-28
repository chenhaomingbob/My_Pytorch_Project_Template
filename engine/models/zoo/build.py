#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
from utils.utils_registry import MODEL_REGISTRY


def build_model(cfg, phase='train', **kwargs):
    """
        return : model Instance
    """
    model_name = cfg.model.name

    model_instance = MODEL_REGISTRY.get(model_name)(cfg, phase, **kwargs)
    #
    # # if phase == TRAIN_PHASE and cfg.MODEL.INIT_WEIGHTS:
    # #     model_instance.train()
    #
    # if phase != TRAIN_PHASE:
    #     model_instance.eval()

    return model_instance


def get_model_hyperparameter(cfg, **kwargs):
    """
        return : model class
    """
    model_name = cfg.model.name

    hyper_parameters_setting = MODEL_REGISTRY.get(model_name).get_model_hyper_parameters(cfg)

    return hyper_parameters_setting
