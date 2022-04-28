# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description: 
"""
from utils.utils_registry import CORE_FUNCTION_REGISTRY


def build_core_function(cfg, **kwargs):
    core_function = CORE_FUNCTION_REGISTRY.get(cfg.core_function)(cfg, **kwargs)

    return core_function
