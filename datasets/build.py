#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import torch.utils.data
import torch.utils.data.distributed as dist
from utils.utils_distributed import get_rank, get_world_size
from utils.utils_registry import DATASET_REGISTRY
from utils.utils_distributed import is_dist_avail_and_initialized

__all__ = ["get_dataset_name", "build_train_dataloader", "build_eval_dataloader", "build_train_loader", "build_eval_loader"]


def get_dataset_name(cfg):
    dataset_name = cfg.dataset.name
    if dataset_name.startswith("PoseTrack"):
        dataset_version = "18" if cfg.dataset.is_posetrack18 else "17"
        dataset_name = dataset_name + dataset_version

    return dataset_name


def build_train_dataloader(cfg):
    distributed = is_dist_avail_and_initialized()
    dataset_name = cfg.dataset.name
    train_dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase="train")

    if distributed:
        train_sampler = dist.DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
        batch_size = cfg.train.batch_size_per_gpu
    else:
        #  nn.Dataparallel DP
        train_sampler = None
        batch_size = cfg.train.batch_size_per_gpu * len(cfg.gpus)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=cfg.pin_memory
    )
    return train_dataset, train_loader, train_sampler


# for val / test dataloader
def build_eval_dataloader(cfg, phase):
    distributed = is_dist_avail_and_initialized()
    dataset_name = cfg.dataset.name
    eval_dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase=phase)

    if phase == 'validate':
        batch_size_per_gpu = cfg.val.batch_size_per_gpu
    elif phase == 'test':
        batch_size_per_gpu = cfg.test.batch_size_per_gpu
    else:
        raise ValueError(f"Unknown phase :{phase}")

    if distributed:
        eval_sampler = dist.DistributedSampler(eval_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
        batch_size = batch_size_per_gpu
    else:
        #  nn.Dataparallel DP
        eval_sampler = None
        batch_size = batch_size_per_gpu * len(cfg.gpus)

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=cfg.pin_memory
    )
    return eval_dataset, eval_loader, eval_sampler


def build_train_loader(cfg, **kwargs):
    cfg = cfg.clone()

    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase='train')

    batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return train_loader


def build_eval_loader(cfg, phase):
    cfg = cfg.clone()

    # dataset_name = cfg.DATASET.NAME
    dataset_name = cfg.DATASET.NAME
    # dataset_name = cfg.DATASET.DATASET
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, phase=phase)
    if phase == 'validate':
        batch_size = cfg.VAL.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    elif phase == 'test':
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
    else:
        raise BaseException

    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return eval_loader
