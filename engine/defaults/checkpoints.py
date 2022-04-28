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
from pathlib import Path
import torch
from timm.utils import get_state_dict
import glob
from utils.utils_folder import list_immediate_childfile_paths
from utils.utils_constant import TRAIN_PHASE, VAL_PHASE, TEST_PHASE


def auto_load_model(cfg, model, model_without_ddp, optimizer, model_ema=None):
    logger = logging.getLogger(__name__)

    output_dir = Path(cfg.checkpoint_dir)
    if cfg.train.auto_resume:

        all_checkpoints = glob.glob(osp.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            cfg.train.resume = osp.join(output_dir, f'checkpoint-{latest_ckpt}.pth')
        logger.info("Auto resume checkpoint: %s" % cfg.train.resume)

    resume_file = cfg.train.resume
    if resume_file:
        if resume_file.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume_file, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume_file, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info("Resume checkpoint %s" % resume_file)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            if isinstance(checkpoint['optimizer'], list):
                optimizer.load_state_dict(checkpoint['optimizer'][0])
                # TODO
                logger.warning("Here needs fixing")
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str):  # does not support resuming with 'best', 'best-ema'
                cfg.train.begin_epoch = checkpoint['epoch'] + 1
            else:
                assert cfg.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(cfg, 'model_ema') and cfg.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])


def get_latest_checkpoint(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None

    latest_checkpoint = checkpoint_saves_paths[0]
    # we define the format of checkpoint like "epoch_0_state.pth"
    latest_index = int(osp.basename(latest_checkpoint).split("_")[1])
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoint_save_file_name = osp.basename(checkpoint_save_path)
        now_index = int(checkpoint_save_file_name.split("_")[1])
        if now_index > latest_index:
            latest_checkpoint = checkpoint_save_path
            latest_index = now_index
    return latest_checkpoint


def get_all_checkpoints(checkpoint_save_folder):
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None
    checkpoints_list = []
    # we define the format of checkpoint like "epoch_0_state.pth"
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoints_list.append(checkpoint_save_path)
    return checkpoints_list


def save_model(cfg, epoch, model, model_without_ddp, optimizer, lr_scheduler, wd_scheduler, loss_scaler=None, model_ema=None):
    model_save_path = osp.join(cfg.checkpoint_dir, 'checkpoint-{}.pth'.format(epoch))

    optimizer_state_dict = []
    if isinstance(optimizer, list):
        for op in optimizer:
            optimizer_state_dict.append(op.state_dict())
    else:
        optimizer_state_dict.append(optimizer.state_dict())

    model_save_dict = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer_state_dict,
        'epoch': epoch,
    }

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, list):
            lr_scheduler_state_dict = []
            for item in lr_scheduler:
                lr_scheduler_state_dict.append(item.state_dict())
        else:
            lr_scheduler_state_dict = lr_scheduler.state_dict()
        model_save_dict['lr_scheduler'] = lr_scheduler_state_dict

    if wd_scheduler is not None:
        if isinstance(wd_scheduler, list):
            wd_scheduler_state_dict = []
            for item in wd_scheduler:
                wd_scheduler_state_dict.append(item.state_dict())
        else:
            wd_scheduler_state_dict = wd_scheduler.state_dict()
        model_save_dict['wd_scheduler'] = wd_scheduler_state_dict

    if model_ema is not None:
        model_save_dict['model_ema'] = get_state_dict(model_ema)

    torch.save(model_save_dict, model_save_path)

    return model_save_path


# def save_checkpoint(epoch, save_folder, model, optimizer, scheduler, **kwargs):
#     model_save_path = osp.join(save_folder, 'checkpoints_{}.pth'.format(epoch))
#     checkpoint_dict = dict()
#
#     # Because nn.DataParallel
#     # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/5
#     model_state_dict = model.state_dict()
#     if list(model_state_dict.keys())[0].startswith('module.'):
#         model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
#
#     checkpoint_dict['begin_epoch'] = epoch
#     checkpoint_dict['state_dict'] = model_state_dict
#
#     optimizer_state_dict = []
#     if isinstance(optimizer, list):
#         for op in optimizer:
#             optimizer_state_dict.append(op.state_dict())
#     else:
#         optimizer_state_dict.append(optimizer.state_dict())
#     checkpoint_dict['optimizer'] = optimizer_state_dict
#
#     scheduler_state_dict = []
#     if isinstance(scheduler, list):
#         for item in scheduler:
#             scheduler_state_dict.append(item.state_dict())
#     else:
#         scheduler_state_dict.append(scheduler.state_dict())
#     checkpoint_dict['scheduler'] = scheduler_state_dict
#
#     torch.save(checkpoint_dict, model_save_path)
#
#     return model_save_path


def resume(model, optimizer, checkpoint_file, **kwargs):
    checkpoint = torch.load(checkpoint_file)
    begin_epoch = checkpoint['begin_epoch'] + 1
    gpus = kwargs.get("gpus", [])
    if len(gpus) <= 1:
        state_dict = {k.replace('module.', '') if k.find('module') == 0 else k: v for k, v in checkpoint['state_dict'].items()}
        # state_dict = {k.replace('module.', '') if k.index('module') == 0 else k: v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint["state_dict"]

    state_dict = {k.replace('preact.', '') if k.find('preact') == 0 else k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    optimizer_state_dict = checkpoint['optimizer']
    if isinstance(optimizer, list):
        assert type(optimizer_state_dict) == type(optimizer)
        for opz in optimizer:
            for op_sd in optimizer_state_dict:
                if len(opz.state_dict()['param_groups'][0]['params']) == len(op_sd['param_groups'][0]['params']):
                    opz.load_state_dict(op_sd)
                    for state in opz.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                    optimizer_state_dict.remove(op_sd)
                    break
                else:
                    logger = logging.getLogger(__name__)
                    logger.error("bad resume")
    else:
        optimizer.load_state_dict(optimizer_state_dict[0])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    return model, optimizer, begin_epoch


def list_evaluate_model_files(cfg, phase, checkpoints_save_folder, eval_from_checkpoint_id=-1):
    subCfgNode = cfg.VAL if phase == VAL_PHASE else cfg.TEST
    evaluate_model_state_files = []
    if subCfgNode.MODEL_FILE:
        evaluate_model_state_files.append(subCfgNode.MODEL_FILE)
    else:
        if eval_from_checkpoint_id == -1:
            model_state_file = get_latest_checkpoint(checkpoints_save_folder)
            evaluate_model_state_files.append(model_state_file)
        else:
            candidate_model_files = get_all_checkpoints(checkpoints_save_folder)
            for model_file in candidate_model_files:
                model_file_epoch_num = int(osp.basename(model_file).split("_")[1])
                if model_file_epoch_num >= eval_from_checkpoint_id:
                    evaluate_model_state_files.append(model_file)
    return evaluate_model_state_files


def load_checkpoint(cfg, model, checkpoint_file, eval_mode, optimizer=None, scheduler=None, amp=None):
    logger = logging.getLogger(__name__)
    logger.info(f"==============> Resuming form {checkpoint_file}....................")
    if checkpoint_file.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_file, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    # max_accuracy = 0.0
    epoch = 0
    if not eval_mode:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch']
        if 'amp' in checkpoint and cfg.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{checkpoint_file}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()

    return model, epoch
