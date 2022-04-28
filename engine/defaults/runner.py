#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
import os
import os.path as osp
import datetime
import torch.cuda
import logging
from timm.utils import ModelEma

from engine.models.zoo import build_model

from datasets import build_train_dataloader, build_eval_dataloader
from engine.functions import build_core_function
from utils.utils_constant import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
from engine.losses import build_loss
from engine.optimizer import build_optimizer, build_lr_scheduler
from .checkpoints import get_latest_checkpoint, resume, list_evaluate_model_files, load_checkpoint, auto_load_model, save_model
import time
from utils.utils_logger import reset_logger
from tabulate import tabulate
from termcolor import colored
import torch.distributed as dist
from tensorboardX import SummaryWriter
from utils.utils_distributed import get_rank, is_main_process, TensorboardLogger, WandbLogger
import numpy as np
import torch.backends.cudnn as cudnn
from typing import Optional
from utils.utils_distributed import is_dist_avail_and_initialized, get_world_size


class DefaultRunner:
    def __init__(self, cfg, args, **kwargs):
        self.cfg = cfg
        # args
        self.train = args.train
        self.val = args.val
        self.test = args.test
        self.train_with_val = args.train_with_val
        #
        self.distributed = is_dist_avail_and_initialized()
        self.val_from_checkpoint_id = 1

        suffix = ""
        suffix += "-train" if self.train else ""
        suffix += "-val" if self.val else ""
        suffix += "-test" if self.test else ""

        log_file = osp.join(self.cfg.log_dir, "{}{}-R{}.log".format(time.strftime("%Y_%m_%d_%H_%M"), suffix, self.cfg.dist.rank))
        reset_logger(log_file, self.cfg.dist.rank)

        # fix the seed for reproducibility
        seed = self.cfg.seed + self.cfg.dist.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        if self.cfg.dist.rank == 0:
            self.tensorboard_logger = TensorboardLogger(log_dir=self.cfg.tensorboard_dir)
        else:
            self.tensorboard_logger = None
        if self.cfg.dist.rank == 0 and self.cfg.wandb.start:
            self.wandb_logger = WandbLogger(self.cfg)
        else:
            self.wandb_logger = None

        self.core_function = build_core_function(self.cfg)

        # model
        self.model = build_model(self.cfg)
        self.model.to(self.cfg.dist.local_rank)

        self.model_ema = None
        if self.cfg.ema.start:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(
                self.model,
                decay=self.cfg.ema.decay,
                device='cpu' if self.cfg.ema.force_cpu else '',
                resume='')
            logging.info("Using EMA with decay = %.8f" % self.cfg.ema.decay)

        if self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.cfg.dist.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.cfg.dist.local_rank], find_unused_parameters=False)
        else:
            self.model = torch.nn.parallel.DataParallel(self.model)

    def launch(self):

        if self.train:
            self.__launch_train()

        if self.val:
            self.__launch_eval(VAL_PHASE)

        if self.test:
            self.__launch_eval(TEST_PHASE)

    def __launch_train(self):
        # GR: global rank ; LR: local rank
        logger = logging.getLogger(__name__)

        train_dataset, train_dataloader, train_sampler = build_train_dataloader(self.cfg)
        total_train_batch_size = self.cfg.train.batch_size_per_gpu * self.cfg.train.update_freq * get_world_size()
        if self.train_with_val:
            val_dataset, val_dataloader, val_sampler = build_eval_dataloader(self.cfg, VAL_PHASE)
        else:
            val_dataset, val_dataloader, val_sampler = None, None, None

        table_data = [
            ("Phase", TRAIN_PHASE),
            ("Output Dir", self.cfg.output_dir),
            ("Model Name", self.cfg.model.name),
            ("Training Batch Size", total_train_batch_size),
            ("Gradient Update frequent", self.cfg.train.update_freq),
            ("Number of training examples", len(train_dataset)),
        ]
        if self.train_with_val:
            table_data.extend([
                ("Train with val", self.train_with_val), ("Number of val examples", len(val_dataset))
            ])

        table = tabulate(table_data, tablefmt="pipe", headers=["Key", "Value"], numalign="left")
        logger.info(f"=> Executor Operating Parameter Table: \n" + colored(table, "red"))

        optimizer = build_optimizer(self.cfg, self.model)
        lr_scheduler = build_lr_scheduler(self.cfg, optimizer)
        wd_scheduler = None  # TODO weight_decay
        # model resume
        auto_load_model(cfg=self.cfg, model=self.model, model_without_ddp=self.model.module, optimizer=optimizer, )

        max_accuracy = 0.0
        if self.cfg.ema.start and self.cfg.ema.eval:
            max_accuracy_ema = 0.0

        start_time = time.time()
        for epoch in range(self.cfg.train.begin_epoch, self.cfg.train.end_epoch):
            if self.distributed:
                train_sampler.set_epoch(epoch)
            if self.tensorboard_logger:
                pass
            if self.wandb_logger:
                self.wandb_logger.set_steps()

            last_lr = []
            if isinstance(lr_scheduler, list):
                for lr_s in lr_scheduler:
                    last_lr.append(lr_s.get_last_lr())
            else:
                last_lr.append(lr_scheduler.get_last_lr())

            logger.info('=> Start train epoch {}, last_lr{}'.format(epoch, last_lr))
            train_stats = self.core_function.train(model=self.model, epoch=epoch, optimizer=optimizer, train_dataloader=train_dataloader,
                                                   tensorboard_logger=self.tensorboard_logger, wandb_logger=self.wandb_logger)

            if isinstance(lr_scheduler, list):
                for lr_s in lr_scheduler:
                    lr_s.step()
            else:
                lr_scheduler.step()
            if self.cfg.dist.rank == 0 and (epoch % self.cfg.train.save_model_freq == 0 or (epoch + 1 == self.cfg.train.end_epoch)):
                model_save_path = save_model(self.cfg, epoch, None, self.model.module, optimizer, lr_scheduler, wd_scheduler, None, self.model_ema)
                logger.info('=> Save epoch {} model in {}'.format(epoch, model_save_path))

            if self.train_with_val:
                val_stats = self.core_function.val(model=self.model, val_dataloader=val_dataloader,
                                                   tensorboard_logger=self.tensorboard_logger, wandb_logger=self.wandb_logger, epoch=epoch)
                logger.info(f"Accuracy of the model on the {len(val_dataset)} val images: {val_stats['acc']:.1f}%")
                if max_accuracy < val_stats["Mean"]:
                    max_accuracy = val_stats["Mean"]
                    best_model_path = save_model(self.cfg, 'best', self.model, self.model.module, optimizer, lr_scheduler, wd_scheduler, None, self.model_ema)
                    logger.info("=> Save best (epoch {}) model with max accuracy of {.4f} in {}".format(epoch, max_accuracy, best_model_path))

                if self.tensorboard_logger is not None:
                    self.tensorboard_logger.update(val_mAP=val_stats['Mean'], head="perf", step=epoch)
                    # self.tensorboard_logger.update(test_acc5=val_stats['mAP'], head="perf", step=epoch)
                    # self.tensorboard_logger.update(test_loss=val_stats['mAP'], head="perf", step=epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},  # ** 会把后面的东西弄成一个字典
                             **{f'val_{k}': v for k, v in val_stats.items()},  # ** 会把后面的东西弄成一个字典
                             'epoch': epoch,
                             # 'n_parameters': n_parameters
                             }
                # repeat testing routines for EMA, if ema eval is turned on
                if self.cfg.ema.start and self.cfg.ema.eval:
                    val_stats_ema = self.core_function.val(model=self.model_ema, epoch=epoch, val_dataloader=val_dataloader)
                    logger.info(f"Accuracy of the model EMA on the {len(val_dataset)} val images: {val_stats_ema['acc']:.1f}%")
                    if max_accuracy_ema < val_stats_ema["mAP"]:
                        max_accuracy_ema = val_stats_ema["mAP"]

                        save_model(self.cfg, 'best-ema', self.model, self.model.module, optimizer, lr_scheduler, wd_scheduler, None, self.model_ema)
                        logger.info(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                    if self.tensorboard_logger is not None:
                        self.tensorboard_logger.update(val_mAP_ema=val_stats_ema['mAP'], head="perf", step=epoch)
                    log_stats.update({**{f'val_{k}_ema': v for k, v in val_stats_ema.items()}})

            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                             # 'n_parameters': n_parameters
                             }

            logger.info(log_stats)

            if self.wandb_logger:
                self.wandb_logger.log_epoch_metrics(log_stats)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

    @torch.no_grad()
    def __launch_eval(self, phase):
        logger = logging.getLogger(__name__)

        eval_dataloader, eval_sampler = build_eval_dataloader(self.cfg, phase, distributed=self.distributed)

        eval_from_checkpoint_id = 1
        table_header = ["Key", "Value"]
        table_data = [
            ["Phase", phase],
        ]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Executor Operating Parameter Table: \n" + colored(table, "red"))

        eval_checkpoint_files = list_evaluate_model_files(self.cfg, VAL_PHASE, self.checkpoints_save_folder, eval_from_checkpoint_id)

        for checkpoint_file in eval_checkpoint_files:
            model, epoch = load_checkpoint(self.cfg, self.model, checkpoint_file, eval_mode=True)
            if phase == VAL_PHASE:
                self.core_function.val(model=model, epoch=epoch, val_dataloader=eval_dataloader, tb_writer_dict=self.tb_writer_dict)
            elif phase == TEST_PHASE:
                self.core_function.test(model=model, epoch=epoch, test_dataloader=eval_dataloader, tb_writer_dict=self.tb_writer_dict)
            else:
                error_msg = f"Unknown Phase: {phase}"
                logger.error(error_msg)
                ValueError(error_msg)
