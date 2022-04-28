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

import numpy as np
import torch
import torch.distributed as dist

from datasets.commond import get_final_preds
from engine.functions.zoo.base import BaseFunction, AverageMeter
from engine.functions.utils.evaluate import accuracy
from utils.utils_constant import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
from utils.utils_registry import CORE_FUNCTION_REGISTRY
from engine.losses import JointMSELoss
from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_distributed import MetricLogger, TensorboardLogger, WandbLogger, is_main_process, get_world_size

from datasets.commond.pose_process import flip_back


@CORE_FUNCTION_REGISTRY.register()
class DemoFunction(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.distributed = kwargs.get("distributed")
        self.criterion = kwargs.get("criterion", None)
        self.tb_log_dir = kwargs.get("tb_log_dir", None)
        self.writer_dict = kwargs.get("writer_dict", None)
        self.PE_Name = kwargs.get("PE_Name", "DCPOSE")
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0

        self.HeatmapMSELOSS_criterion = JointMSELoss()
        self.FeatmapMSELOSS_criterion = JointMSELoss(use_target_weight=False)

        self.use_mse_loss = self.cfg.loss.heatmap_mse.use

        self.mse_weight = self.cfg.loss.heatmap_mse.weight

    def train(self, model, epoch, optimizer, train_dataloader, tensorboard_logger: TensorboardLogger, wandb_logger: WandbLogger, **kwargs):
        logger = logging.getLogger(__name__)

        model.train(True)

        metric_logger = MetricLogger(epoch, delimiter="\t", logger=logger)
        metric_logger.add_meters(
            ["loss", "loss_mse", "loss_feat", "loss_sup_mse", "loss_mi", "loss_mi_1", "loss_mi_2", "loss_mi_3", "loss_mi_4", "loss_mi_5", "loss_mi_6"], group=0)
        metric_logger.add_meters(
            ["acc", "acc_kf_backbone"], group=1)

        for data_iter_step, batch_data in enumerate(metric_logger.train_log_every(train_dataloader, self.cfg.print_freq)):
            input_x, sup_x, target_heatmaps, target_heatmaps_weight, meta = batch_data

            input_x = input_x.cuda(non_blocking=True)
            sup_x = sup_x.cuda(non_blocking=True)
            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)

            pred_heatmaps, local_warped_sup_hm_list, kf_bb_heatmaps, mi_loss_list, kf_bb_feat, aligned_sup_feat_list = model(input_x, sup_x)

            #
            loss_mse = self.HeatmapMSELOSS_criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight) * self.mse_weight
            metric_logger.update(loss_mse=loss_mse)
            loss = loss_mse

            #
            self.feat_weight = 0.5
            loss_feat = 0
            for aligned_sup_feat in aligned_sup_feat_list:
                loss_feat += self.FeatmapMSELOSS_criterion(aligned_sup_feat, kf_bb_feat, None) * self.feat_weight
            metric_logger.update(loss_feat=loss_feat)
            loss += loss_feat

            loss_sup_mse = 0
            for hm in local_warped_sup_hm_list:
                loss_sup_mse += self.HeatmapMSELOSS_criterion(hm, target_heatmaps, target_heatmaps_weight) * self.mse_weight
            metric_logger.update(loss_sup_mse=loss_sup_mse)
            loss += loss_sup_mse

            loss_mi, alpha, beta = 0, 1, 0.1
            # -1 * beta {I}( {y}_{t} ;  \boldsymbol{\widetilde{z}}_{t+\delta})
            if len(mi_loss_list) > 0:
                loss_mi_1 = mi_loss_list[0]
                metric_logger.update(loss_mi_1=loss_mi_1)
                loss_mi += loss_mi_1 * beta * -1
                # + * beta {I}( {z}_{t} ;  \boldsymbol{\widetilde{z}}_{t+\delta})
                loss_mi_2 = mi_loss_list[1]
                metric_logger.update(loss_mi_2=loss_mi_2)
                loss_mi += loss_mi_2 * beta
                # + {I}( {y}_{t} ;  {z}_{t+\delta})
                loss_mi_3 = mi_loss_list[2]
                metric_logger.update(loss_mi_3=loss_mi_3)
                loss_mi += loss_mi_3
                # - {I}( {z}_{t+\delta} ; \boldsymbol{\widetilde{z}}_{t+\delta})
                loss_mi_4 = mi_loss_list[3]
                metric_logger.update(loss_mi_4=loss_mi_4)
                loss_mi += loss_mi_4 * -1
                # + {I}( {y}_{t}   ; {z}_{t})
                loss_mi_5 = mi_loss_list[4]
                metric_logger.update(loss_mi_5=loss_mi_5)
                loss_mi += loss_mi_5
                # - {I}( {z}_{t}   ; \boldsymbol{\widetilde{z}}_{t+\delta})
                loss_mi_6 = mi_loss_list[5]
                metric_logger.update(loss_mi_6=loss_mi_6)
                loss_mi += loss_mi_6 * -1
                #
                loss_mi = loss_mi * alpha
                metric_logger.update(loss_mi=loss_mi)
                loss += loss_mi
            metric_logger.update(loss=loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            _, avg_acc, cnt1, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            metric_logger.update(acc=avg_acc)

            _, kf_bb_hm_acc, cnt1, _ = accuracy(kf_bb_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            metric_logger.update(acc_kf_backbone=kf_bb_hm_acc)

            # tensorboard
            if tensorboard_logger is not None:
                tensorboard_logger.update(loss=loss, head='loss')
                tensorboard_logger.update(acc=avg_acc, head='acc')
                tensorboard_logger.set_step()

            # wandb
            if wandb_logger is not None:
                wandb_logger._wandb.log({
                    'Rank-0 Batch Wise/train_loss': loss.item(),
                }, commit=False)
                wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': tensorboard_logger.step})

        metric_logger.synchronize_between_processes()
        logger.info(f"Averaged states {str(metric_logger)}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def val(self, model, val_dataloader, tensorboard_logger: TensorboardLogger, wandb_logger: WandbLogger, **kwargs):
        logger = logging.getLogger(__name__)
        model.eval()

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        model_run_time, data_process_time = AverageMeter(), AverageMeter()
        acc_kf_backbone = AverageMeter()
        phase = VAL_PHASE

        # self.max_iter_num = len(dataloader)
        # self.dataloader_iter = iter(dataloader)
        # dataset = dataloader.dataset
        # prepare data fro validate
        if is_main_process():
            import math
            num_samples = len(val_dataloader.dataset)
            num_samples = math.ceil(num_samples / get_world_size()) * get_world_size()

            all_preds = np.zeros((num_samples, self.cfg.model.num_joints, 3), dtype=np.float)
            all_bb = np.zeros((num_samples, self.cfg.model.num_joints, 3), dtype=np.float)
            all_boxes = np.zeros((num_samples, 6))
            image_path = []
            filenames = []
            filenames_map = {}
            filenames_counter = 0
            imgnums = []
            idx = 0
            acc_threshold = 0.7
        ##
        # assert phase in [VAL_PHASE, TEST_PHASE]
        # if phase == VAL_PHASE:
        #     FLIP_TEST = self.cfg.VAL.FLIP
        #     SHIFT_HEATMAP = True
        # elif phase == TEST_PHASE:
        #     FLIP_TEST = self.cfg.TEST.FLIP
        #     SHIFT_HEATMAP = True
        ###
        result_output_dir, vis_output_dir = self.vis_info(logger, phase, kwargs.get("epoch", "xxx"))
        end = time.time()
        for iter_step, batch_data in enumerate(val_dataloader):
            key_frame_input, sup_frame_input, target_heatmaps, target_heatmaps_weight, meta = batch_data
            data_time.update(time.time() - end)

            key_frame_input = key_frame_input.cuda(non_blocking=True)
            sup_frame_input = sup_frame_input.cuda(non_blocking=True)
            target_heatmaps = target_heatmaps.cuda(non_blocking=True)

            t = time.time()
            pred_heatmaps, kf_bb_hm = model(key_frame_input, sup_frame_input)
            model_run_time.update(time.time() - t)

            # if FLIP_TEST:
            #     input_key_flipped = key_frame_input.flip(3)
            #     input_sup_flipped = sup_frame_input.flip(3)
            #
            #     pred_heatmaps_flipped, kf_bb_hm_flipped = model(input_key_flipped.cuda(), input_sup_flipped.cuda())
            #
            #     pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(), dataset.flip_pairs)
            #     kf_bb_hm_flipped = flip_back(kf_bb_hm_flipped.cpu().numpy(), dataset.flip_pairs)
            #
            #     pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
            #     kf_bb_hm_flipped = torch.from_numpy(kf_bb_hm_flipped.copy()).cuda()
            #
            #     if SHIFT_HEATMAP:
            #         pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :, 0:-1]
            #         kf_bb_hm_flipped[:, :, :, 1:] = kf_bb_hm_flipped.clone()[:, :, :, 0:-1]
            #     pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
            #     kf_bb_hm = (kf_bb_hm + kf_bb_hm_flipped) * 0.5

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            _, kf_bb_hm_acc, cnt1, _ = accuracy(kf_bb_hm.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc_kf_backbone.update(kf_bb_hm_acc, cnt1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            dist.barrier()
            p_t = time.time()
            if is_main_process():
                word_size = get_world_size()
                gather_output_list = {
                    'image': [None for _ in range(word_size)],
                    'center': [None for _ in range(word_size)],
                    'scale': [None for _ in range(word_size)],
                    'score': [None for _ in range(word_size)],
                    'pred_heatmaps': [torch.empty(size=pred_heatmaps.size()).cuda() for _ in range(word_size)],
                    'kf_bb_hm': [torch.empty(size=kf_bb_hm.size()).cuda() for _ in range(word_size)]
                }
            else:
                gather_output_list = None
            dist.gather_object(meta['image'], gather_output_list['image'] if is_main_process() else None)
            dist.gather_object(meta['center'], gather_output_list['center'] if is_main_process() else None)
            dist.gather_object(meta['scale'], gather_output_list['scale'] if is_main_process() else None)
            dist.gather_object(meta['score'], gather_output_list['score'] if is_main_process() else None)
            dist.gather(pred_heatmaps, gather_output_list['pred_heatmaps'] if is_main_process() else None)
            dist.gather(kf_bb_hm, gather_output_list['kf_bb_hm'] if is_main_process() else None)
            ##
            if is_main_process():

                image = []
                for image_item in gather_output_list['image']:
                    image.extend(image_item)
                center = torch.cat(gather_output_list['center'], dim=0)
                scale = torch.cat(gather_output_list['scale'], dim=0)
                score = torch.cat(gather_output_list['score'], dim=0)
                pred_heatmaps = torch.cat(gather_output_list['pred_heatmaps'], dim=0)
                kf_bb_hm = torch.cat(gather_output_list['kf_bb_hm'], dim=0)

                num_images = len(image)
                #### for eval ####
                for ff in range(len(image)):
                    cur_nm = image[ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = center.numpy()
                scale = scale.numpy()
                score = score.numpy()

                # logger.info(f"num_images: {num_images}\t size_pred_heatmaps: {pred_heatmaps.size()}\t idx: {idx}\t len_all_preds:{len(all_preds)}")

                pred_coord, our_maxvals = get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, :2] = pred_coord
                all_preds[idx:idx + num_images, :, 2:3] = our_maxvals

                bb_coord, bb_maxvals = get_final_preds(kf_bb_hm.clone().cpu().numpy(), center, scale)
                all_bb[idx:idx + num_images, :, :2] = bb_coord
                all_bb[idx:idx + num_images, :, 2:3] = bb_maxvals

                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score

                image_path.extend(image)
                idx += num_images
            data_process_time.update(time.time() - p_t)
            dist.barrier()

            if iter_step % self.cfg.print_freq == 0 or iter_step + 1 == len(val_dataloader):
                msg = 'Val: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Model Run Time {model_run_time.val:.3f}s ({model_run_time.avg:.3f}s)\t' \
                      'Data Process Time {data_process_time.val:.3f}s ({data_process_time.avg:.3f}s)\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(iter_step, len(val_dataloader),
                                                                        batch_time=batch_time, model_run_time=model_run_time,
                                                                        data_process_time=data_process_time,
                                                                        data_time=data_time, acc=acc)
                logger.info(msg)

        logger.info('########################################')
        logger.info('{}'.format(self.cfg.experiment_name))
        model_name = self.cfg.model.name

        if is_main_process():
            acc_printer = PredsAccPrinter(self.cfg, all_boxes, val_dataloader.dataset, filenames, filenames_map, imgnums, model_name, result_output_dir,
                                          self._print_name_value)
            logger.info("====> Predicting key frame heatmaps by the backbone network")
            bb_mAP = acc_printer(all_bb)
            logger.info("====> Predicting key frame heatmaps by the local warped hm")
            pred_mAP = acc_printer(all_preds)
        else:
            bb_mAP = None
            pred_mAP = None

        dist.barrier()
        dist.broadcast_object_list([pred_mAP])

        logger.info(pred_mAP)
        return pred_mAP

    @torch.no_grad()
    def test(self, model, dataloader, tb_writer_dict, **kwargs):
        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        acc_kf_backbone = AverageMeter()
        phase = TEST_PHASE
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_bb = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        acc_threshold = 0.7
        ##
        assert phase in [VAL_PHASE, TEST_PHASE]
        if phase == VAL_PHASE:
            FLIP_TEST = self.cfg.val.flip
            SHIFT_HEATMAP = True
        elif phase == TEST_PHASE:
            FLIP_TEST = self.cfg.test.flip
            SHIFT_HEATMAP = True
        ###
        result_output_dir, vis_output_dir = self.vis_info(logger, phase, epoch)
        ###
        logger.info("PHASE:{}, FLIP_TEST:{}, SHIFT_HEATMAP:{}".format(phase, FLIP_TEST, SHIFT_HEATMAP))
        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                key_frame_input, sup_frame_input, target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)
                data_time.update(time.time() - end)
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                pred_heatmaps, kf_bb_hm = model(key_frame_input.cuda(), sup_frame_input.cuda(), iter_step=iter_step)

                if FLIP_TEST:
                    input_key_flipped = key_frame_input.flip(3)
                    input_sup_flipped = sup_frame_input.flip(3)

                    pred_heatmaps_flipped, kf_bb_hm_flipped = model(input_key_flipped.cuda(), input_sup_flipped.cuda())

                    pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(), dataset.flip_pairs)
                    kf_bb_hm_flipped = flip_back(kf_bb_hm_flipped.cpu().numpy(), dataset.flip_pairs)

                    pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
                    kf_bb_hm_flipped = torch.from_numpy(kf_bb_hm_flipped.copy()).cuda()

                    if SHIFT_HEATMAP:
                        pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :, 0:-1]
                        kf_bb_hm_flipped[:, :, :, 1:] = kf_bb_hm_flipped.clone()[:, :, :, 0:-1]
                    pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
                    kf_bb_hm = (kf_bb_hm + kf_bb_hm_flipped) * 0.5

                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                _, kf_bb_hm_acc, cnt1, _ = accuracy(kf_bb_hm.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
                acc_kf_backbone.update(kf_bb_hm_acc, cnt1)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = 'Val: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(iter_step, num_batch,
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, acc=acc)
                    logger.info(msg)

                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = key_frame_input.size(0)

                pred_coord, our_maxvals = get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, :2] = pred_coord
                all_preds[idx:idx + num_images, :, 2:3] = our_maxvals

                bb_coord, bb_maxvals = get_final_preds(kf_bb_hm.clone().cpu().numpy(), center, scale)
                all_bb[idx:idx + num_images, :, :2] = bb_coord
                all_bb[idx:idx + num_images, :, 2:3] = bb_maxvals

                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score

                image_path.extend(meta['image'])
                idx += num_images

                self.global_steps += 1

                self.vis_hook(meta["image"], pred_coord, our_maxvals, vis_output_dir, center, scale)

        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))
        model_name = self.cfg.MODEL.NAME

        acc_printer = PredsAccPrinter(self.cfg, all_boxes, dataset, filenames, filenames_map, imgnums, model_name, result_output_dir, self._print_name_value)
        logger.info("====> Predicting key frame heatmaps by the backbone network")
        acc_printer(all_bb)
        logger.info("====> Predicting key frame heatmaps by the local warped hm")
        acc_printer(all_preds)
        tb_writer_dict["global_steps"] = self.global_steps
        self.tb_writer.close()
    def vis_info(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir, "use_gt_box" if self.cfg.val.use_gt_bbox else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if self.cfg.debug.vis_skeleton:
            logger.info("=> VIS_SKELETON")
        if self.cfg.debug.vis_bbox:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir

    def vis_hook(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg

        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from engine.functions.utils.vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON, vis_bbox=cfg.DEBUG.VIS_BBOX,
                                          sure_threshold=0.1)


class PredsAccPrinter(object):
    def __init__(self, cfg, all_boxes, dataset, filenames, filenames_map, imgnums, model_name, result_output_dir, print_name_value_func):
        self.cfg = cfg
        self.all_boxes = all_boxes
        self.dataset = dataset
        self.filenames = filenames
        self.filenames_map = filenames_map
        self.imgnums = imgnums
        self.model_name = model_name
        self.result_output_dir = result_output_dir
        self.print_name_value_func = print_name_value_func

    def __call__(self, pred_result):
        name_values, perf_indicator = self.dataset.evaluate(self.cfg, pred_result, self.result_output_dir, self.all_boxes, self.filenames_map,
                                                            self.filenames, self.imgnums)
        if isinstance(name_values, list):
            for name_value in name_values:
                self.print_name_value_func(name_value, self.model_name)
        else:
            self.print_name_value_func(name_values, self.model_name)

        return name_values
