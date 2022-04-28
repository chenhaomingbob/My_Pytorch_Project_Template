#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description:
"""
from .config import CfgNode

_C = CfgNode()
# distributed parameters
_C.dist = CfgNode()
_C.dist.dist_url = "tcp://127.0.0.1:55555"  # url used to set up distributed training/evaluation
_C.dist.world_size = 1
_C.dist.rank = 0  # rank in the world
_C.dist.local_rank = 0  # rank in the machine
# Weights and Biases
_C.wandb = CfgNode()
_C.wandb.start = False
_C.wandb.project = "My_Pytorch_DeepLearning_Template"
_C.wandb.ckpt = False
# ema
_C.ema = CfgNode()
_C.ema.start = False  # Creating model ema
_C.ema.decay = 0.9999
_C.ema.force_cpu = False
_C.ema.eval = False  # 'Using ema to eval during training.'
#
_C.seed = 8
_C.root_dir = ''
_C.experiment_name = ''
_C.output_dir = ''
_C.save_heatmaps = False
_C.load_heatmaps = False
_C.save_preds = False
_C.preds_sfx = ''
_C.load_preds = False
_C.save_offsets = False
_C.log_dir = ''
_C.data_dir = ''
_C.model_dir = ''
_C.gpus = (0,)
_C.workers = 8
_C.print_freq = 20
_C.pin_memory = False
_C.distance = 2
_C.number_sup = 2
_C.core_function = ""

# cudnn related params
_C.cudnn = CfgNode()
_C.cudnn.benchmark = True
_C.cudnn.deterministic = False
_C.cudnn.enabled = True

# common params for network
_C.model = CfgNode()
_C.model.name = 'pose_hrnet'
_C.model.init_weights = True
_C.model.freeze_weights = False
_C.model.freeze_prednet_weights = True
_C.model.pretrained = ''
_C.model.backbone_pretrained = ''
_C.model.num_joints = 17
_C.model.target_type = 'gaussian'
_C.model.image_size = [256, 256]  # width * height, ex: 192 * 256
_C.model.heatmap_size = [64, 64]  # width * height, ex: 24 * 32
_C.model.sigma = 2
_C.model.extra = CfgNode(new_allowed=True)
_C.model.deforam_conv_version = 1
_C.model.deformable_conv = CfgNode(new_allowed=True)
_C.model.use_rectifier = True
_C.model.use_margin = True
_C.model.use_group = True
_C.model.high_resolution = False
_C.model.freeze_hrnet_weights = False
_C.model.mpii_pretrained = False
_C.model.use_warping_train = True
_C.model.use_warping_test = True
_C.model.warping_reverse = False
_C.model.use_gt_input_test = False
_C.model.use_gt_input_train = False
_C.model.evaluate = True
_C.model.dilation_exp = 0
_C.model.visualize_offsets = False
_C.model.use_pixel_level_offset = True
_C.model.use_prf = True
_C.model.prf_basicblock_num = 10
_C.model.prf_inner_ch = 12
_C.model.use_ptm = True
_C.model.ptm_basicblock_num = 10
_C.model.ptm_inner_ch = 12
_C.model.prf_ptm_combine_inner_ch = 10
_C.model.prf_ptm_combine_basicblock_num = 10
_C.model.use_pcn = True
_C.model.temporal_interpolation = False
_C.model.backbone_precompute = False
_C.model.with_dcpose = False

#### loss ####
_C.loss = CfgNode()

_C.loss.grad_max_norm = 0.02
_C.loss.use_different_joints_weight = False

_C.loss.heatmap_mse = CfgNode()
_C.loss.heatmap_mse.use = True
_C.loss.heatmap_mse.weight = 1.0
_C.loss.heatmap_mse.divided_num_joints = True

_C.loss.featmap_mse = CfgNode()
_C.loss.featmap_mse.use = False
_C.loss.featmap_mse.weight = 0.5

#### dataset ####
_C.dataset = CfgNode()
_C.dataset.random_aux_frame = True
_C.dataset.root = ''
_C.dataset.name = ''
_C.dataset.dataset = 'mpii'
_C.dataset.train_set = 'train'
_C.dataset.test_set = 'test'
_C.dataset.val_set = 'val'
_C.dataset.hybrid_joints_type = ''
_C.dataset.select_data = False
_C.dataset.test_on_train = False
_C.dataset.json_file = ''
_C.dataset.json_dir = ''
_C.dataset.posetrack17_json_dir = ''
_C.dataset.posetrack18_json_dir = ''
_C.dataset.img_dir = ''
_C.dataset.posetrack17_img_dir = ''
_C.dataset.posetrack18_img_dir = ''
_C.dataset.is_posetrack18 = False
_C.dataset.color_rgb = False
_C.dataset.test_img_dir = ''
_C.dataset.posetrack17_test_img_dir = ''
_C.dataset.posetrack18_test_img_dir = ''
_C.dataset.input_type = ''
_C.dataset.bbox_enlarge_factor = 1.0

_C.dataset.use_global_ref = False
_C.dataset.use_local_ref = False
_C.dataset.num_ref = 0
##
_C.dataset.split_version = 1

#### train ####
_C.train = CfgNode()  # cfg.Node
_C.train.update_freq = 1  # gradient accumulation steps
_C.train.save_model_freq = 2
_C.train.batch_size_per_gpu = 32
_C.train.auto_resume = True
_C.train.resume = ""
# _C.train.save_model_per_epoch = 2

_C.train.shuffle = True
_C.train.loss_alpha = 1.0
_C.train.loss_beta = 1.0
_C.train.loss_gama = 1.0
_C.train.lr_factor = 0.1
_C.train.lr_step = [90, 110]
_C.train.milestones = [8, 12, 16]
_C.train.gamma = 0.99
_C.train.lr = 0.001
_C.train.stsn_lr = 0.001
_C.train.optimizer = 'Adam'
_C.train.momentum = 0.9
_C.train.wd = 0.0001
_C.train.nesterov = False
_C.train.gamma1 = 0.99
_C.train.gamma2 = 0.0
_C.train.begin_epoch = 0
_C.train.end_epoch = 140
_C.train.auto_resume = False
_C.train.flip = True
_C.train.scale_factor = 0.25
_C.train.rot_factor = 30
_C.train.prob_half_body = 0.0
_C.train.num_joints_half_body = 8
_C.train.lr_scheduler = 'multisteplr'
##
_C.train.lr_second_group = [None]
_C.train.lr_second_group_value = 1e-6
_C.train.random_sample_in_entire_track_sequence = False
_C.train.sample_max_distance = 1
_C.train.bidirectional_supervision = False
_C.train.track_seq = True
_C.train.train_gt_heatmaps_transform = True
_C.train.train_agg = False

#### val ####
_C.val = CfgNode()
_C.val.batch_size_per_gpu = 1
_C.val.model_file = ''
_C.val.annot_dir = ''
_C.val.coco_bbox_file = ''
_C.val.use_gt_bbox = False
_C.val.flip_val = False
_C.val.bbox_thre = 1.0
_C.val.image_thre = 0.1
_C.val.in_vis_thre = 0.0
_C.val.nms_thre = 0.6
_C.val.oks_thre = 0.5
_C.val.shift_heatmap = False
_C.val.soft_nms = False
_C.val.post_process = False
_C.val.flip = False
#### test ####
_C.test = CfgNode()
_C.test.batch_size_per_gpu = 1
_C.test.model_file = ''
_C.test.annot_dir = ''
_C.test.coco_bbox_file = ''
_C.test.use_gt_bbox = False
_C.test.flip_test = False
_C.test.bbox_thre = 1.0
_C.test.image_thre = 0.1
_C.test.in_vis_thre = 0.0
_C.test.nms_thre = 0.6
_C.test.oks_thre = 0.5
_C.test.shift_heatmap = False
_C.test.soft_nms = False
_C.test.post_process = False
_C.test.flip = False
### inference ###
_C.inference = CfgNode()
_C.inference.model_file = ''
# debug
_C.debug = CfgNode()
_C.debug.vis_skeleton = False
_C.debug.vis_bbox = False
_C.debug.debug = False
_C.debug.save_batch_images_gt = False
_C.debug.save_batch_images_pred = False
_C.debug.save_heatmaps_gt = False
_C.debug.save_heatmaps_pred = False
