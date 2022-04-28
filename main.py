# -*-coding:utf-8-*-
"""
    Author: chenhaomingbob
    E-mail: chenhaomingbob@163.com
    Time: 2022/04/16
    Description: 
"""

import os, sys

sys.path.insert(0, os.path.abspath('/'))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from engine import get_base_cfg, update_config, DefaultRunner
from tools.argument_parser import default_parse_args


def main_worker(local_rank, ngpus_per_node, cfg, args):
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            cfg.dist.rank = int(os.environ["RANK"])
            cfg.dist.world_size = int(os.environ['WORLD_SIZE'])
            cfg.dist.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            cfg.dist.local_rank = local_rank  # rank in this node (machine)
            cfg.dist.world_size = args.world_size  # the total number of gpus
            cfg.dist.rank = cfg.dist.rank * ngpus_per_node + local_rank  # rank in the world

        torch.cuda.set_device(cfg.dist.local_rank)

        print('==> distributed init (rank {}): {}, rank/gpu {}'.format(cfg.dist.rank, cfg.dist.dist_url, cfg.dist.local_rank), flush=True)
        dist.init_process_group(backend="nccl", init_method=cfg.dist.dist_url, world_size=cfg.dist.world_size, rank=cfg.dist.rank)
        dist.barrier()
    else:
        print('==> Not using distributed mode')
    my_runner = DefaultRunner(cfg, args)
    my_runner.launch()


if __name__ == '__main__':
    args = default_parse_args()
    cfg = get_base_cfg(args)
    # args > cfg
    update_config(cfg, args)
    cfg.defrost()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str, cfg.gpus)))

    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or ngpus_per_node > 1
    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, cfg, args)
