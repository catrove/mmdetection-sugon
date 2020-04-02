from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.runner import init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
import argparse
from mmcv import Config
#from mmdet.apis import init_dist
import torch
from collections import OrderedDict
from mmdet.core import CocoDistEvalmAPHook
from mmdet.models import RPN
from mmdet.apis.train import parse_losses, batch_processor

def myeval():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi', 'gloo'],default='gloo',help='job launcher')
    parser.add_argument('--load_from')
    parser.add_argument('--rank', type=int, default=-1, help='global rank')
    parser.add_argument('--world_size', type=int, default=-1, help='world size')
    parser.add_argument('--dist_url', type=str, default='env://', help='dist url')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.total_epochs=1
    cfg.model.pretrained = None
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        cfg.dist_params['backend']='gloo'
        cfg.dist_params['world_size'] = args.world_size
        cfg.dist_params['rank'] = args.rank
        cfg.dist_params['dist_url'] = args.dist_url
        init_dist(args.launcher, **cfg.dist_params)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)

    #data_loaders = [
    #    build_dataloader(
    #        dataset,
    #        cfg.data.imgs_per_gpu,
    #        cfg.data.workers_per_gpu,
    #        dist=True)
    #]
    model = MMDistributedDataParallel(model.cuda())
    runner = Runner(model,batch_processor,cfg.optimizer, cfg.work_dir,cfg.log_level)
    runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
    runner.load_checkpoint(args.load_from)
    runner.call_hook('before_run')
    runner.call_hook('after_train_epoch')
    runner.call_hook('after_run')

if __name__ == '__main__':
    myeval()

