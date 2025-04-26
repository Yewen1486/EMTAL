""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from k_means_constrained import KMeansConstrained
import sklearn
from sklearn import preprocessing
import argparse
import time
import yaml
import os
import logging
import numpy as np
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from random import shuffle
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *

from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from data import create_loader, create_dataset, create_dataset_union
from utils import * #get_file_n, Center_Loss, Proto_Loss, Proto_cls, Part_Proto_Loss
from optim_factory import create_optimizer_v2, optimizer_kwargs

import ipdb

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# finetuning
parser = adding(parser)
# making it more tight
const_seed_as_vpt = 42
# sys.stdout.reconfigure(line_buffering=True, encoding='utf-8', errors='replace', width=200)

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    setup_default_logging()
    args, args_text = _parse_args()
    # import models
    vt_name = args.vt_name #'vision_transformer_ori_type_full_part_hidden'
    # from models  import swin_transformer, convnext, as_mlp
    import importlib
    vt = importlib.import_module('models.' + vt_name)
    
    args.pretrain_type = args.pretrain_type.upper()
    args.subtle = args.subtle.upper()
    
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
             
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cpu'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    # fix result
    random_seed(const_seed_as_vpt, args.rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.fuser:
        set_jit_fuser(args.fuser)

    file_n = get_file_n(args)

    
    args.num_classes = 598 #+ 555
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        tuning_mode=args.tuning_mode
    )
    ori_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        tuning_mode=args.tuning_mode
    )

    
    if 1:
        #######################################################################
        # act_r.pth comes from saving ckpt using timm
        # examples:
        #       1.Run train_EMTAL.py using std Vit-B/16
        #       2.Save the loaded model to ckpt.tar.pth, before the training
        #       3.Stop the training of train_EMTAL.py
        #       4.Cluster ckpt.tar.pth into 16e_weight_basic_wo-grad.pth
        #######################################################################
        
        actr = torch.load('/home/PEFT4FGVC_3090/MTL/ERUDITE_V/union_FFN/ckpt.tar.pth')['val']
        
        model = model.cpu()
        for n_clusters in [16]:
            rand_c = torch.zeros(12, 16)
            cluster_c = torch.zeros(12, 16)
            
            rank_rand = torch.zeros(12)
            rank_cluster = torch.zeros(12)
            
            
            block_num = -1
        
            print( f' args.ckpt_output {args.ckpt_output}') 
            print('='*50)
            
            cluster_flag = 1
            cluster_idx = {}
            result111 = {'before':[], 'after':[]}
            print(f' cluster_idx {cluster_idx}')
            
            layer_id = 0 
            for name, param in model.named_parameters():
                print(f'name {name} shape {param.shape} {param.data.mean()}')
                # print(f'ori {name} shape {ori_model.state_dict()[name].shape} {ori_model.state_dict()[name].mean()}')
                # continue
                if 'lora' in name:
                    print("     skip")
                    continue
                
                if 'fc1.weight' in name:
                    block_num += 1
                    # weight = param.data.cpu()
                    weight = actr[block_num]#.cpu()
                    ffn_weight_norm = sklearn.preprocessing.normalize(weight)
                    
                    m1 = 768*4//n_clusters #16#
                    m2 = 768*4//n_clusters #128#
                    
                    print(f'-----------------------------------------starting---------------------------------------')
                    if cluster_flag:
                        kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=m1, size_max=m2, random_state=0).fit(ffn_weight_norm, None)
                        labels = [x for x in kmeans.labels_]
                        cluster_idx[f'block{block_num}'] = labels
                    else:
                        labels = cluster_idx[f'block{block_num}']
                        
                    idx = torch.tensor(labels).argsort()#.cuda()
                    
                    
                    
                    # cal nodes num for each cluster    
                    # print(f'labels: {labels}')
                    count = [0 for _ in range(n_clusters)]
                    for i in labels:
                        count[i] += 1
                    print(f'count {count}')
                    # print(f'labels: {labels}')
                    # list  argsort
                    weight = torch.index_select(param.data, 0, idx).reshape(768*4,768)
                    # weight = weight.mean(dim=1)
                    
                    fc1_before = param.data
                    fc1_after = weight
                    rank_before = 0
                    rank_after = 0
                    st = 0
                    ed = 0
                    tmp1, tmp2=0, 0
                    for expert in range(n_clusters):
                        ed += count[expert]
                        rank_before = torch.svd(fc1_before[st:ed]).S
                        rank_after = torch.svd(fc1_after[st:ed]).S
                        st += count[expert]
                        
                        
                        rank_after = rank_after / torch.sum(rank_after)
                        rank_before = rank_before / torch.sum(rank_before)
                        tmp1 += rank_before[:3].sum()
                        tmp2 += rank_after[:3].sum()
                        # rand_c[block_num] += rank_before[:3]
                        # cluster_c[block_num] += rank_after[:3]
                        
                    print(f'        before {tmp1}')
                    print(f'        after {tmp2}')
                        
                    
                    
                    
                elif 'fc1.bias' in name:
                    weight = torch.index_select(param.data, 0, idx).reshape(768*4)
                    # weight = weight.mean(dim=1)
                elif 'fc2.weight' in name:
                    # weight = torch.index_select(param.data.T, 0, idx).reshape(768*2,2,768).sum(dim=1)
                    weight = torch.index_select(param.data.T, 0, idx).reshape(768*4,768)#.sum(dim=1)
                    weight = weight.T
                else:
                    weight = param.data
                ori_model.state_dict()[name].copy_(weight)

            print('*'*50)
            
            print('*'*50)
            
            print(f'result {result111}')
            print('-'*50)
                
            file_ckpt = args.ckpt_output
            ckpt = {'model': ori_model.state_dict()}
            torch.save(ckpt, file_ckpt)
        exit()
    
    

if __name__ == '__main__':
    main()
 