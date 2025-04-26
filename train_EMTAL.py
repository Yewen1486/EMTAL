#!/usr/bin/env python3
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
from utils import * 
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
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
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

    
    args.num_classes = 598
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
        tuning_mode=args.tuning_mode,
        Lora_alpha=args.Lora_alpha,
        mlp_ratio_=args.mlp_r,
        router_tao=args.router_tao,
        # subtle=args.subtle,
        drop_head=args.drop_head,
        attn_cls=args.attn_cls
    )
    
    file_ckpt = args.ckpt_kmeans
    if os.path.exists(file_ckpt):
        ckpt = torch.load(file_ckpt)
        model.load_state_dict(ckpt['model'], strict=False)
        print('*'*60)
        print(f'loading model {file_ckpt}')
        print('*'*60)
    else:
        ckpt = {'model': ori_model.state_dict()}
        torch.save(ckpt, file_ckpt)
    
    
    

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)


    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    root_dir = args.root_dir
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                file_n,
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ])
        output_dir = get_outdir(os.path.join(f'{root_dir}/union_FFN', args.output) if args.output else './output/train', exp_name)
        _logger.setLevel(logging.DEBUG)
        log_file = str(os.path.join(output_dir, file_n + '.log'))
        test_log = logging.FileHandler(log_file, 'a', encoding='utf-8')
        test_log.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')
        test_log.setFormatter(formatter)
        _logger.addHandler(test_log)
        
    import shutil
    import sys
    if args.local_rank == 0:
        shutil.copy(os.path.join(f'{root_dir}/', sys.argv[0]), os.path.join(output_dir, sys.argv[0]))
        shutil.copy(os.path.join(f'{root_dir}/models/{vt_name}.py', ), os.path.join(output_dir, f'{vt_name}.py'))
        _logger.info('cp {} to {}'.format(sys.argv[0], output_dir))
        _logger.info('cp {} to {}'.format(f'{vt_name}.py', output_dir))

    optimizer = create_optimizer_v2(model, _logger, **optimizer_kwargs(cfg=args))
    
    for k, v in model.named_parameters():
            _logger.info(F'{k} : {v.shape} : {v.requires_grad}, {v.mean().item()}  {v.var().item()}')
    _logger.info('*********************************load finish********************************')

    if args.local_rank == 0:
        _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
        _logger.info(f"number of params for requires grad: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        _logger.info(f"percentage : {sum(p.numel() for p in model.parameters() if p.requires_grad) / sum([m.numel() for m in model.parameters()])}")



    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')


    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)


    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    prefix_p = '/root/autodl-tmp/DATASET'
    datasets = ['cub2011', 'stanford_cars', 'aircraft', 'oxford_flowers']
    dataset_roots = [prefix_p+'/cub2011', prefix_p+'/Stanford_cars/train-test', prefix_p+'/Airoutput', prefix_p+'/102flowers']
    
    datasets_train={}
    datasets_eval={}
    datasets_val={}
    
    
    loss_task = {
        'data_0':10,
        'data_1':10,
        'data_2':10,
        'data_3':10,
    }
    
    bases = [0,200, 396, 496, 598]
    # create the train and eval datasets
    for dataset_i, root_i in zip(datasets, dataset_roots):
        print(f'dataset_i {dataset_i}')
        peft_type = 'ssf' # if 'ssf' in args.tuning_mode else ''
        if dataset_i=='oxford_flowers':
            peft_type = ''
            
        dataset_train = create_dataset(
            dataset_i, root=root_i, split=args.train_split, is_training='train'+peft_type,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats)
        
        dataset_eval = create_dataset(
            dataset_i, root=root_i, split=args.val_split, is_training='test'+peft_type,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)
        
        dataset_val = create_dataset(
            dataset_i, root=root_i, split=args.val_split, is_training='val'+peft_type,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)
        
        datasets_train[dataset_i] = dataset_train
        datasets_eval[dataset_i] = dataset_eval
        datasets_val[dataset_i] = dataset_val
        print(f'len {len(dataset_train)};   {len(dataset_eval)};   {len(dataset_val)}')
        print('-'*40)
    _logger.info(datasets_train)
    
    # load all samples from different datasets into one
    Uni_dataset = datasets_train['stanford_cars']
    for idx in tqdm(range(len(Uni_dataset))):
        path, label = Uni_dataset.parser.samples[idx]
        Uni_dataset.parser.samples[idx]=((path, label + bases[1]))
    for i, key in enumerate(datasets):
        _logger.info('='*40)
        _logger.info(key)
        _logger.info(f'datasets _ train : {datasets_train[key]}')
        if i==2 or i==3 :
            for idx in tqdm(range(len(datasets_train[key]))):
                path, label = datasets_train[key].parser.samples[idx]
                # datasets_train[key].parser.samples[idx] = (path, label + bases[i])
                Uni_dataset.parser.samples.append((path, label + bases[i]))
            _logger.info(f'Uni_dataset : {len(Uni_dataset.parser.samples)}')
        try:
            _logger.info(f'parser : {datasets_train[key].parser.samples[:40] }') 
            _logger.info(f'parser clsname  : {datasets_train[key].parser.class_to_idx  }')
        except:
            _logger.info('no parser ')
        _logger.info(f'eval {len(datasets_eval[key])}')
    
    _logger.info(f'Uni_dataset {len(Uni_dataset)}')
    _logger.info(f'Uni_dataset : {len(Uni_dataset.parser.samples)}')
    for idx in range(len(datasets_train['cub2011'])):
        sample = datasets_train['cub2011'].data.iloc[idx]
        Uni_dataset.parser.samples.append((os.path.join('/root/autodl-tmp/DATASET/cub2011/CUB_200_2011/images', sample.filepath), sample.target-1))
        
    _logger.info(f'Uni_dataset {len(Uni_dataset)}')
    _logger.info(f'Uni_dataset : {len(Uni_dataset.parser.samples)}')
    
    
    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None


    ## set no data aug
    if args.using_aug == 0:
        mixup_active = False
        args.color_jitter = 0
        args.reprob = 0


    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    
    
    loaders_eval={}
    loaders_train={} 
    _logger.info(f'args.prefetcher {args.prefetcher}')
    # create the train and eval datasets
    for dataset_i in datasets:
        loader_eval = create_loader(
            datasets_eval[dataset_i],
            input_size=data_config['input_size'],
            batch_size=32,
            is_training=False,
            use_prefetcher=1,
            direct_resize=args.direct_resize,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )
        loaders_eval[dataset_i] = loader_eval
    loader_union = create_loader(
            Uni_dataset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            simple_aug=args.simple_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_repeats=args.aug_repeats,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            worker_seeding=args.worker_seeding,
        )

    
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    proto_loss_fn = nn.CrossEntropyLoss(reduction='none').cuda()
    train_loss_fn = train_loss_fn.cuda()
    router_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    loss_reg = nn.MSELoss()
        
    
    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None

    if args.rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    
    print(f'mtl_type {args.mtl_type}')
    ema_alpha ={'iter':0}
    if args.mtl_type == '1sample':
        fb = FeatureBank_single(598,598)
    else:
        fb = FeatureBank(598, 598)
    
    
    
    start = time.time()
    try:
        cnt = 0
        single_best = [0,0,0,0,0]
        single_best_epoch = [0,0,0,0,0]
        mean_best = 0
        best_val = 0
        best_val_epoch = 0
        
        
        
        
        for epoch in range(start_epoch, num_epochs):
            start_epoch_time = time.time()
            cnt += 1
            _logger.info(file_n)
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            
                
            train_metrics, alpha = train_one_epoch_union(
                epoch, model, loader=loader_union, optimizer=optimizer, loss_fn=train_loss_fn, args=args, proto_loss_fn=proto_loss_fn,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, loss_router=router_loss_fn, fb=fb, loss_task=loss_task,
                ema_alpha=ema_alpha)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if (epoch >= 60 and epoch % 3 == 0) or (epoch < 60 and epoch % 10 == 0):
                res_epoch = ''
                tot = 0
                print(f'trained alpha : {alpha}')
                for batch_i, dataset_i in enumerate(datasets):
                    eval_metrics = validate(bases[batch_i], model, loaders_eval[dataset_i], validate_loss_fn, args, amp_autocast=amp_autocast, domain_gt=batch_i, last_alpha=alpha)
                    res_epoch += dataset_i + ':' + str(str(round(eval_metrics['top1'],3))) + '; '
                    tot += eval_metrics['top1']
                    
                    if eval_metrics['top1'] > single_best[batch_i]:
                        single_best[batch_i] = eval_metrics['top1']
                        single_best_epoch[batch_i] = epoch
                    
                tot /= len(datasets)
                if tot>mean_best:
                    mean_best = tot
                    best_val = tot
                    best_val_epoch = epoch

                # print(eval_metrics)
                res_epoch += 'mean:' + str(tot) + ' '
                res = os.path.join(output_dir, res_epoch +  ' from ' + str(epoch))
                file_res = open(res, 'w')
                file_res.write(str(res))
                file_res.close()

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None: 
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = best_val
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                # if best_epoch == epoch:
                #     torch.save(center, os.path.join(output_dir, 'best_center.gp'))
            
            
            end_epoch_time = time.time()
            _logger.info(f'{args.dataset} epoch time: {(end_epoch_time - start_epoch_time)/60} min  expected time: {(end_epoch_time - start)/60/(epoch+1)*(num_epochs-epoch-1)} min')

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_val, best_val_epoch))
        _logger.info('*** Best metric with ema: {0} (epoch {1})'.format(best_metric, best_epoch))
        res_final = ''
        for batch_i, dataset_i in enumerate(datasets):
            res_final += dataset_i + ':' + str(round(single_best[batch_i],3)) + ' from '+ str(single_best_epoch[batch_i]) +';  '
        _logger.info(f'*** {res_final}')
            
        res = os.path.join(output_dir, res_final)
        file_res = open(res, 'w')
        file_res.write(str(res))
        file_res.close()

    end = time.time()
    
    _logger.info("output_dir: ", output_dir)
    if args.debug:
        _logger.info("debuggin removing")
        shutil.rmtree(output_dir)  # , ignore_errors=True)
    else:
        _logger.info("removing ckpt")
        for file_name in os.listdir(output_dir):
            if "checkpoint-" in file_name or 'last.pth.tar' in file_name:
                os.remove(os.path.join(output_dir, file_name))
    _logger.info(end - start)
    if cnt == num_epochs or end - start >= 3600:
        write_csv(args, best_metric, best_epoch)
            

def train_one_epoch_union(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        proto_loss_fn='',
        loss_scaler=None, model_ema=None, mixup_fn=None,loss_router=None,fb=None,loss_task=None,ema_alpha=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m_domain = AverageMeter()
    
        # 0-199-> d1   200->395->d2  396->495->d3  496->597->d4
    # set to int
    cls_id2dataset = torch.zeros(598, dtype=torch.int).cuda()
    cls_id2dataset[0:200] += 0
    cls_id2dataset[200:396] += 1
    cls_id2dataset[396:496] += 2
    cls_id2dataset[496:598] += 3
    # set to int
    cls_id2dataset = cls_id2dataset.long()
    

    model.train()

    st = time.time()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    loss_dis = 0
    # print('     one iter start')
    ite = ema_alpha['iter']
    
    
    tot_iter = len(loader) * 110
    alpha = 0
    for batch_idx, (input, target) in enumerate(loader):
        alpha = min(1/tot_iter*2 * ite, 1)
        ite+=1
        start_time_batch = time.time()
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        # print(f'loader cost time: {time.time()-start_time_batch}')
        start_time_batch = time.time()
        idx = cls_id2dataset[target]
        with amp_autocast():
            output, fea, part_fea, router_layer3 = model(input, alpha=alpha)
            
            loss_ce = loss_fn(output, target)
        # print(f'idx {idx}')
        loss_weight = proto_loss_fn(output, target)
        # print(f'loss_weight {loss_weight}')
        loss_t1 = []
        for i in range(4):
            mask = idx==i
            if mask.sum()!=0:
                loss_task1 = mask * loss_weight.detach()
                loss_task[f'data_{i}'] = loss_task[f'data_{i}'] * 0.95 + loss_task1.sum()/mask.sum() * 0.05
            loss_t1.append(loss_task[f'data_{i}'])
        # exit()
        weight = torch.tensor([loss_t1[id] for id in idx]).cuda()
        loss_r = 0 
        if args.mtl_type == '1sample':
            kl = ( (kl_divergence_loss(output, fb.bank[target,0], args.aug_epoch) * 2)).sum(dim=-1)
        elif args.mtl_type == '2sample':
            kl = ( (kl_divergence_loss(output, fb.bank[target,0], args.aug_epoch) + kl_divergence_loss(output, fb.bank[target,1], args.aug_epoch))).sum(dim=-1)
        else:
            raise ValueError('mtl_type should be 1sample or 2sample')
        loss_dis = 1 * ( kl / weight).mean()
        loss = loss_r + loss_ce + loss_dis
        fb.update_bank(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            
        start_time_batch = time.time()
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        
        # start from different epoch: 10, 20, 40
        
        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0: 
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                # print(target)
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Loss_ce: {loss_ce:#.4g}   '
                    # 'Loss_r: {loss_r:#.4g} '
                    'loss_dis: {loss_dis:#.4g} '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'
                    # 'Domain@1: {top1_domain.avg:>7.4f}'
                    .format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        loss_ce=loss_ce,
                        # loss_r=loss_r,
                        loss_dis=loss_dis,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                        # top1_domain=top1_m_domain
                        ))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        end = time.time()
        
        start_time_batch = time.time()
        # end for
    _logger.info(f'epoch cost training time{time.time() - st} tot {(time.time() - st)*110 / 3600}')

    ema_alpha['iter'] = ite
    
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)]), alpha


def kl_divergence_loss(output, output_aug, temperature=2.0):
    log_prob = F.log_softmax(output / temperature, dim=-1)
    prob_aug = F.softmax(output_aug / temperature, dim=-1)
    
    loss = F.kl_div(log_prob, prob_aug, reduction='none') * (temperature ** 2)
    l_batch = F.kl_div(log_prob, prob_aug, reduction='batchmean') * (temperature ** 2)
    return loss

class FeatureBank():
    def __init__(self, num_classes, feature_dim, samples_per_class=2):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.samples_per_class = samples_per_class
        self.bank = torch.zeros((num_classes, samples_per_class, feature_dim)).half().cuda()
        
    def update_bank(self, features, target):
        
        self.bank[target,0] = self.bank[target,1]
        self.bank[target,1] = features.detach()


class FeatureBank_single():
    def __init__(self, num_classes, feature_dim, samples_per_class=1):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.samples_per_class = samples_per_class
        self.bank = torch.zeros((num_classes, samples_per_class, feature_dim)).half().cuda()
        
    def update_bank(self, features, target):
        self.bank[target,0] = self.bank[target,0]*0.5 + features.detach() * 0.5
        
def validate(shift, model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', center='', domain_gt=0, last_alpha=1):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    top1_m_domain = AverageMeter()
    
    # 0-199-> d1   200->395->d2  396->495->d3  496->597->d4
    # set to int
    cls_id2dataset = torch.zeros(598, dtype=torch.int).cuda()
    cls_id2dataset[0:200] += 0
    cls_id2dataset[200:396] += 1
    cls_id2dataset[396:496] += 2
    cls_id2dataset[496:598] += 3
    # set to int
    cls_id2dataset = cls_id2dataset.long()
    

    model.eval()
    model.dual_input = 0

    end = time.time()
    last_idx = len(loader) - 1
    print(f'    into val : last_alpha {last_alpha}')
    with torch.no_grad():
        
        for batch_idx, (input, target) in enumerate(loader):
            domain_gt_list = torch.tensor([domain_gt] * input.shape[0]).cuda()
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda() + shift
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output, _, _, domain = model(input, alpha=last_alpha)

            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            domain = None
            if domain is not None:
                
                acc1_domain, acc2 = accuracy(domain, domain_gt_list, topk=(1,2))
                top1_m_domain.update(acc1_domain.item(), output.size(0))
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    # 'Domain@1: {top1_domain.avg:>7.4f}'
                    .format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m, 
                        # top1_domain=0
                        ))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    
    return metrics


if __name__ == '__main__':
    main()
 