import argparse
import time
import yaml
import os
import logging
import numpy as np
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import csv

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

from data import create_loader, create_dataset
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


setup_default_logging()
args, args_text = _parse_args()
# import models
vt_name = args.vt_name #'vision_transformer_ori_type_full_part_hidden'
from models  import swin_transformer, convnext, as_mlp
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



import torch
from contextlib import suppress
def gain_center(model, loader, args, center='', proto='', amp_autocast=suppress, log_suffix=''):
    '''
        features = features.reshape(features.shape[0], -1)
        target_centers = centers[target]
        target_centers = torch.nn.functional.normalize(target_centers, dim=-1)
        center_offset = (1-alpha)*(features.detach() - target_centers)
        distance = torch.pow(features - target_centers, 2)
        distance = torch.sum(distance, dim=-1)
        center_loss = torch.mean(distance)
    '''
    model.eval()
    model.dual_input = 0

    end = time.time()
    last_idx = len(loader) - 1

    sample_counter = torch.zeros(args.num_classes).cuda()
    from tqdm import tqdm
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with amp_autocast():
                output, feature, part_fea = model(input)

            for sample in range(input.shape[0]):
                sample_counter[target[sample]] += 1
                center[target[sample]] += feature[sample]
                proto[target[sample]] += part_fea[sample]
            
            
            if isinstance(output, (tuple, list)):
                output = output[0]


            torch.cuda.synchronize()
            end = time.time()
    print(center.shape)
    center = center / sample_counter.expand(768,sample_counter.shape[0]).permute(1, 0)
    proto = proto / sample_counter.expand(768,sample_counter.shape[0]).permute(1, 0)
    print(center.shape)
    print(sample_counter)
    return center, proto


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        # center='', part_center='', 
        proto_loss_fn='',
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, loss_reg=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    model.dual_input = args.dual_input

    if args.rank == 0:
        requires_grad_params_counter = 0
        for k, v in model.named_parameters():
            requires_grad_params_counter = requires_grad_params_counter + (1 if  v.requires_grad else 0)
        now_cnt = 0
        for k, v in model.named_parameters():
            now_cnt = now_cnt + (1 if v.requires_grad else 0)
            if now_cnt >= requires_grad_params_counter - 7 and v.requires_grad:
                print('{:.7f}  {:.7f}'.format(v.mean().item(), v.var().item()), " ", " ; ", k, ' :', v.shape, ": ", v.requires_grad)
        print('*'*20, 'start training one epoch', '*'*20)


    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    print("start one epoch")
    print(f'tot batch {last_idx + 1}')
    beta = args.beta
    b1 = {
        'cub2011':0.5,
        'stanford_cars':0.5,
        'stanford_dogs':0.1,
        'nabirds':0.5,
        'oxford_flowers':0.1,
    }
    b2 = {
        'cub2011':0.05,
        'stanford_cars':0.1,
        'stanford_dogs':0.01,
        'nabirds': '???',
    }
    print(f'beta {beta}')
    print(f'args center {args.center}')
    chosen_batch = 0
    for batch_idx, (input, target) in enumerate(loader):
        # print(input.shape)
        # print(target.shape)
        # exit()
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)


        constrain = 0
        with amp_autocast():
            debug = last_batch
            output, fea, part_fea = model(input, debug, target)
            
            
            proto_loss = 0 
            part_loss = 0
            # print(args.drop_head)
            if args.center>0 or args.drop_head>0:
                sim, center_diff = Proto_Loss_dogs(fea, model.centers, target, alpha=0.95)

                if last_batch:
                    torch.set_printoptions(linewidth=args.num_classes)
                    _logger.info('sim')
                    prob_fea = torch.gather(sim, 1, target.unsqueeze(dim=-1)).squeeze(dim=-1)
                    _logger.info(prob_fea)
                    _logger.info('diff')
                    _logger.info(sim.max(dim=-1)[0] -  prob_fea)

                sim *= args.tao
                proto_loss = proto_loss_fn(sim, target)
                
                model.update_center(center_diff.detach(), target)

            ## oght on cls_token
            loss_reg_cls = 0
            if args.reg_type==2:
                oght_fea = fea.reshape(target.shape[0], 12, 64)
                val = torch.bmm(oght_fea, oght_fea.permute(0, 2, 1))
                loss_reg_cls = loss_reg(val, torch.eye(12).cuda().expand(target.shape[0],-1,-1))
            if args.reg_type==1:
                oght_fea = part_fea.reshape(target.shape[0], 12, 64)
                val = torch.bmm(oght_fea, oght_fea.permute(0, 2, 1))
                loss_reg_cls = loss_reg(val, torch.eye(12).cuda().expand(target.shape[0],-1,-1))
            loss1 = loss_fn(output, target)
            ###
            if args.dataset not in b1.keys():
                b1[args.dataset] = 0.5
            loss = (loss1 + b1[args.dataset] * (proto_loss).mean()) if args.center>0 else loss1
            if args.reg_type:
                loss = loss + beta * loss_reg_cls

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

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
   
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Proto: ({loss_dist:#.3g})  '
                    'Orth: ({loss_reg:#.3g})  '
                    'CE : ({loss_dist2:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'drop {drop:.2e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        loss_dist= b1[args.dataset] * (proto_loss).mean() if args.center>0 else (proto_loss).mean(),#b1[args.dataset]
                        loss_reg=beta * loss_reg_cls,### 有问题需要修改的
                        loss_dist2=loss1,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        drop=args.drop_head,
                        data_time=data_time_m))

        if args.save_images and output_dir:
            torchvision.utils.save_image(
                input,
                os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                padding=0,
                normalize=True)
            print(f'path : {os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx)}')
            if batch_idx == chosen_batch:
                exit()

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        end = time.time()
        # end for

    print("end one epoch")
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])# , center, part_center


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', center=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    model.dual_input = 0

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

            if isinstance(output, (tuple, list)):
                # fea = output[1]
                output = output[0]
                # print(f'output shape: {output.shape}')
            # output = Proto_cls(fea, center)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

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
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    
    return metrics