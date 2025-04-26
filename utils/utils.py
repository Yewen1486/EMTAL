# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

# Modified by: Daquan Zhou

'''
- resize_pos_embed: resize position embedding
- load_for_transfer_learning: load pretrained paramters to model in transfer learning
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
'''

import os
import sys
import time
import torch
import math

import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

def resize_pos_embed(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)   # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)   # [1, 24*24+1, dim]
    return posemb

def resize_pos_embed_cait(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    posemb_grid = posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)   # [1, dim, 24, 24] -> [1, 24*24, dim]
    return posemb_grid


def resize_pos_embed_nocls(posemb, posemb_new): # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    gs_old = posemb.shape[1]     # 14
    gs_new = posemb_new.shape[1]             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb
    posemb_grid = posemb_grid.permute(0, 3, 1, 2)  # [1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1)   # [1, dim, 24, 24]->[1, 24, 24, dim]
    return posemb_grid


def load_state_dict(checkpoint_path,model, use_ema=False, num_classes=1000, no_pos_embed=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            old_aux_head_weight = state_dict.pop('aux_head.weight', None)
            old_aux_head_bias = state_dict.pop('aux_head.bias', None)
        if not no_pos_embed:
            old_posemb = state_dict['pos_embed']
            if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
                if len(old_posemb.shape)==3:
                    if int(math.sqrt(old_posemb.shape[1]))**2==old_posemb.shape[1]:
                        new_posemb = resize_pos_embed_cait(old_posemb, model.pos_embed)
                    else:
                        new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
                elif len(old_posemb.shape)==4:
                    new_posemb = resize_pos_embed_nocls(old_posemb, model.pos_embed)
                state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_for_transfer_learning(model, checkpoint_path, use_ema=False, strict=True, num_classes=1000):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)

def load_for_probing(model, checkpoint_path, use_ema=False, strict=False, num_classes=19167):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes=19167, no_pos_embed=True)
    info=model.load_state_dict(state_dict, strict=strict)
    print(info)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothing_CrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]    
        log_preds = F.log_softmax(preds, dim=-1)    
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)      
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)   
        return linear_combination(loss / n, nll, self.epsilon)


var_list = [
        # 'pretrained',
        # 'pretrain_type',
        'tuning_mode',
        'vpt',
        'opt',
        # 'img_size',
        # 'using_aug',
        # 'tune_cls',
        # 'cplx_head',
        # 'gp',
        # 'seb',
        # 'subtle',
        # 'b1',
        'beta',
        'beta_orth',
        # 'tao',
        # 'center',
        'seed',
        'drop_head',
        # 'reg_type',
        'drop',
        'length',
        'drop_out',
        'seed',
        # 'weighted_loss',
        'epochs',
        'Lora_alpha',
        'vt_name',
        'aug_epoch',
        'router_tao',
        'shots',
        'kl_weight'
        # 'model_ema_decay',
    ]
def get_file_n(args):
    '''

    :return: the total file name
    '''
    file_n = ""
    outname = 0
    for var in var_list:
        try:
            file_n += var+"-"+str(getattr(args, var))+" " if outname else str(getattr(args, var)) + " "
            if var =='opt':
                file_n += str(args.lr/args.world_size/ args.batch_size * 32)+" "
                outname = 1
        except:
            print("No such attribute: ", var)
            pass
    return file_n


import csv
def write_csv(args, best_metric, best_epoch):
    with open('exp_record.csv', 'a') as f:
        writer = csv.writer(f)
        row = [str(getattr(args, var)) for var in var_list]
        row+= [args.dataset, args.lr / args.world_size, best_metric, best_epoch]
        writer.writerow(row)
    print("write finish")


def adding(parser):
    
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='whether using trained ckpt')

    parser.add_argument('--resume-ssf', default=None, type=str,
                        help='resume_ssf')

    parser.add_argument('--resume-ours', default=None, type=str,
                        help='resume_ours')
    
    parser.add_argument('--using_aug', default=0, type=int,
                        help='whether using augmentation')

    parser.add_argument('--pretrain_type', default='cls', type=str,
                        help='CLS, MAE or SAM')

    parser.add_argument('--debug', default=0, type=int,
                        help='whether using debugging')

    parser.add_argument('--tune_cls', default=0, type=int,
                        help='whether using tuning cls_token')

    parser.add_argument('--cplx_head', default=0, type=int,
                        help='whether using cplx_head')

    parser.add_argument('--vpt', default='None', type=str,
                        help='vpt deep')

    parser.add_argument('--seb', default=0, type=int,
                        help='whether using seb')

    parser.add_argument('--mini', default=0, type=int,
                        help='whether using mini dataset')

    parser.add_argument('--loss', default='CE', type=str,
                        help='CE RCE')

    parser.add_argument('--subtle', default='None', type=str,
                        help='NONE subtle weighted_subtle')

    parser.add_argument('--dual_input', default=0, type=int,
                        help='dual_input')

    parser.add_argument('--beta-constrain', default=0, type=float,
                        help='constrain reg loss')
    
    parser.add_argument('--b1', default=0, type=float,
                        help='protoloss')
    
    parser.add_argument('--beta', default=0, type=float,
                        help='weight reg loss')
    
    parser.add_argument('--beta_orth', default=0, type=float,
                        help='weight reg loss')

    parser.add_argument('--center', default=0, type=float,
                        help='using prototype?')

    parser.add_argument('--tao', default=10.0, type=float,
                        help='tao?')
    
    parser.add_argument('--part-hidden', default=10, type=int,
                        help='tao?')
    
    parser.add_argument('--ogth', default=0, type=float,
                        help='reg')
    
    parser.add_argument('--vt-name', default='vision_transformer', type=str,
                        help='doing which exp')
    
    parser.add_argument('--ckpt_output', default='ckpt_disolve.pth', type=str,
                        help='doing which exp')
    
    parser.add_argument('--find-hard-simple', action='store_true', default=False,
                        help='wether find hard simple')
    
    parser.add_argument('--reg-type', default=0, type=int,
                        help='00 01 10 11')
    
    parser.add_argument('--drop-head', default=0, type=float,
                        help='head drop out')
    
    parser.add_argument('--drop-out', default=0, type=float,
                        help='head drop out')
    
    parser.add_argument('--subtle-num', default=0, type=int,
                        help='num of subtle tokens')
    
    parser.add_argument('--Lora_alpha', default=2, type=int,
                        help='Lora_alpha')
    
    parser.add_argument('--length', default=1, type=int,
                        help='CIEs')
    
    parser.add_argument('--aug_epoch', default=0, type=int,
                        help='CIEs')
    
    parser.add_argument('--mlp_r', default=4, type=float,
                        help='FFN mlp_r')
    
    parser.add_argument('--kl-weight', default=1, type=float,
                        help='kl-weight')
    
    
    parser.add_argument('--shots', default=0, type=int,
                        help='shots')
    
    
    parser.add_argument('--mtl_type', default='2sample', type=str,
                        help='keep how many samples in (1sample, 2sample, 3sample)')
    
    parser.add_argument('--router_tao', default=5, type=float,
                        help='router_tao moe')
    
    parser.add_argument('--attn_cls', default=0, type=int,
                        help='attn_cls')
    
    parser.add_argument('--lora_rank', default=16, type=int,
                        help='lora_rank')
    
    parser.add_argument('--ckpt_kmeans', default='ckpt_ratio2_sum.pth', type=str,
                        help='which ckpt_kmeans')
    
    
    parser.add_argument('--init-peft', default='/data/zhonghanwen_2022/23wb/Erudite_training_logs/output_beta/ver_beta0.0.base/stanford_cars/CLS ssf None adamw 0.01 subtle-NONE beta-0.0 dual_input-0 tao-10.0 part_hidden-10 center-0.0 seed-42 ogth-0.0 drop_head-0 beta_constrain-0 subtle_num-0 -20230926-164337/model_best.pth.tar', type=str,
                        help='path of ckpt to init PEFT')
    
    parser.add_argument('--act-subtle', default='RELU', type=str,
                        help='using act layer for subtle')
    # prompt_tokens
    parser.add_argument('--prompt-tokens', default='200', type=int,help='prompt tokens')
    
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='test',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='Validation batch size override (default: None)')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='torch.jit.script the full model')
    parser.add_argument('--fuser', default='', type=str,
                        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing through model blocks/stages')


    # finetuning
    parser.add_argument('--tuning-mode', default=None, type=str,
                        help='Method of fine-tuning (default: None')


    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate')


    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                        help='list of decay epoch indices for multistep lr. must be increasing')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--simple-aug', action='store_true', default=False,
                        help='Only randomresize and flip training augmentation, override other train aug args')
    parser.add_argument('--direct-resize', action='store_true', default=False,
                        help='Direct resize image in validation')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--weighted-loss', action='store_true', default=False,
                        help='During union train, whether to weight loss from different datasets')
    
    parser.add_argument('--root_dir', default='/root/autodl-tmp/EMTAL', type=str, metavar='PATH',
                        help='path to root dir')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')

    return parser




def Center_Loss(features, centers, target, alpha=0.95):
    features = features.reshape(features.shape[0], -1)
    target_centers = centers[target]
    target_centers = torch.nn.functional.normalize(target_centers, dim=-1)
    center_offset = (1-alpha)*(features.detach() - target_centers)
    distance = torch.pow(features - target_centers, 2)
    distance = torch.sum(distance, dim=-1)
    center_loss = torch.mean(distance)

    return center_loss, center_offset


# 计算当前feature和k个类中心Ci的ProtoType loss，拉进他到对应中心的距离 推远和其他中心的距离
# 基于余弦相似度和Softmax损失计算Prototype loss
def Proto_Loss(features, centers, target, alpha=0.95):
    ## 注意这里你Reshape了啊
    features = features.reshape(features.shape[0], -1)
    centers = torch.nn.functional.normalize(centers, dim=-1)
    # 计算类中心更新量
    target_centers = centers[target]
    target_centers = torch.nn.functional.normalize(target_centers, dim=-1)
    center_offset = (1-alpha)*(features.detach() - target_centers)
    sim = torch.cosine_similarity(features.unsqueeze(1), centers.unsqueeze(0), dim=-1)
    return sim, center_offset

def Proto_Loss_dogs(features, centers, target, alpha=0.95):
    ## 注意这里你Reshape了啊
    # B = features.shape[0]
    # features = features.reshape(B, -1)
    # 计算类中心更新量
    centers1 = centers.clone()
    target_centers = centers1[target]
    center_offset = (1-alpha)*(features.detach() - target_centers)
    sim = torch.cosine_similarity(features.unsqueeze(1), centers1.unsqueeze(0), dim=-1)
    return sim, center_offset

def Part_Proto_Loss(features, centers, target, alpha=0.95):
    '''
        features: [32*12, 64]
        ceter: [K*12, 64]
        target: [32*12]
    '''
    ## 注意这里你Reshape了啊
    centers = torch.nn.functional.normalize(centers, dim=-1)
    target_centers = centers[target]
    target_centers = torch.nn.functional.normalize(target_centers, dim=-1)
    center_offset = (1-alpha)*(features.detach() - target_centers)
    sim = torch.cosine_similarity(features.reshape(-1, 12, 64).permute(1, 0, 2).unsqueeze(2), centers.reshape(-1, 12, 64).permute(1, 0, 2).unsqueeze(1), dim=-1)
    return sim, center_offset


def Part_Proto_Loss_dogs(features, centers, target, alpha=0.95):
    '''
        features: [32*12, 64]
        ceter: [K*12, 64]
        target: [32*12]
    '''
    ## 注意这里你Reshape了啊
    # print(type(centers))
    # print('*' * 20)
    # exit()
    centers1 = centers.clone()
    target_centers = centers1[target]
    center_offset = (1-alpha)*(features.detach() - target_centers)
    sim = torch.cosine_similarity(features.reshape(-1, 12, 64).permute(1, 0, 2).unsqueeze(2), centers1.reshape(-1, 12, 64).permute(1, 0, 2).unsqueeze(1), dim=-1)
    return sim, center_offset


def Proto_cls(features, centers):
    centers = torch.nn.functional.normalize(centers, dim=-1)
    sim = torch.cosine_similarity(features.unsqueeze(1), centers.unsqueeze(0), dim=-1)
    return sim

def params(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("parameter:%fM" % (total/1e6))
    
    tensor = (torch.rand(1, 3, 224, 224).cuda(),)
    from thop import profile
    flops, params = profile(model, inputs=tensor)
    print("FLOPs=", str(flops/1e9) + '{}'.format("G"))
    print("params=", str(params/1e6) + '{}'.format("M"))
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # 分析FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total()/1e9)
    # 分析parameters
    # print(parameter_count_table(model))
    exit()
    

