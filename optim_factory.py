""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2021 Ross Wightman
"""
import json
from itertools import islice
from typing import Optional, Callable, Tuple, Dict, Union


import torch
import torch.nn as nn
import torch.optim as optim

#from timm.models.helpers import group_parameters

from timm.optim.adabelief import AdaBelief
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lamb import Lamb
from timm.optim.lars import Lars 
from timm.optim.lookahead import Lookahead
from timm.optim.madgrad import MADGRAD
from timm.optim.nadam import Nadam 
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP



try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False



def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def _group(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def _layer_map(model, layers_per_group=12, num_groups=None):
    def _in_head(n, hp):
        if not hp:
            return True
        elif isinstance(hp, (tuple, list)):
            return any([n.startswith(hpi) for hpi in hp])
        else:
            return n.startswith(hp)

    head_prefix = getattr(model, 'pretrained_cfg', {}).get('classifier', None)
    names_trunk = []
    names_head = []
    for n, _ in model.named_parameters():
        names_head.append(n) if _in_head(n, head_prefix) else names_trunk.append(n)

    # group non-head layers
    num_trunk_layers = len(names_trunk)
    if num_groups is not None:
        layers_per_group = -(num_trunk_layers // -num_groups)
    names_trunk = list(_group(names_trunk, layers_per_group))

    num_trunk_groups = len(names_trunk)
    layer_map = {n: i for i, l in enumerate(names_trunk) for n in l}
    layer_map.update({n: num_trunk_groups for n in names_head})
    return layer_map



def group_with_matcher(
        named_objects,
        group_matcher: Union[Dict, Callable],
        output_values: bool = False,
        reverse: bool = False
):
    if isinstance(group_matcher, dict):
        # dictionary matcher contains a dict of raw-string regex expr that must be compiled
        compiled = []
        for group_ordinal, (group_name, mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            # map all matching specifications into 3-tuple (compiled re, prefix, suffix)
            if isinstance(mspec, (tuple, list)):
                # multi-entry match specifications require each sub-spec to be a 2-tuple (re, suffix)
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal,), sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal,), None)]
        group_matcher = compiled

    def _get_grouping(name):
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = (prefix, r.groups(), suffix)
                    # map all tuple elem to int for numeric sort, filter out None entries
                    return tuple(map(float, chain.from_iterable(filter(None, parts))))
            return float('inf'),  # un-matched layers (neck, head) mapped to largest ordinal
        else:
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return ord,
            return tuple(ord)

    # map layers into groups via ordinals (ints or tuples of ints) from matcher
    grouping = defaultdict(list)
    for k, v in named_objects:
        grouping[_get_grouping(k)].append(v if output_values else k)

    # remap to integers
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])

    if reverse:
        assert not output_values, "reverse mapping only sensible for name output"
        # output reverse mapping
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id

    return layer_id_to_param


def group_parameters(
        module: nn.Module,
        group_matcher,
        output_values=False,
        reverse=False,
):
    return group_with_matcher(
        module.named_parameters(), group_matcher, output_values=output_values, reverse=reverse)


def group_modules(
        module: nn.Module,
        group_matcher,
        output_values=False,
        reverse=False,
):
    return group_with_matcher(
        named_modules_with_params(module), group_matcher, output_values=output_values, reverse=reverse)




def param_groups_layer_decay(
        model: nn.Module,
        weight_decay: float = 0.05,
        no_weight_decay_list: Tuple[str] = (),
        layer_decay: float = .75,
        end_layer_decay: Optional[float] = None,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, 'group_matcher'):
        # FIXME interface needs more work
        layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    else:
        # fallback
        layer_map = _layer_map(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    # FIXME temporary output to debug new feature
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
        tuning_mode=cfg.tuning_mode)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'layer_decay', None) is not None:
        kwargs['layer_decay'] = cfg.layer_decay
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model_or_params,
        _logger,
        opt: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        tuning_mode: str = None,
        filter_bias_and_bn: bool = True,
        layer_decay: Optional[float] = None,
        param_group_fn: Optional[Callable] = None,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # TODO: for fine-tuning 
        print(f'tuning_mode {tuning_mode}')
        if tuning_mode:
            for name, param in model_or_params.named_parameters():
                # print(name)
                if tuning_mode == 'linear_probe':
                    if "head" not in name:
                        param.requires_grad = False
                elif 'cluster' in tuning_mode:
                    if 'fc1.weight' not in name:
                        param.requires_grad = False
                elif 'WEMOE' in tuning_mode:
                    if 'gate_weight' not in name:
                        param.requires_grad = False
                
                elif 'LoRA' in tuning_mode:
                    if "head" not in name and "lora" not in name: # and 'router' not in name:
                        param.requires_grad = False
                elif 'ssf_norm' in tuning_mode:
                    if "head" not in name and "ssf" not in name and 'norm' not in name  and 'bias' not in name :
                        param.requires_grad = False
                elif 'ssf_relu' in tuning_mode:
                    if "head" not in name and "ssf" not in name and 'fc1' not in name  and 'fc2' not in name :
                        param.requires_grad = False
                elif 'full' in tuning_mode:
                    pass
                    # if "head" not in name and "bias" not in name:
                    #     param.requires_grad = False
                elif 'ARC' in tuning_mode:
                    if "head" not in name and "adapter" not in name and "down_projection" not in name:
                        param.requires_grad = False
                elif 'adapter' in tuning_mode:
                    if "head" not in name and "adapter" not in name:
                        param.requires_grad = False
                elif 'bias' in tuning_mode:
                    if "head" not in name and "bias" not in name:
                        param.requires_grad = False
                elif 'ssf' == tuning_mode:
                    if "head" not in name and "ssf" not in name \
                            and "subtle" not in name and "prompt" not in name and "reweight" not in name and "dataset_router" not in name:
                        param.requires_grad = False
                elif 'norm' in tuning_mode:
                    if "head" not in name and "norm" not in name and "bias" not in name and 'reweight_' not in name:
                        param.requires_grad = False
                elif 'vpt' in tuning_mode:
                    if "head" not in name and "prompt" not in name:
                        param.requires_grad = False
                else:
                    if "head" not in name and "subtle" not in name and "prompt" not in name:
                        param.requires_grad = False
            #     print(name)
            # exit()

            print('freezing parameters finished!')

            for k, v in model_or_params.named_parameters():
                # if v.requires_grad:
                #     # _logger.info(k+f' {v.shape} '+' {:.7f}  {:.7f}  '.format(v.mean().item(), v.var().item())+" ; ")
                #     _logger.info(k+f' {v.shape} '+' {:.7f}  {:.7f}  '.format(v.mean().item(), v.var().item())+" ; " + str(v.requires_grad))
                _logger.info(k+f' {v.shape} '+' {:.7f}  {:.7f}  '.format(v.mean().item(), v.var().item())+" ; " + str(v.requires_grad))


        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, 'no_weight_decay'):
            no_weight_decay = model_or_params.no_weight_decay()

        if param_group_fn:
            parameters = param_group_fn(model_or_params)
        elif layer_decay is not None:
            parameters = param_groups_layer_decay(
                model_or_params,
                weight_decay=weight_decay,
                layer_decay=layer_decay,
                no_weight_decay_list=no_weight_decay)
            weight_decay = 0.
        elif weight_decay and filter_bias_and_bn:
            parameters = param_groups_weight_decay(model_or_params, weight_decay, no_weight_decay)
            weight_decay = 0.
        else:
            parameters = model_or_params.parameters()





    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params



    opt_lower = opt.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)

    # basic SGD & related
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adamw':
        
        if tuning_mode == 'LoRA_B':
            lora_b_params = [p for name, p in model_or_params.named_parameters() if 'lora_b' in name]
            other_params = [p for name, p in model_or_params.named_parameters() if 'lora_b' not in name]
            optimizer = optim.AdamW([
                {'params': other_params, 'lr': lr},  # 其他参数使用全局学习率
                {'params': lora_b_params, 'lr': lr * 8}  # lora_b的参数学习率设为L*8
            ])
        elif tuning_mode == 'LoRA_head':
            head1 = [p for name, p in model_or_params.named_parameters() if 'head1' in name]
            print(f'head1: {head1}')
            other_params = [p for name, p in model_or_params.named_parameters() if 'head1' not in name]
            optimizer = optim.AdamW([
                {'params': other_params, 'lr': lr},  # 其他参数使用全局学习率
                {'params': head1, 'lr': lr / 5}  # head1的参数学习率设为L*8
            ])
            
        else:
            optimizer = optim.AdamW(parameters, **opt_args)
            
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'nadam':
        try:
            # NOTE PyTorch >= 1.10 should have native NAdam
            optimizer = optim.Nadam(parameters, **opt_args)
        except AttributeError:
            optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = optim.Adamax(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'radabelief':
        optimizer = AdaBelief(parameters, rectify=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == 'lambc':
        optimizer = Lamb(parameters, trust_clip=True, **opt_args)
    elif opt_lower == 'larc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, **opt_args)
    elif opt_lower == 'lars':
        optimizer = Lars(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'nlarc':
        optimizer = Lars(parameters, momentum=momentum, trust_clip=True, nesterov=True, **opt_args)
    elif opt_lower == 'nlars':
        optimizer = Lars(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'madgradw':
        optimizer = MADGRAD(parameters, momentum=momentum, decoupled_decay=True, **opt_args)
    elif opt_lower == 'novograd' or opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)

    # second order
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)

    # NVIDIA fused optimizers, require APEX to be installed
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
