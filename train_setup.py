import torch
import torch.nn as nn
import torch.optim as optim

from timm.optim.adabelief import AdaBelief
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.nadam import Nadam
from timm.optim.lookahead import Lookahead
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

from timm.loss.cross_entropy import LabelSmoothingCrossEntropy


def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    else:
        assert False, f'Unknown clip mode ({mode}).'

def create_optimizer(model_params, lr_params):

    _optimizer = lr_params['optimizer'].lower()

    weight_decay = lr_params['weight-decay']
    lr = lr_params['lr']
    momentum = lr_params['momentum']
    eps = lr_params['epsilon']
    betas = lr_params['betas']
    clip_gradient = lr_params['clip-gradient']

    if _optimizer == 'sgd' and lr_params['nesterov']:
        optimizer = optim.SGD(model_params, lr=lr,
                              momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif _optimizer == 'momentum':
        optimizer = optim.SGD(model_params, lr=lr,
                              momentum=momentum, nesterov=False, weight_decay=weight_decay)
    elif _optimizer == 'sgdp':
        optimizer = SGDP(model_params, momentum=momentum,
                         nesterov=True, weight_decay=weight_decay, eps=eps)
    elif _optimizer == 'adam':
        optimizer = optim.Adam(model_params, lr=lr,
                               betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adamw':
        optimizer = optim.AdamW(model_params, lr=lr,
                                betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adamp':
        optimizer = AdamP(model_params, wd_ratio=0.01, nesterov=True, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'nadam':
        optimizer = Nadam(model_params, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'radam':
        optimizer = RAdam(model_params, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adamax':
        optimizer = optim.Adamax(model_params,lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adabelief':
        optimizer = AdaBelief(model_params, rectify=False,lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'radabelief':
        optimizer = AdaBelief(model_params, rectify=True, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adadelta':
        optimizer = optim.Adadelta(model_params, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adagrad':
        optimizer = optim.Adagrad(model_params,lr=lr,
                                        betas=betas, eps=1e-8, weight_decay=weight_decay)
    elif _optimizer == 'adafactor':
        optimizer = Adafactor(model_params, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay, clip_threshold=clip_gradient)
    elif _optimizer == 'novograd':
        optimizer = NvNovoGrad(model_params,  lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model_params, alpha=0.9, momentum=momentum,
                                        lr=lr, eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'rmsproptf':
        optimizer = RMSpropTF(model_params, alpha=0.9, momentum=momentum,lr=lr,
                                             eps=eps, weight_decay=weight_decay)
    elif _optimizer == 'adahessian':
        optimizer = Adahessian(model_params, lr=lr,
                                        betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if lr_params['lookahead']:
       optimizer = Lookahead(optimizer, k=lr_params['lookahead-steps'])

    return optimizer

def create_scheduler(optimizer, sch_params):

    seed = sch_params['seed']

    num_epochs = sch_params['epochs']

    lr_noise = sch_params['lr-noise']

    lr_noise_pct = sch_params['lr-noise-pct']
    lr_noise_std = sch_params['lr-noise-std']

    cycle_mul = sch_params['lr-cycle-mul']
    cycle_decay = sch_params['lr-cycle-decay']
    cycle_limit = sch_params['lr-cycle-limit']


    min_lr = sch_params['min-lr']
    warmup_lr = sch_params['warmup-lr']
    warmup_epochs = sch_params['warmup-epochs']
    decay_epochs  = sch_params['decay-epochs']
    decay_rate = sch_params['decay-rate']
    patience_epochs = sch_params['patience-epochs']

    if isinstance(lr_noise, (list, tuple)):
        noise_range = [n * num_epochs for n in lr_noise]
        if len(noise_range) == 1:
            noise_range = noise_range[0]
    else:
        noise_range = lr_noise * num_epochs
    '''
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(sch_params, 'lr_noise_pct', 0.67),
        noise_std=getattr(sch_params, 'lr_noise_std', 1.),
        noise_seed=getattr(sch_params, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(sch_params, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(sch_params, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(sch_params, 'lr_cycle_limit', 1),
    )
    '''
    lr_scheduler = None
    if sch_params['sched'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,

            #k_decay=1.0,

            noise_range_t=noise_range,

            decay_rate=cycle_decay,
            t_mul=cycle_mul,
            cycle_limit=cycle_limit,

            noise_pct=lr_noise_pct,
            noise_std=lr_noise_std,
            noise_seed=seed)

    elif sch_params['sched'] == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            t_in_epochs=True,

            noise_range_t=noise_range,

            cycle_decay=cycle_decay,
            cycle_mul=cycle_mul,
            cycle_limit=cycle_limit,

            noise_pct=lr_noise_pct,
            noise_std=lr_noise_std,
            noise_seed=seed)

    elif sch_params['sched'] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_epochs,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,

            noise_range_t=noise_range,

            noise_pct=lr_noise_pct,
            noise_std=lr_noise_std,
            noise_seed=seed)

    elif sch_params['sched'] == 'plateau':
        mode = 'min' if 'loss' == sch_params['track-func'] else 'max'

        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=decay_rate,
            patience_t=patience_epochs,
            lr_min=sch_params.min_lr,
            mode=mode,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_epochs,
            cooldown_t=0,

            noise_range_t=noise_range,

            noise_pct=lr_noise_pct,
            noise_std=lr_noise_std,
            noise_seed=seed )

    num_epochs = lr_scheduler.get_cycle_length() + sch_params['cooldown-epochs']

    return lr_scheduler, num_epochs

def create_loss_fn(model_params):
    if model_params['smoothing']:
        loss_fn = LabelSmoothingCrossEntropy(model_params['smoothing'])
    else:
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn if model_params['device']=='cpu' else loss_fn.cuda()
    return loss_fn
