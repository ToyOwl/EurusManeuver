import inspect
import io
import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

import timm
from timm.optim import (AdaBelief, Adafactor, Adahessian, AdamP, Nadam, Lookahead,
                        NvNovoGrad, RAdam, RMSpropTF, SGDP)
from timm.scheduler import (CosineLRScheduler, TanhLRScheduler, PlateauLRScheduler, StepLRScheduler)
from timm.loss import LabelSmoothingCrossEntropy

from models.resnet_model import ResNetHinton, ResNetWrapper
from loss import (HintonLoss, HardMiningLoss)

from optimization.quantization import load_quantized_model

def clip_grad(parameters, value: float, mode="norm", norm_type=2.0):
  if mode == "norm":
    torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
  elif mode == "value":
    torch.nn.utils.clip_grad_value_(parameters, value)
  else:
    raise ValueError(f"Unknown clip mode ({mode}).")

def get_optimizer(optimizer_name, model_params, **kwargs):

    optimizer_map = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adamax': optim.Adamax,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
        'sgdp': SGDP,
        'adamp': AdamP,
        'nadam': Nadam,
        'radam': RAdam,
        'adabelief': lambda params, **kw: AdaBelief(params, rectify=False, **kw),
        'radabelief': lambda params, **kw: AdaBelief(params, rectify=True, **kw),
        'adafactor': Adafactor,
        'novograd': NvNovoGrad,
        'rmsproptf': RMSpropTF,
        'adahessian': Adahessian
    }

    optimizer_class = optimizer_map.get(optimizer_name.lower())
    if not optimizer_class:
        raise ValueError(f"Invalid optimizer {optimizer_name}")

    sig = inspect.signature(optimizer_class)
    supported_params = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return optimizer_class(model_params, **supported_params)
def get_scheduler(scheduler_name, optimizer, **kwargs):
    scheduler_map = {
        'cosine': CosineLRScheduler,
        'tanh': TanhLRScheduler,
        'step': StepLRScheduler,
        'plateau': PlateauLRScheduler
    }
    scheduler_class = scheduler_map.get(scheduler_name.lower())
    if not scheduler_class:
        raise ValueError(f"Invalid scheduler {scheduler_name}")
    return scheduler_class(optimizer, **kwargs)
def create_optimizer(model_params, lr_params):
    optimizer_config = {
        "lr": lr_params["lr"],
        "weight_decay": lr_params["weight-decay"],
        "momentum": lr_params.get("momentum"),
        "nesterov": lr_params.get("nesterov", False),
        "betas": lr_params.get("betas"),
        "eps": lr_params.get("epsilon")
    }
    optimizer = get_optimizer(lr_params["optimizer"], model_params, **optimizer_config)
    if lr_params.get("lookahead"):
        optimizer = Lookahead(optimizer, k=lr_params["lookahead-steps"])
    return optimizer
def create_scheduler(optimizer, sch_params):
    scheduler_config = {
        't_initial': sch_params['epochs'],
        'lr_min': sch_params['min-lr'],
        'warmup_lr_init': sch_params['warmup-lr'],
        'warmup_t': sch_params['warmup-epochs'],
        'noise_range_t': sch_params['lr-noise'] * sch_params['epochs'],
        'noise_pct': sch_params['lr-noise-pct'],
        'noise_std': sch_params['lr-noise-std'],
        'noise_seed': sch_params['seed']
    }
    return get_scheduler(sch_params["sched"], optimizer, **scheduler_config)
def get_loss_function(loss_name, **kwargs):
    loss_map = {
        "smoothing": LabelSmoothingCrossEntropy,
        "crossentropy": nn.CrossEntropyLoss,
        "hinton": HintonLoss,
        "bceloss":nn.BCEWithLogitsLoss
    }
    loss_func = loss_map.get(loss_name.lower())
    if not loss_func:
        raise ValueError(f"Invalid loss function {loss_name}")

    sig = inspect.signature(loss_func)
    supported_params = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return loss_func(**supported_params)

def create_loss_fn(loss_name,  loss_params, device="cpu"):

  loss_map = {
      "reduction": loss_params["reduction"]}
  if "weight" in loss_map.keys():
    loss_map["weight"] = torch.as_tensor(loss_params["weight"]).to(device)
  elif "pos_weight" in loss_map.keys():
    loss_map["pos_weight"] = torch.as_tensor([loss_params["pos_weight"]]).to(device)

  loss_fn = get_loss_function(loss_name, **loss_map)

  if loss_params.get("hard-mining"):
     return HardMiningLoss(loss_fn, **loss_params.get("hard-mining-params"))

  return loss_fn

def create_model(session_params):
  if not "resnet" in session_params["model"].lower():
    raise ValueError(f"currently supported models only ResNet family")

  if session_params["classes"] == 1:
      return ResNetWrapper(session_params["model"], **session_params["model-params"]) if "model-params" \
                                                                in session_params else ResNetWrapper(session_params["model"])

  model = ResNetHinton(session_params["model"], **session_params["model-params"]) if "model-params" \
                                                                in session_params else ResNetHinton(session_params["model"])
  return model

def load_config(config_file):
  with io.open(os.path.abspath(config_file)) as file:
      config = yaml.load(file, Loader=yaml.FullLoader)
  return config

def load_model(model_path):
  if os.path.exists(model_path):
     return torch.load(model_path)
  else:
     raise FileNotFoundError(f"No model found at {model_path}")

def get_model(model_name, model_path):
  if "fp32" in model_name:
    return load_model(model_path)
  elif "qint" in model_name:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    qat_model = load_quantized_model(checkpoint["base_model"],checkpoint["quantized_state"])
    del checkpoint
    return qat_model
  else:
    return None