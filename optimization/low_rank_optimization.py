import numpy as np
import torch
import torch.nn as nn

from tensorly.decomposition import tucker, parafac, partial_tucker
from optimization.matrix_factorization import tucker_rank, cp_rank

def cp_decompose_conv_layer(layer, rank=None):

  if rank is None:
   rank = cp_rank(layer)

  weight = layer.weight.data.cpu().numpy()
  _, factors = parafac(weight, rank=rank, init='random')
  last, first, vertical, horizontal = factors

  s_to_r_pointwise = nn.Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], kernel_size=1, padding=0,
                                                                                                          bias=False)

  r_to_r_depthwise = nn.Conv2d(in_channels=rank, out_channels=rank, kernel_size=vertical.shape[0],
           stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=rank, bias=False)

  r_to_t_pointwise = nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[0], kernel_size=1, padding=0,
                                                                   bias=True if layer.bias is not None else False)

  if layer.bias is not None:
    r_to_t_pointwise.bias.data = layer.bias.data

  vertical = torch.tensor(vertical, dtype=torch.float32)
  horizontal = torch.tensor(horizontal, dtype=torch.float32)

  s_to_r_pointwise.weight.data = torch.tensor(first, dtype=torch.float32).t().unsqueeze(-1).unsqueeze(-1)
  r_to_t_pointwise.weight.data = torch.tensor(last, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
  r_to_r_depthwise.weight.data = torch.stack([
        vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze(1)

  layers = [s_to_r_pointwise, r_to_r_depthwise, r_to_t_pointwise]

  return nn.Sequential(*layers)

def tucker_decompose_conv_layer(layer, rank=None):

  weights = layer.weight.data.cpu().numpy()

  if rank is None:
   [rank1, rank2] = tucker_rank(layer)
  else:
    rank1, rank2 = rank, rank

  [core, factors], _ = partial_tucker(weights, rank=[rank1, rank2], modes =[0,1])

  first_layer = nn.Conv2d(in_channels=factors[1].shape[0], out_channels=factors[1].shape[1], kernel_size=1, stride=1,
                                                                         padding=0, dilation=layer.dilation, bias=False)

  first_layer.weight.data =\
      torch.transpose(torch.tensor(factors[1], dtype=torch.float32), 1, 0).unsqueeze(-1).unsqueeze(-1)


  core_layer = nn.Conv2d(in_channels=core.shape[1], out_channels=core.shape[0], kernel_size=layer.kernel_size,
                               stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=False)
  core_layer.weight.data = torch.tensor(core, dtype=torch.float32)

  last_layer = nn.Conv2d(in_channels=factors[0].shape[1], out_channels=factors[0].shape[0], kernel_size=1,
                            stride=1, padding=0, dilation=layer.dilation,  bias=layer.bias is not None)
  last_layer.weight.data = torch.tensor(factors[0], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

  if layer.bias is not None:
     new_last_layer.bias.data = layer.bias.data.clone()

  layers= [first_layer, core_layer, last_layer]
  return nn.Sequential(*layers)

def decompose_conv_layer(layer, rank=None, use_tucker=True):
  if use_tucker:
    return tucker_decompose_conv_layer(layer, rank)

  return cp_decompose_conv_layer(layer, rank)

def create_small_model(model, layers=None, rank=None, use_tucker=True):

  if layers is None:
    layers = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]

  def recurse_model(module, module_path=''):
     for child_name, child in module.named_children():
        child_path = f'{module_path}.{child_name}' if module_path else child_name
        if any(child_path.startswith(layer) for layer in layers):
          if isinstance(child, nn.Conv2d):
            setattr(module, child_name,decompose_conv_layer(child, rank, use_tucker))
          else:
            recurse_model(child, child_path)
        else:
          recurse_model(child, child_path)

  recurse_model(model)
  return model