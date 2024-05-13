import itertools
from itertools import islice

import copy
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init

import torch_pruning as tp

from utils.layer_features import find_linear_layers
from utils.tensor_ops import normalize

def create_small_model(model, data_loader, pruning_conf, setup_config, loss_func=None):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  mean, std = setup_config['mean'], setup_config['std']
  n_classes = setup_config["classes"]

  n_batches = pruning_conf.get("num-batches", 10)
  steps = pruning_conf.get("steps", 5)

  pruning_ratio = pruning_conf["pruning-ratio"]
  is_taylor = pruning_conf["is-taylor"]

  distill_model = copy.deepcopy(model)

  ignored_layers =\
    [find_linear_layers(distill_model, submodule) for submodule in pruning_conf["ignored-layers"]]
  ignored_layers = itertools.chain.from_iterable(ignored_layers)


  examples = []
  for image, _ in islice(data_loader, n_batches):
    examples.append(normalize(image, mean=mean, std=std))

  input_examples = torch.cat(examples, dim=0)

  input_examples = input_examples.to(device)
  distill_model = distill_model.to(device)

  if not is_taylor:
    l2_importance = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
    pruner = tp.pruner.MagnitudePruner(model=distill_model, example_inputs=input_examples, global_pruning=False,
            importance=l2_importance, iterative_steps=steps, pruning_ratio=pruning_ratio, ignored_layers=ignored_layers)
    pruner.step()
    return distill_model.cpu()

  taylor_criteria = tp.importance.GroupTaylorImportance()
  pruner = tp.pruner.MetaPruner(model=distill_model, example_inputs=input_examples, importance=taylor_criteria,
    pruning_ratio=pruning_ratio, global_pruning=False, ignored_layers=ignored_layers)

  distill_model.train()

  pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Taylor Pruning")

  for _, batch in pbar:
    sources, targets = batch[0], batch[1]

    sources, targets = (
      sources.to(device), targets.to(device, dtype=torch.float if n_classes == 1 else torch.long))

    sources = normalize(sources, mean=mean, std=std)

    logits = distill_model(sources)

  if n_classes == 1:
    targets = torch.unsqueeze(targets, 1)

  loss = loss_func(logits, targets)

  loss.backward()

  for idx, g in enumerate(pruner.step(interactive=True)):
    g.prune()

  return distill_model.cpu()
