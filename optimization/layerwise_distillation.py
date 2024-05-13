import copy

import torch
import torch.nn as nn
import torch_pruning as tp

from utils.checkpoint_handler import load_model_state
from optimization.distillation_training import distill_train
from optimization.low_rank_optimization import create_small_model
def freeze_except(model, layers_to_train=None):
  for param in model.parameters():
     param.requires_grad = False

  def enable_gradients(module, module_path=""):
     for child_name, child in module.named_children():

         child_path = f"{module_path}.{child_name}" if module_path else child_name
         if any(child_path.startswith(layer) for layer in layers_to_train):
            for param in child.parameters(recurse=True):
                    param.requires_grad = True
         else:
            enable_gradients(child, child_path)

  enable_gradients(model)
  return model

def enable_all_gradients(model):
 for parameter in model.parameters():
    parameter.requires_grad = True

def layerwise_distill_train(train_config, setup_config, teacher_model, train_dataloader, valid_dataloader, layers):
  student_model = copy.deepcopy(teacher_model)
  layer_map = {layer: layer for layer in layers}
  input_example = torch.rand(1, 3, *setup_config["input-size"])
  for idx, layer in enumerate(layers):
     student_model = create_small_model(copy.deepcopy(student_model), layers=[layer],
                                                 use_tucker=True if setup_config["student"] == "tucker" else False)

     ops, params = tp.utils.count_ops_and_params(student_model, input_example)
     print(f"Student model complexity: {ops / 1e6} MMAC, {params / 1e6} M params")

     freeze_except(student_model, [layer])

     student_model=distill_train(train_config=train_config,
                   setup_config=setup_config,
                   teacher_model=teacher_model,
                   student_model=student_model,
                   train_dataloader=train_dataloader,
                   valid_dataloader=valid_dataloader,
                   layers_map={layer: layer})

     torch.cuda.empty_cache()
     enable_all_gradients(student_model)

     student_model=distill_train(train_config=train_config,
                   setup_config=setup_config,
                   teacher_model=teacher_model,
                   student_model=student_model,
                   train_dataloader=train_dataloader,
                   valid_dataloader=valid_dataloader,
                   layers_map=layer_map)
  return student_model

