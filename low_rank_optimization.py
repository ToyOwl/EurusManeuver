import copy
import os

import torch
import torch_pruning as tp

from datasets.zürich_bicycle_dataset import zürich_collision_dataloaders
from utils.data_io import create_model, create_loss_fn, load_config
from utils.checkpoint_handler import load_model_state

from optimization.distillation_training import distill_train
from optimization.layerwise_distillation import layerwise_distill_train
from optimization.low_rank_optimization import create_small_model

from inference.metrics_estimation import classification_metrics

if __name__ == "__main__":

  yaml_config = load_config("config/distill-resnet18-tucker-one-shot.yaml")

  train_setup = yaml_config["train_params"]
  setup_config = yaml_config["session_params"]

  if setup_config.get("teacher-checkpoint", None) is not None:
     teacher = create_model(setup_config)
     teacher = load_model_state(teacher, setup_config["teacher-checkpoint"])
  else:
     teacher = torch.load(setup_config["teacher-model"])

  input_example = torch.rand(1, 3, *setup_config["input-size"])

  ops, params = tp.utils.count_ops_and_params(teacher, input_example)
  print(f"Baseline model complexity: {ops / 1e6} MMAC, {params / 1e6} M params")

  layers = train_setup.get("intermediate-layers", None)

  train_dataloader, val_dataloader =\
      zürich_collision_dataloaders(root_dir=setup_config["data-root"],
                                   batch_size=setup_config["batch-size"],
                                   img_size=setup_config["input-size"],
                                   is_multiclass_problem=False if setup_config["classes"] == 1 else True)

  layers_map = None

  if layers is not None:
   layers_map = {layer: layer for layer in layers}


  if setup_config["one-shot"]:
    student_model = copy.deepcopy(teacher)
    student_model = create_small_model(student_model, layers=layers,
                                         use_tucker=True if setup_config["student"] == "tucker" else False)


    ops, params = tp.utils.count_ops_and_params(student_model, input_example)
    print(f"Student model complexity: {ops / 1e6} MMAC, {params / 1e6} M params")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_loss = create_loss_fn(train_setup["loss"], train_setup["loss-params"], device)

    student_metrics = classification_metrics(model=student_model, loader=val_dataloader, setup_config=setup_config)

    print("\nStudent Metrics: recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
        student_metrics["recall"], student_metrics["accuracy"], student_metrics["fpr"], student_metrics["fnr"]))

    student_model = distill_train(train_config=train_setup,
                                   setup_config=setup_config,
                                   teacher_model=teacher,
                                   student_model=student_model,
                                   train_dataloader=train_dataloader,
                                   valid_dataloader=val_dataloader,
                                   layers_map=layers_map)
  else:
    student_model = layerwise_distill_train(train_config=train_setup,
                                             setup_config=setup_config,
                                             teacher_model=copy.deepcopy(teacher),
                                             train_dataloader=train_dataloader,
                                             valid_dataloader=val_dataloader,
                                             layers=layers)


  student_metrics = classification_metrics(model=student_model, loader=val_dataloader, setup_config=setup_config)

  print("\nStudent Metrics After Tuning: recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
         student_metrics["recall"], student_metrics["accuracy"], student_metrics["fpr"], student_metrics["fnr"]))

  model_name = student_model.__class__.__name__ + "_final.pth"

  student_model.zero_grad()
  model_pth = os.path.join(setup_config.get("checkpoint-dir", "out"), model_name)
  torch.save(student_model, model_pth)