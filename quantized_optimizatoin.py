import copy
import os

import torch
import torch_pruning as tp

from datasets.zürich_bicycle_dataset import zürich_collision_dataloaders
from utils.data_io import create_model, create_loss_fn, load_config
from utils.checkpoint_handler import load_model_state

from optimization.distillation_training import distill_train
from optimization.quantization import quantized_static, fake_quantization, create_int_model

from inference.metrics_estimation import classification_metrics

if __name__ == "__main__":

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  yaml_config = load_config("config/distill-resnet18-quantization.yaml")

  train_setup = yaml_config["train_params"]
  setup_config = yaml_config["session_params"]

  if setup_config.get("teacher-checkpoint", None) is not None:
      teacher = create_model(setup_config)
      teacher = load_model_state(teacher, setup_config["teacher-checkpoint"])
  else:
      teacher = torch.load(setup_config["teacher-model"])

  teacher_copy = copy.deepcopy(teacher)

  mean, std = setup_config["mean"], setup_config["std"]

  input_example = torch.rand(1, 3, *setup_config["input-size"])

  ops, params = tp.utils.count_ops_and_params(teacher, input_example)
  print(f"Baseline model complexity: {ops / 1e6} MMAC, {params / 1e6} M params")

  layers = train_setup.get("intermediate-layers", None)
  n_batches = setup_config["quantization-params"].get("num-batches", 10)

  train_dataloader, val_dataloader =\
      zürich_collision_dataloaders(root_dir=setup_config["data-root"],
                                   batch_size=setup_config["batch-size"],
                                   img_size=setup_config["input-size"],
                                   is_multiclass_problem=False if setup_config["classes"] == 1 else True)


  student_model = copy.deepcopy(teacher)

  if setup_config["quantization-params"]["type"] != "training-aware":
     qat_model=\
         quantized_static(student_model, train_dataloader, n_batches, device, mean, std)

     confmat = classification_metrics(model=student_model, loader=val_dataloader, setup_config=setup_config, on_cpu=True)

     print("\nQuantized Student Metrics: recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
          confmat["recall"], confmat["accuracy"], confmat["fpr"], confmat["fnr"]))

     model_name = "quantized_"+ teacher.__class__.__name__ + "_{:.3f}.pth".format(
         confmat["recall"])

     torch.save({"base_model":teacher, "quantized_state": qat_model.model.state_dict()},
                os.path.join(setup_config.get("checkpoint-dir", "out"), model_name))

  else:
     qat_model = \
          fake_quantization(student_model, train_dataloader, n_batches,  mean, std)

     ops, params = tp.utils.count_ops_and_params(qat_model, input_example)
     print(f"Student model complexity: {ops / 1e6} MMAC, {params / 1e6} M params")

     student_loss = create_loss_fn(train_setup["loss"], train_setup["loss-params"], device)

     confmat = classification_metrics(model=qat_model, loader=val_dataloader, setup_config=setup_config)

     print("\nQuantized Student Metrics: recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
           confmat["recall"], confmat["accuracy"], confmat["fpr"], confmat["fnr"]))

     qat_model = distill_train(train_config=train_setup,
                               setup_config=setup_config,
                               teacher_model=teacher,
                               student_model=qat_model,
                               train_dataloader=train_dataloader,
                               valid_dataloader=val_dataloader)

     confmat = classification_metrics(model=qat_model, loader=val_dataloader, setup_config=setup_config)

     print("\nStudent Metrics After QAT: recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
            confmat["recall"], confmat["accuracy"], confmat["fpr"], confmat["fnr"]))


     int_model = create_int_model(qat_model)

     confmat = classification_metrics(model=int_model, loader=val_dataloader, setup_config=setup_config)

     print("\nStudent Metrics After Tuning: recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
          confmat["recall"], confmat["accuracy"], confmat["fpr"], confmat["fnr"]))

     model_name = "quantized_int_" + teacher.__class__.__name__ + "_{:.3f}.pth".format(confmat["recall"])

     torch.save({"base_model": teacher_copy, "quantized_state": int_model.state_dict()},
           os.path.join(setup_config.get("checkpoint-dir", "out"), model_name))

     torch.save(int_model, os.path.join(setup_config.get("checkpoint-dir", "out"), "ResNetHintonint8.pth"))
