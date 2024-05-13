import os
import torch

from datasets.zürich_bicycle_dataset import zürich_collision_dataloaders
from train import train
from utils.data_io import load_config, create_model

if __name__ == "__main__":
  yaml_config = load_config("config/resnet-18-hint-train.yaml")

  train_setup = yaml_config["train_params"]
  setup_config = yaml_config["session_params"]

  train_dataloader, val_dataloader =\
      zürich_collision_dataloaders(root_dir=setup_config["data-root"],
                                   batch_size=setup_config["batch-size"],
                                   img_size=setup_config["input-size"],
                                   is_multiclass_problem= False if setup_config["classes"] == 1 else True)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = create_model(setup_config)
  model = train(train_dataloader=train_dataloader, valid_dataloader=val_dataloader, model=model,
            device=device, train_config=train_setup, setup_config=setup_config)

  model_pth = os.path.join(setup_config.get("checkpoint-dir", "out"), model.__class__.__name__ + "_final.pth")
  torch.save(model, model_pth)
