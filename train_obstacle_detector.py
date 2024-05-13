from utils.data_io import (load_config, create_model, create_scheduler, create_optimizer, create_loss_fn)

if __name__ == "__main__":
  config = load_config("config/resnet-18-hint-train.yaml")
  model  = create_model(config["train_params"])
  optimizer = create_optimizer(model.parameters(), config["train_params"]["optimizer-parameters"])
  pass

