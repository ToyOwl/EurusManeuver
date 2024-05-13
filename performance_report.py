import argparse
import os
import pandas as pd

import numpy as np
import torch

from inference.performance_estimation import measure_model_performance
from inference.metrics_estimation import classification_metrics

from datasets.zürich_bicycle_dataset import zürich_collision_dataloaders
from utils.data_io import get_model

config = {"classes": 2,
          "decision-threshold": 0.5,
          "target-class": 1,
          "mean": (0.485, 0.456, 0.406),
          "std": (0.229, 0.224, 0.225)}


def main():
  parser = argparse.ArgumentParser(description="Measure Performance Metrics for PyTorch models on both CPU and GPU.")
  parser.add_argument("--models", nargs="+", help="List of model types or paths to model files (.pt or .pth)")
  parser.add_argument("--data_root", default='', type=str, metavar="PATH", help="path to dataset root dir")
  parser.add_argument("--batch_size", default=64, type=int,  help="size of val batch")
  args = parser.parse_args()


  _, val_dataloader =\
      zürich_collision_dataloaders(root_dir=args.data_root,
                                   batch_size=args.batch_size,
                                   img_size=(224, 224),
                                   is_multiclass_problem=True)

  devices =[torch.device("cpu")]
  if torch.cuda.is_available():
    devices.append(torch.device('cuda'))

  results = []
  for model_path in args.models:
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = get_model(model_name, model_path=model_path)
    for device in devices:
       if device.type == "cuda" and "qint" in model_name:
          continue

       latency, flops, ram_usage, macs, params = measure_model_performance(model, device)

       if device.type == "cuda" or "qint" in model_name:
          confmat = classification_metrics(model, val_dataloader, config)
          recall, accuracy, fpr, fnr = confmat["recall"], confmat["accuracy"], confmat["fpr"], confmat["fnr"]
       else:
          recall, accuracy, fpr, fnr = np.nan, np.nan, np.nan, np.nan

       results.append({
                "Model": model_name,
                "Device": str(device),
                "Latency (ms)": latency,
                "FLOPS (MFLOPS)": flops,
                "RAM Usage (Mb)": ram_usage,
                "MMACS": macs/1e6,
                "Params (M)": params/1e6,
                "Params": params,
                "MACS": macs,
                "Recall": recall,
                "Accuracy": accuracy,
                "FPR": fpr,
                "FNR": fnr
            })


  df = pd.DataFrame(results)
  df.to_csv("reports/performance_metrics.csv", index=False)
  print("Metrics Exported to reports/performance_metrics.csv")


if __name__ == "__main__":
    main()