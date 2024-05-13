import numpy as np
import torch

from tqdm import tqdm

from utils.metrics import *

from utils.tensor_ops import (to_numpy, normalize)

def classification_metrics(model, loader, setup_config, on_cpu=False):

  n_classes = setup_config["classes"]
  decision_threshold = setup_config["decision-threshold"]
  target_class = setup_config["target-class"]

  model.eval()

  device = torch.device("cuda" if torch.cuda.is_available() and on_cpu else "cpu")

  model = model.to(device)

  recalls, accuracies, fprs, fnrs = [], [], [], []

  mean, std = setup_config["mean"], setup_config["std"]

  with torch.no_grad():

    with tqdm(enumerate(loader), total=len(loader), desc="Metrics Estimation", leave=False) as metrics_pbar:
      for batch_idx, batch in enumerate(loader):

       sources, targets = batch[0], batch[1]

       sources, targets =\
                sources.to(device), targets.to(device, dtype=torch.float if n_classes == 1 else torch.long)

       sources = normalize(sources, mean=mean, std=std)

       outputs = model(sources)

       if n_classes == 1:
         targets = torch.unsqueeze(targets, 1)

       preds = to_numpy(outputs)
       truths = to_numpy(targets)

       if n_classes == 1:
           recalls.append(compute_recall(truths, preds, threshold=decision_threshold))
           accuracies.append(compute_accuracy(truths, preds, threshold=decision_threshold))
           fprs.append(compute_fpr(truths, preds, threshold=decision_threshold))
           fnrs.append(compute_fnr(truths, preds, threshold=decision_threshold))
       else:
           recalls.append(compute_recall_probs(truths, preds, class_index=target_class))
           accuracies.append(compute_accuracy_probs(truths, preds))
           fprs.append(compute_fpr_probs(truths, preds, class_index=target_class))
           fnrs.append(compute_fnr_probs(truths, preds, class_index=target_class))

       metrics_pbar.set_description("Metrics estimation: recalls={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
              recalls[batch_idx], accuracies[batch_idx], fprs[batch_idx], fnrs[batch_idx]))
       metrics_pbar.update()

  metrics_pbar.close()
  return {"recall": np.mean(recalls), "accuracy": np.mean(accuracies), "fpr": np.mean(fprs), "fnr": np.mean(fnrs)}