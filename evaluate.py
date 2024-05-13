import numpy as np
import torch

from tqdm import tqdm

from utils.metrics import*

from utils.logging import log_metrics

from utils.tensor_ops import (to_numpy, normalize)

def validate_func(model, loader, loss_func, device, log_writer, step, setup_config):

  n_classes = setup_config["classes"]
  decision_threshold = setup_config["decision-threshold"]
  target_class = setup_config["target-class"]

  model.eval()

  losses, recalls, accuracies, fprs, fnrs = [], [], [], [], []

  mean, std = setup_config['mean'], setup_config['std']

  with torch.no_grad():

    with tqdm(enumerate(loader), total=len(loader), desc="Valid", leave=False) as valid_pbar:
      for batch_idx, batch in enumerate(loader):

       sources, targets = batch[0], batch[1]

       sources, targets =\
                sources.to(device), targets.to(device, dtype=torch.float if n_classes == 1 else torch.long)

       sources = normalize(sources, mean=mean, std=std)

       outputs = model(sources)

       if n_classes == 1:
         targets = torch.unsqueeze(targets, 1)

       loss = loss_func(outputs, targets)
       losses.append(loss.item())

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

       intermediate_step = step * len(loader) + batch_idx
       log_metrics(log_writer,
                   prefix="valid",
                   loss=losses[batch_idx],
                   recall=recalls[batch_idx],
                   accuracy=accuracies[batch_idx],
                   fpr=fprs[batch_idx],
                   fnr=fnrs[batch_idx],
                   idx=intermediate_step)

       valid_pbar.set_description("Valid: loss={},recalls={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
             losses[batch_idx], recalls[batch_idx], accuracies[batch_idx], fprs[batch_idx], fnrs[batch_idx]))
       valid_pbar.update()

  log_metrics(log_writer,
              prefix="valid-epoch",
              loss=np.mean(losses),
              recall=np.mean(recalls),
              accuracy=np.mean(accuracies),
              fpr=np.mean(fprs),
              fnr=np.mean(fnrs),
              idx=step)

  valid_pbar.close()
  return {"loss": np.mean(losses), "recall": np.mean(recalls), "accuracy": np.mean(accuracies),
             "fpr": np.mean(fprs), "fnr": np.mean(fnrs), "step": step}