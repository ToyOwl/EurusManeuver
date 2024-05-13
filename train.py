import copy
import os

from tqdm import tqdm

import numpy as np
import torch

import torchvision.transforms.functional as tf

from timm.utils import dispatch_clip_grad

from evaluate import validate_func

from utils.logging import (log_metrics, visualize_samples)

from utils.metrics import *

from utils.data_io import (create_optimizer, create_scheduler, create_loss_fn)

from utils.checkpoint_handler import (CheckpointSaver, save_checkpoint)

from utils.tensor_ops import (to_numpy, normalize)

from visualdl import LogWriter


def train(model, train_dataloader, valid_dataloader, device, train_config, setup_config):

  model = model.to(device)

  save_frequency = setup_config.get("save-frequency", 200)

  checkpoint_path = setup_config.get("checkpoint-dir", "out")

  log_writer = LogWriter(logdir=setup_config.get("log-path", "logs"))

  optimizer = create_optimizer(model.parameters(), train_config["optimizer-parameters"])
  scheduler = create_scheduler(optimizer, train_config["scheduler-parameters"])

  collision_loss = create_loss_fn(train_config["loss"], train_config["loss-params"], device)

  grad_scaler = torch.cuda.amp.GradScaler() if train_config.get("grad-scaler", False) and device == "cuda" else None
  clip_params = train_config["clip-parameters"]

  eval_metric = setup_config.get("eval-metric", "loss")
  n_classes = setup_config["classes"]
  decision_threshold = setup_config["decision-threshold"]
  target_class = setup_config["target-class"]

  mean, std = setup_config["mean"], setup_config["std"]

  checkpoint_saver = CheckpointSaver(device, model, optimizer, scheduler,  dir=checkpoint_path,
        is_loss=False if eval_metric != "loss" else True, max_history=setup_config.get("checkpoint-history", 10))

  start_epoch, start_batch = 0, 0

  if setup_config["initial-checkpoint"]:
    start_epoch, start_batch =\
            restore_from_checkpoint(setup_config["initial-checkpoint"], model, optimizer, scheduler, device)

  losses, accuracies, recalls, fprs, fnrs = [], [], [], [], []

  try:

    for epoch in range(start_epoch, setup_config["epochs"]):

     model.train()

     pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Train")


     losses, accuracies, recalls, fprs, fnrs = [], [], [], [], []

     for batch_idx, batch in pbar:
        if batch_idx < start_batch:
          continue

        sources, targets = batch[0], batch[1]
        viz_samples = copy.deepcopy(sources)

        sources, targets = (sources.to(device), targets.to(device, dtype=torch.float if n_classes == 1 else torch.long))

        sources = normalize(sources, mean=mean, std=std)

        if n_classes == 1:
          targets = torch.unsqueeze(targets, 1)

        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
           outputs = model(sources)
           loss = collision_loss(outputs, targets)

        optimizer.zero_grad()

        if grad_scaler is not None:
           grad_scaler.scale(loss).backward()
           grad_scaler.step(optimizer)
           grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if clip_params['mode']:
          dispatch_clip_grad(model.parameters(), clip_params['gradient'], clip_params['mode'])

        preds = to_numpy(outputs)
        truths = to_numpy(targets)

        losses.append(loss.item())

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

        intermediate_step = epoch * len(train_dataloader) + batch_idx

        log_metrics(log_writer,
                        prefix="train",
                        loss=losses[batch_idx],
                        recall=recalls[batch_idx],
                        accuracy=accuracies[batch_idx],
                        fpr=fprs[batch_idx],
                        fnr=fnrs[batch_idx],
                        idx=intermediate_step)

        if (batch_idx + 1) % save_frequency == 0:
            metrics = {"loss": np.mean(losses),
                       "accuracy": np.mean(accuracies),
                       "recall": np.mean(recalls),
                       "fpr": np.mean(fprs),
                       "fnr": np.mean(fnrs)}
            checkpoint_saver.save_checkpoint(epoch, batch_idx, metrics[eval_metric])

        visualize_samples(log_writer, intermediate_step, to_numpy(viz_samples), np.argmax(preds, axis=1), targets)

        pbar.set_description("Train {}/{}: loss={}, recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
                  epoch+1, setup_config["epochs"], losses[batch_idx], recalls[batch_idx], accuracies[batch_idx],
                                                                                    fprs[batch_idx], fnrs[batch_idx]))

     pbar.close()
     log_metrics(log_writer,
                 prefix="train-epoch",
                 loss=np.mean(losses),
                 recall=np.mean(recalls),
                 accuracy=np.mean(accuracies),
                 fpr=np.mean(fprs),
                 fnr=np.mean(fnrs),
                 idx=epoch)

     metrics = validate_func(model, valid_dataloader, collision_loss, device, log_writer, epoch, setup_config)

     checkpoint_saver.save_checkpoint(epoch,model.state_dict(), batch_idx, metrics[setup_config["eval-metric"]])

     scheduler.step(epoch)
  except KeyboardInterrupt:
    metrics = {"loss": np.mean(losses),
               "accuracy": np.mean(accuracies),
               "recall": np.mean(recalls),
               "fpr": np.mean(fprs),
               "fnr": np.mean(fnrs)}
    save_checkpoint(model,
                    os.path.join(checkpoint_path, "checkpoint_interrupt.pth"),
                    optimizer,
                    scheduler,
                    epoch,
                    batch_idx,
                    metrics[setup_config["eval-metric"]])

  return copy.deepcopy(model.cpu())