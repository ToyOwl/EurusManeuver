import copy
from tqdm import tqdm
from visualdl import LogWriter

import torch
import torch.nn as nn

from utils.layer_features import *

from evaluate import validate_func

from utils.logging import (log_metrics, visualize_samples)

from utils.metrics import *

from utils.data_io import (create_optimizer, create_scheduler, create_loss_fn)

from utils.checkpoint_handler import (CheckpointSaver, save_checkpoint, load_model_state)

from utils.tensor_ops import (to_numpy, normalize)

from loss import (intermediate_layers_loss, kl_loss, penultimate_loss_func)

def log_metrics_distill(log_writer, prefix, idx, loss, penultimate_loss, intermediate_loss,
                                                                                recall, accuracy, fpr, fnr):

    log_writer.add_scalar(tag=f"penultimate-loss/{prefix}", value=penultimate_loss, step=idx)

    if intermediate_loss is not None:
      log_writer.add_scalar(tag=f"intermediate-loss/{prefix}", value=intermediate_loss, step=idx)

    log_metrics(log_writer, prefix, idx, loss, recall, accuracy, fpr, fnr)
def distill_train(train_config, setup_config, teacher_model, student_model,
                                                                train_dataloader, valid_dataloader, layers_map=None):


  state_holder = copy.deepcopy(student_model)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  save_frequency = setup_config.get("save-frequency", 200)
  checkpoint_path = setup_config.get("checkpoint-dir", "out")

  temperature = train_config["temperature"]
  penultimate_weight = train_config["penultimate-weight"]
  inter_weights = train_config["intermediate-weights"]

  grad_scaler = torch.cuda.amp.GradScaler() if train_config.get("grad-scaler", False) else None

  log_writer = LogWriter(logdir=setup_config.get("log-path", "logs"))

  student_loss = create_loss_fn(train_config["loss"], train_config["loss-params"], device)

  eval_metric = setup_config.get("eval-metric", "loss")
  n_classes = setup_config["classes"]
  decision_threshold = setup_config["decision-threshold"]
  target_class = setup_config["target-class"]

  mean, std = setup_config['mean'], setup_config['std']

  teacher_student_fits, msk = None, None

  if layers_map is not None:
    teacher_student_layers, feats_mapping =\
            get_layer_feats(teacher_model, student_model, layers_map)

    teacher_student_fits, msk = fit_net_projections(feats_mapping)

    teacher_features, student_features =\
          register_intermediate_features(teacher_model, student_model, teacher_student_layers)


  penult_teacher_features = register_layer_features(teacher_model, [train_config["penultimate-layer"]])
  penult_student_features = register_layer_features(student_model, [train_config["penultimate-layer"]])

  if teacher_student_fits is not None:
     teacher_student_fits.to(device)

  teacher_model.to(device)
  student_model.to(device)
  state_holder.to(device)

  teacher_model.eval()

  if teacher_student_fits:
      optimizer =\
        torch.optim.AdamW(list(student_model.parameters()) + list(teacher_student_fits.parameters()),
                                                                                               lr=train_config["lr"])
  else:
      optimizer = \
            torch.optim.AdamW(list(student_model.parameters()), lr=train_config["lr"])

  checkpoint_saver = CheckpointSaver(device, state_holder, optimizer,
                dir=checkpoint_path,
                save_state=setup_config.get("state-save", False),
                is_loss=False if eval_metric != "loss" else True,
                max_history=setup_config.get("checkpoint-history", 10))


  for epoch in range(setup_config["epochs"]):

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Train")

    total_losses, accuracies, recalls, fprs, fnrs = [], [], [], [], []
    penultimate_losses, intermediate_losses = [], []

    student_model.train()

    for batch_idx, batch in pbar:

      sources, targets = batch[0], batch[1]

      sources, targets = (
         sources.to(device), targets.to(device, dtype=torch.float if n_classes == 1 else torch.long))

      sources = normalize(sources, mean=mean, std=std)

      optimizer.zero_grad()

      if layers_map is not None:
        teacher_features.clear()
        student_features.clear()

      penult_teacher_features.clear()
      penult_student_features.clear()

      student_logits = student_model(sources)

      if n_classes == 1:
          targets = torch.unsqueeze(targets, 1)

      loss = student_loss(student_logits, targets)

      with torch.no_grad():
        _ = teacher_model(sources)

      penultimate_loss =\
          penultimate_loss_func(penult_student_features[0], penult_teacher_features[0],  temperature)

      total_loss = loss * penultimate_weight + (1 - penultimate_weight)*penultimate_loss

      if layers_map is not None:
        intermediate_loss =\
             intermediate_layers_loss(student_features, teacher_features, inter_weights, teacher_student_fits, msk)

        total_loss +=intermediate_loss


      if grad_scaler is not None:
       grad_scaler.scale(loss).backward()
       grad_scaler.step(optimizer)
       grad_scaler.update()
      else:
       total_loss.backward()
       optimizer.step()

      preds = to_numpy(student_logits)
      truths = to_numpy(targets)

      total_losses.append(loss.item())
      penultimate_losses.append(penultimate_loss.item())

      if layers_map is not None:
         intermediate_losses.append(intermediate_loss.item())


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

      log_metrics_distill(log_writer,
                     prefix="train",
                     loss=total_losses[batch_idx],
                     penultimate_loss=penultimate_losses[batch_idx],
                     intermediate_loss=intermediate_losses[batch_idx] if layers_map is not None else None,
                     recall=recalls[batch_idx],
                     accuracy=accuracies[batch_idx],
                     fpr=fprs[batch_idx],
                     fnr=fnrs[batch_idx],
                     idx=intermediate_step)

      if (batch_idx + 1) % save_frequency == 0:
         metrics = {"loss": np.mean(total_losses),
                    "accuracy": np.mean(accuracies),
                    "recall": np.mean(recalls),
                     "fpr": np.mean(fprs),
                     "fnr": np.mean(fnrs)}

         checkpoint_saver.save_checkpoint(student_model.state_dict(), epoch, batch_idx, metrics[eval_metric])

      pbar.set_description("Train: loss={}, recall={:.3f}, accuracy={:.3f}, fpr={:.3f}, fnr={:.3f}".format(
          total_losses[batch_idx], recalls[batch_idx], accuracies[batch_idx],
          fprs[batch_idx], fnrs[batch_idx]))

    pbar.close()
    validate_func(student_model, valid_dataloader, student_loss, device, log_writer, epoch, setup_config)
  state_holder.load_state_dict(student_model.state_dict(), strict=False)
  return copy.deepcopy(state_holder.cpu())





