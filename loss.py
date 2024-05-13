import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

class HardMiningLoss(nn.Module):

  def __init__(self, loss_func, patience_epochs=3, hard_samples=5):
     super(HardMiningLoss, self).__init__()
     self.loss_func = loss_func
     self.patience_epochs = patience_epochs
     self.hard_samples = hard_samples

  def forward(self, predicts, targets, epoch):

    current_batch_size = predicts.size(0)

    patience_factor = (epoch - self.patience_epochs) / self.patience_epochs
    patience_factor = torch.clamp(patience_factor, min=0.0)
    effective_hard_samples = min(self.hard_samples, current_batch_size)  # Ensure it does not exceed batch size
    top_k = round(current_batch_size -
                      (current_batch_size - effective_hard_samples) * (1.0 - torch.exp(-1.0 * patience_factor)))


    loss = self.loss_func(predicts, targets)
    topk_losses, _ = torch.topk(loss, int(top_k))

    return topk_losses.mean()

class HintonLoss(nn.Module):

  def __init__(self, weight=None, reduction="mean", eps=1e-06):
    super(HintonLoss, self).__init__()
    self.weight = weight
    self.reduction = reduction
    self.eps = eps

  def forward(self, predicts, targets):
    clamped_predicts = torch.clamp(predicts, min=self.eps)
    log_predicts = torch.log(clamped_predicts)
    return f.nll_loss(log_predicts, targets, weight=self.weight, reduction=self.reduction)

class FitNet(nn.Module):

  def __init__(self, input_channels, output_channels, name="layer"):
     super(FitNet, self).__init__()
     self.name = name
     self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
     self.initialize()

  def forward(self, x):
     return self.conv(x)

  def initialize(self):
     nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

     if self.conv.bias is not None:
       nn.init.constant_(self.conv.bias, 0)

def kl_loss(input, target):
   return nn.KLDivLoss(reduction='batchmean')(input, target)

def penultimate_loss_func(student_logits, teacher_logits, temperature): #исправить на penutale layer
   student_soft = nn.functional.log_softmax(student_logits / temperature, dim=-1)
   teacher_soft = nn.functional.softmax(teacher_logits / temperature, dim=-1)
   return kl_loss(student_soft, teacher_soft) * (temperature ** 2)

def itermediate_layers_loss_impl(student_features, teacher_features, weights):
   intermediate_loss = 0
   for student, teacher, weight in zip(student_features, teacher_features,  weights):
      intermediate_loss += nn.functional.mse_loss(student, teacher) * weight
   return intermediate_loss

def itermediate_layer_proj_loss_impl(student_features, teacher_features, weights, projections, msk):
    if msk is None:
       raise ValueError('mask values must be set')

    itermediate_loss = 0
    proj_idx = 0

    for idx in range(len(msk)):
      teacher_feat = teacher_features[idx]
      student_feat = student_features[idx]
      if not msk[idx]:
        intermediate_loss += nn.functional.mse_loss(student, teacher) * weights[idx]
        continue

      itermediate_loss += nn.functional.mse_loss(student_feat, projections[proj_idx](teacher_feat))*weights[idx]
      proj_idx +=1

    return itermediate_loss

def intermediate_layers_loss(student_features, teacher_features, weights, projections=None, msk=None):
   if projections is None:
      return itermediate_layers_loss_impl(student_features, teacher_features, weights)
   return itermediate_layer_proj_loss_impl(student_features, teacher_features, weights, projections, msk)



