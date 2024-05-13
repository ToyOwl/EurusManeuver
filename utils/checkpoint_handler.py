import os
import copy
import torch

def restore_from_checkpoint(path, model, optimizer, scheduler, device, silent=False):
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint["model_state"])

  if device.type != "cpu":
    model = model.cuda()

  optimizer.load_state_dict(checkpoint["optimizer_state"])
  scheduler.load_state_dict(checkpoint["scheduler_state"])

  if not silent:
   print(f"Training state restored from {path}")

  return checkpoint.get("epoch", 0), checkpoint.get("batch", 0)

def save_checkpoint(model, path, optimizer, scheduler, epoch, batch, score, state_save=True, silent=True):
   session_map = \
       {
        "epoch": epoch,
        "batch": batch,
        "score": score}

   if state_save:
      session_map["model_state"] = model.state_dict()
   else:
      session_map["model"] = model

   if optimizer is not None:
      session_map["optimizer_state"] = optimizer.state_dict()

   if scheduler is not None:
      session_map["scheduler_state"] = scheduler.state_dict()

   torch.save(session_map, path)
   if not silent:
     print(f"\n Model saved as {path}")

   return f"Model saved as {path}"

def load_model_state(model, path):
   with torch.no_grad():
     model.eval()
     checkpoint = torch.load(path, map_location=torch.device('cpu'))
     model.load_state_dict(checkpoint["model_state"], strict=False)
     print('Model state loaded from {}'.format(path))
     del checkpoint

   return model
class CheckpointSaver:

  def __init__(self,
               device,
               model,
               optimizer=None,
               save_state=True,
               scheduler=None,
               dir=None,
               is_loss=False,
               max_history=10,
               silent=True):

     self.device = device
     self.model = model
     self.optimizer = optimizer
     self.scheduler = scheduler
     self.state_saving = save_state


     self.ckpt_dir = dir or "chkpnts"

     if not os.path.exists(self.ckpt_dir):
       os.makedirs(self.ckpt_dir)

     self.checkpoints = {}
     self.is_loss = is_loss
     self.best_score = float("inf") if self.is_loss else float("-inf")
     self.model_name = self.model.__class__.__name__
     self.last_checkpoint = 0
     self.last_bestpoint = None
     self.max_history = max_history
     self.silent = silent

  def save_checkpoint(self, model_state, epoch, batch, score):

    self.model.load_state_dict(model_state, strict=False)

    if (self.is_loss and score < self.best_score) or (not self.is_loss and score > self.best_score):

       if self.last_bestpoint is not None and os.path.exists(self.last_bestpoint):
          os.remove(self.last_bestpoint)

       self.last_bestpoint = os.path.join(self.ckpt_dir, f"best_{self.model_name}_score_{score:.3f}.pth")

       save_checkpoint(self.model,
                       self.last_bestpoint,
                       self.optimizer,
                       self.scheduler,
                       epoch,
                       batch,
                       score,
                       self.state_saving,
                       self.silent)
       self.best_score = score

    else:
       self.last_checkpoint += 1
       self.checkpoints[self.last_checkpoint] = os.path.join( self.ckpt_dir,
                            f"{self.model_name}_epc_{epoch}_btch_{batch}_score_{score:.3f}.pth" )

       if len(self.checkpoints) >= self.max_history - 1:
          ckpt = self.checkpoints.pop(next(iter(self.checkpoints)))
          if os.path.exists(ckpt):
             os.remove(ckpt)

       save_checkpoint(self.model,
                       self.checkpoints[self.last_checkpoint],
                       self.optimizer,
                       self.scheduler,
                       epoch,
                       batch,
                       score,
                       self.silent)


