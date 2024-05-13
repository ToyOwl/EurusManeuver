import torch
import torchvision.transforms.functional as tf
def to_numpy(tensor):
   return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def normalize(tensor, mean, std):
  tensor = tensor / 255.0
  tensor = tensor.permute((0, 3, 1, 2))
  tensor = tf.normalize(tensor, mean, std)
  return tensor
