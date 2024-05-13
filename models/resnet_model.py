import torch
import torch.nn as nn
import timm

from models.towers import HintonTower, Tower

class ResNetHinton(nn.Module):

  def __init__(self, model_type="resnet18", n_subclass=5, extractor_out=512, towers_hidden=256) -> None:

    super(ResNetHinton, self).__init__()

    self.extractor_out = extractor_out
    self.towers_hidden = towers_hidden

    if not model_type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",]:
        raise RuntimeError("the model must belong to the ResNet family")

    self.backbone = timm.models.create_model(model_type, pretrained=True, num_classes=0, global_pool='avg')

    if self.backbone is None:
       raise RuntimeError("a feature extractor is None")

    self.collision = HintonTower(self.extractor_out, 2, self.towers_hidden, n_subclass)

  def forward(self, x)->torch.Tensor:
    features = self.backbone(x)
    out = self.collision(features)
    return out

class ResNetWrapper(nn.Module):

  def __init__(self, model_type="resnet18",  extractor_out=512, towers_hidden=256) -> None:

    super(ResNetWrapper, self).__init__()

    self.extractor_out = extractor_out
    self.towers_hidden = towers_hidden

    if not model_type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",]:
        raise RuntimeError("the model must belong to the ResNet family")

    self.backbone = timm.models.create_model(model_type, pretrained=True, num_classes=0, global_pool='avg')

    if self.backbone is None:
       raise RuntimeError("a feature extractor is None")

    self.collision = Tower(self.extractor_out, 1, self.towers_hidden)

  def forward(self, x) -> torch.Tensor:
    features = self.backbone(x)
    out = self.collision(features)
    return out


