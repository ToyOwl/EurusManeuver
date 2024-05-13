from copy import deepcopy
from itertools import islice

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.quantization.quantize_fx import convert_fx
from torch.quantization.quantize_fx import fuse_fx
from torch.quantization.quantize_fx import prepare_fx
from torch.quantization.quantize_fx import prepare_qat_fx

from utils.tensor_ops import normalize

QCONFIG_MAPPING = get_default_qat_qconfig_mapping("x86")

def fake_quantization(model, data_loader, n_batches, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
   prepared_model = deepcopy(model)
   prepared_model.eval()
   input_examples = []
   for image, _ in islice(data_loader, n_batches):
       input_examples.append(
           normalize(image, mean=mean, std=std))

   input_examples = torch.cat(input_examples, dim=0)
   prepared_model = prepare_qat_fx(model=prepared_model, qconfig_mapping=QCONFIG_MAPPING,
                                                                  example_inputs=(input_examples, ),)
   return prepared_model

def quantized_static(model, data_loader, n_batches, device, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
  prepared_model = deepcopy(model)
  prepared_model.eval()

  input_examples = []
  for image, _ in islice(data_loader, n_batches):
      input_examples.append(
          normalize(image, mean=mean, std=std))

  input_examples = torch.cat(input_examples, dim=0)
  prepared_model = prepare_fx(model=prepared_model, qconfig_mapping=QCONFIG_MAPPING, example_inputs=(input_examples, ),)
  device = torch.device(device)

  prepared_model.eval()
  prepared_model.to(device)

  with torch.no_grad():
      for image, _ in islice(data_loader, n_batches):
          prepared_model(normalize(image.to(device), mean=mean, std=std))
          prepared_model.cpu()

      quantized_model = convert_fx(prepared_model)
  return quantized_model

def create_int_model(model):
  int_model = deepcopy(model).to("cpu")
  int_model = convert_fx(int_model)
  return fuse_fx(int_model)

def load_quantized_model(modelfp32, int8dict, img_sz=(224, 224)):
   modelfp32.eval()
   fqat_model = prepare_qat_fx(model=prepared_model,
                                   config_mapping=QCONFIG_MAPPING,
                                   example_inputs=(torch.rand(1,3, *img_sz), ),)
   int_model = create_int_model(fqat_model)
   int_model.load_state_dict(int8dict, strict=False)
   return int_model