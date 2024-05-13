import numpy as np
import time
import torch
import torch_pruning as tp
from torch.profiler import profile, record_function, ProfilerActivity

def measure_model_performance(model, device):

  inputs = torch.randn(1, 3, 224, 224).to(device)
  model.to(device)
  model.eval()

  for _ in range(10):
    _ = model(inputs)

  ops, params = tp.utils.count_ops_and_params(model, inputs)

  if device.type == 'cuda':

    with torch.no_grad():
      start_time = torch.cuda.Event(enable_timing=True)
      end_time = torch.cuda.Event(enable_timing=True)

      start_time.record()
      outputs = model(inputs)
      end_time.record()
      torch.cuda.synchronize()
      latency = start_time.elapsed_time(end_time)
  else:
      start_time = time.perf_counter()
      _ = model(inputs)
      latency = (time.perf_counter() - start_time) * 1000

  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
       with record_function("model_inference"):
            _ = model(inputs)

  flops = sum([entry.self_cpu_time_total for entry in prof.key_averages()]) * 1e-6
  if device.type == "cuda":
     ram_usage = torch.cuda.max_memory_allocated(device)/(1024*1024)
  else:
     ram_usage = np.nan


  return latency, flops, ram_usage, ops, params