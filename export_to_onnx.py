import argparse
import os

import torch
from torch import nn
import torch.onnx

import onnx


def convert_to_onnx(model_path, output_path, opset_version, dynamic_axes, constant_folding, export_params):
  try:
     model = torch.load(model_path)
     model.eval()

     dummy_input = torch.randn(1, 3, 224, 224)

     output_file = os.path.join(output_path, os.path.splitext(os.path.basename(model_path))[0]+ '.onnx')

     torch.onnx.export(model, dummy_input, output_file, verbose=True,opset_version=opset_version, input_names=['input'],
                       output_names=['output'],
                       dynamic_axes=dynamic_axes if dynamic_axes else None,
                       do_constant_folding=constant_folding,
                       export_params=export_params)
     print(f"Model converted successfully: {output_file}")

     onnx_model = onnx.load(output_file)
     onnx.checker.check_model(onnx_model)
     print(f"ONNX model verified: {output_file}")

  except Exception as e:
     print(f"Failed to convert model {model_path}: {str(e)}")

def main():
   parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX format.")
   parser.add_argument("--input_dir", type=str, help="Directory containing PyTorch models.")
   parser.add_argument("--output_dir", type=str, help="Directory where ONNX models will be saved.")
   parser.add_argument("--opset_version", type=int, default=17, help="Opset version to use for ONNX models.")
   parser.add_argument("--dynamic_axes", nargs='+',
                        help="Specify dynamic axes as key value pairs e.g., input:0,2 output:0,2")
   parser.add_argument("--constant_folding", action='store_true',
                        help="Enable constant folding optimization.")
   parser.add_argument("--export_params", action='store_true', default=False,
                        help="Export model parameters with the model.")

   args = parser.parse_args()

   if not os.path.exists(args.output_dir):
       os.makedirs(args.output_dir)

   dynamic_axes = {}
   if args.dynamic_axes:
     for pair in args.dynamic_axes:
         key, value = pair.split(':')
         dynamic_axes[key] = list(map(int, value.split(',')))


   for filename in os.listdir(args.input_dir):
      if filename.endswith('.pt') or filename.endswith('.pth'):
         model_path = os.path.join(args.input_dir, filename)
         convert_to_onnx(model_path, args.output_dir, args.opset_version, dynamic_axes,
                            args.constant_folding, args.export_params)

if __name__ == "__main__":
    main()
