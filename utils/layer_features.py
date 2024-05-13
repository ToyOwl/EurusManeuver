import torch
import torch.nn as nn

from loss import FitNet

def register_layer_features(model, layer_name):

  layer_feat = []

  def register_layer_features(module, input, output):
      layer_feat.append(output)

  for name, submodule in model.named_modules():
      if name in layer_name:
         submodule.register_forward_hook(register_layer_features)

  return layer_feat

def register_intermediate_features(teacher, student, layer_mapping):
    teacher_features, student_features = [], []

    def get_teacher_features(module, input, output):
        teacher_features.append(output)

    def get_student_features(module, input, output):
       student_features.append(output)

    for teacher_name, student_name in layer_mapping.items():
        teacher_module = teacher
        for name in teacher_name.split('.'):
            teacher_module = getattr(teacher_module, name)
        teacher_module.register_forward_hook(get_teacher_features)

        student_module = student
        for name in student_name.split('.'):
            student_module = getattr(student_module, name)
        student_module.register_forward_hook(get_student_features)

    return teacher_features, student_features
def fit_net_projections(feature_mapping):
    fit_nets = []
    msk = [False]*len(feature_mapping)
    idx = -1
    for layer_name, (teacher_out, student_out) in feature_mapping.items():
       idx+=1
       if teacher_out == student_out:
          continue

       fit_nets.append(FitNet(teacher_out, student_out, layer_name))
       msk[idx] = True
    if not fit_nets:
      return None, msk

    return nn.ModuleList([*fit_nets]), msk
def register_hooks(model, layer_names):

    output_channels = {}

    def create_hook(name):

       def hook(module, input, output):
          output_channels[name] = output.shape[1]
       return hook

    for name, submodule in model.named_modules():
        if name in layer_names:
            submodule.register_forward_hook(create_hook(name))

    return output_channels
def get_out_layer_fetures_hooks(model, layer_names):
   out_channels = register_hooks(model, layer_names)
   dummy_input = torch.randn(1, 3, 224, 224)
   model.cpu()
   model(dummy_input)
   return [out_channels.get(layer_name, -1) for layer_name in layer_names]

def get_out_layer_features(model, layer_names):
    output_channels = {}

    def find_last_conv(module):
        last_conv = None
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                last_conv = child
        return last_conv

    def traverse_modules(full_name, module):
        submodule = module
        for name in full_name.split('.'):
            submodule = getattr(submodule, name)
        last_conv = find_last_conv(submodule)
        if last_conv is not None:
            output_channels[full_name] = last_conv.out_channels
        else:
            output_channels[full_name] = -1

    for layer_name in layer_names:
        traverse_modules(layer_name, model)

    return output_channels

def find_linear_layers(model, submodule_name):
  linear_layers = []

  def search_submodule(module, name):
      for child_name, child in module.named_children():
         if child_name == name:
            collect_linear_layers(child, linear_layers)
            break
         else:
            search_submodule(child, name)


  def collect_linear_layers(submodule, layers):

     for child in submodule.children():
        if isinstance(child, nn.Linear):
           layers.append(child)
        elif not list(child.children()):  # Only recurse if the child has further submodules
           continue
        else:
           collect_linear_layers(child, layers)

  search_submodule(model, submodule_name)
  return linear_layers
def get_layer_feats(teacher_model, student_model, layer_map, use_hooks=False):
    out_features = {}
    feat_func = get_out_layer_fetures_hooks if use_hooks else get_out_layer_features

    teacher_layers = []
    student_layers = []
    for tlayer, slayer in layer_map.items():
       teacher_layers.append(tlayer)
       student_layers.append(slayer)

    teacher_features = feat_func(teacher_model, teacher_layers)
    student_features = feat_func(student_model, student_layers)

    feature_mapping={}
    for teacher_layer, student_layer in zip(teacher_features, student_features):
        teacher_out, student_out =\
            teacher_features[teacher_layer], student_features[student_layer]
        if teacher_out == -1 or student_out == -1:
           continue
        out_features[teacher_layer] = student_layer
        feature_mapping[teacher_layer] = (teacher_out, student_out)

    return out_features, feature_mapping

