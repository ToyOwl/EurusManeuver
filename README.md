# URFU 2024 Master Thesis Source Code

![URFU Thesis Project Overview](imgs/title.png)

This library assists UAVs (Unmanned Aerial Vehicles) in navigating and operating autonomously in challenging environments. It processes visual data on onboard devices to effectively detect and avoid obstacles.

## Key Features

### Neural Network Training

- **Training Neural Network Models for Obstacle Detection**
  - `train_base_model.py` — Training base FP32 model.

### Optimization Methods

- **Various Methods for Optimizing Neural Network Models**
  - `optimization/distillation_training.py` — Knowledge distillation training for small network models.
  - `optimization/layerwise_distillation.py` — Layerwise distillation training.
  - `optimization/low_rank_optimization.py` — Canonical and Tucker tensor decomposition of convolutional layers.
  - `optimization/pruning.py` — Magnitude and Taylor feature maps pruning.
  - `optimization/quantization.py` — Quantization-Aware Training (QAT) model training.

## Results

![Title Picture](imgs/compression.png)

### Model Compression Using Tensor Decomposition

- **Base FP32 Model Compression**
  - Base FP32 Model: ![FP32 Model](imgs/layer-1-fp32.png)
  - Canonical Decomposition: ![CP Decomposition](imgs/layer-1-cp-fp32.png)
  - Tucker Decomposition: ![Tucker Decomposition](imgs/layer-1-tucker-fp32.png)

### Model Compression Using Feature Maps Pruning

- **Base FP32 Model with Taylor Pruning**
  - ![Taylor Pruning](imgs/layer-1-taylor-fp32.png)

### Inference Performance Metrics

- **MMACS vs Accuracy**
  - ![MMACS vs Accuracy Graph](reports/Latency_vs_Accuracy.png)

- **Parameters (M) vs Accuracy**
  - ![Params vs Accuracy Graph](reports/Params_vs_Accuracy.png)
