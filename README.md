# Lightweight Vision Transformer (ViT)

## Overview
This project implements a **Lightweight Vision Transformer (ViT)** for image classification.  
It is designed to be efficient while maintaining strong accuracy, making it suitable for smaller datasets and resource-limited environments.  

The model introduces:
- **Local Window Attention**: captures fine-grained image details.
- **Global Window Attention**: integrates contextual relationships across patches.
- **Patch Embedding & Positional Encoding** for transformer input.
- **Custom Keras layers** for window partitioning, concatenation, and hybrid attention.
- **Data augmentation pipeline** with normalization, flips, rotations, and zooms.
- **Training callbacks** (checkpointing, early stopping, learning-rate scheduling).

### Key Features
- Lightweight yet effective ViT implementation.
- TensorFlow/Keras based and easily extensible.
- Configurable hyperparameters (image size, patch size, transformer depth, etc.).
- Evaluation metrics include **accuracy** and **top-5 accuracy**.
- Saves best weights automatically during training.

### Example Usage
```python
# Run experiment
lightweight_vit_history, lightweight_vit_model = run_experiment_with_lightweight_vit()
```

### Default Hyperparameters
- Image size: 32×32
- Patch size: 8
- Window size: 4
- Dim: 64, Heads: 6
- Transformer layers: 5
- Classes: 8

### Results
On CIFAR-like datasets, the lightweight ViT achieved:
- **Test Accuracy:** ~85%  
- **Top-5 Accuracy:** ~98%  
- Stable convergence with early stopping and learning rate scheduling.  
These results show that the model balances efficiency with competitive performance compared to standard ViTs.

---

## License
MIT License. Free to use and adapt.
