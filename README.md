# Lightweight Vision Transformer (ViT) with Local–Global Attention

## 📌 Overview
This repository contains an implementation of a **Lightweight Vision Transformer (ViT)** model for image classification.  
The model balances **accuracy** and **efficiency** by combining **local attention (within image windows)** and **global attention (across windows)**.  

It is designed to run on relatively small image datasets while maintaining high classification performance.

---

## ⚙️ Features
- **Custom attention layers**:
  - Local window-based attention
  - Global cross-window attention
- **Transformer-style encoder blocks** with skip connections & MLPs
- **Data augmentation pipeline** (flip, rotation, zoom, normalization)
- **AdamW optimizer** with weight decay
- **Training callbacks** for checkpointing, learning-rate scheduling, and early stopping
- **Evaluation with accuracy and top-5 accuracy**

---

## 📦 Requirements
Install the dependencies before running:

```bash
pip install tensorflow keras

🚀 Usage
1. Load Dataset

Replace the dataset loading with your preferred dataset (default is CIFAR-like 32×32 images):

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

2. Run Training

The notebook defines a training function:

lightweight_vit_history, lightweight_vit_model = run_experiment_with_lightweight_vit()


This will:

Build the Lightweight ViT

Train on your dataset

Save best weights to lightweight_vit_model.weights.h5

Evaluate performance on the test set

🧩 Model Architecture

Patches + Patch Encoder → Embed image patches

LightweightViTBlock(s):

Local Window Attention

Global Window Attention

Skip connections + Layer Norm

Feed-forward MLP

Classification Head:

Global Average Pooling

Dense layers with GELU

Final classification layer

📊 Results

During training, you can monitor:

Accuracy

Top-5 Accuracy

Validation curves (from history object)

Example output:

Test accuracy: 85.32%
Test top 5 accuracy: 98.47%

📂 Project Structure
.
├── l-vit.ipynb         # Main notebook
├── lightweight_vit_model.weights.h5  # Saved best model weights (after training)
└── README.md           # This file

🔧 Hyperparameters

Default setup inside run_experiment_with_lightweight_vit:

image_size = 32

patch_size = 8

window_size = 4

dim = 64

num_heads = 6

transformer_layers = 5

mlp_head_units = [2048, 1024]

num_classes = 8

These can be modified for different datasets.

📜 License

MIT License. Feel free to use and adapt.

🙌 Acknowledgments

Inspired by Vision Transformer (ViT) and Swin Transformer concepts, adapted for lightweight use-cases in TensorFlow/Keras.
