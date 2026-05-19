# Modular CNN Framework for Image Classification

This repository provides a flexible and powerful framework for building, training, and experimenting with modern Convolutional Neural Networks (CNNs). Built with TensorFlow 2 and Keras, the project emphasizes modularity, configurability, and best practices in deep learning pipelines.

While developed and tested on the CIFAR-10 dataset, it is designed as a general-purpose testbed for image classification that can be easily extended to other datasets and computer vision tasks.

---

## Overview

This is more than just a single model implementation; it is a reusable framework designed for rapid prototyping and rigorous experimentation. The core philosophy is to separate concerns, allowing for independent modification of the data pipeline, model architecture, and training process through a central configuration file.

This project is ideal for practitioners, researchers, and students who want to:

- Understand and compare different modern CNN building blocks.
- Experiment with advanced data augmentation strategies.
- Build a robust, end-to-end training pipeline from scratch.
- Use a solid foundation for more advanced projects like object detection.

---

## Key Features

### 1. Centralized YAML Configuration

All aspects of the experiment—from the model's architecture to augmentation parameters and training hyperparameters—are controlled through a single, easy-to-read `config.yaml` file. This eliminates the need to modify the source code for experimentation and ensures perfect reproducibility.

```yaml
# Example snippet from config.yaml
model:
  block_type: "mbconv" # Options: resnet_basic, preact_resnet, mbconv, convnext...
  repeats: [3, 3, 3]
  filters: [24, 40, 80]
  strides: [1, 2, 2]

training:
  batch_size: 64
  epochs: 200
  initial_lr: 0.1

augmentation:
  pad_crop_style: "cifar10"
  cutout_prob: 0.5
```

### 2. Dynamic Model Architecture with a Block Registry

The `model.py` file implements a powerful "Block Registry" pattern, allowing you to dynamically construct a wide range of CNN architectures by simply changing a string in the config file.

**Supported Blocks:**

- `resnet_basic`: The original ResNet Basic Block.
- `preact_resnet`: The "Pre-Activation" ResNet variant.
- `resnet_bottleneck`: The efficient ResNet Bottleneck block.
- `inverted_residual`: The core block of MobileNetV2.
- `mbconv`: The advanced Mobile-inverted Bottleneck block from EfficientNet (with Squeeze-and-Excitation).
- `convnext`: A modern, ConvNet-based block inspired by Vision Transformers.

This design makes it trivial to benchmark different architectural choices without rewriting any model code.

### 3. Advanced and Composable Data Augmentation Pipeline

The `data_generator.py` provides a comprehensive suite of modern data augmentation techniques, applied in a carefully controlled sequence.

#### Spatial Augmentations

- **Random Padding & Cropping:** Supports both a standard CIFAR-style `(pad -> random crop)` and an ImageNet-style `(random_pad -> random_crop_with_aspect_ratio)` for greater flexibility.
- **Random Horizontal Flip**
- **Cutout:** Randomly erases a square region of the image to improve model regularization.

#### Photometric (Color) Augmentations

- Implements an **SSD-style photometric distortion** with randomized order of operations for greater color diversity.
- Includes random adjustments for **brightness, contrast, saturation, and hue**.

#### Flexible Normalization

- All augmentations are applied on the original `[0, 1]` float pixel scale, which is the recommended format for TensorFlow `tf.image` operations and ensures consistent behavior for photometric augmentations such as brightness, contrast, saturation, and hue adjustments.
- Normalization is the final step, with support for scaling to `[0, 1]`, `[-1, 1]`, or standardization using dataset-specific `mean` and `std`.

### 4. Efficient and Robust Training Pipeline

The framework incorporates standard best practices for training deep neural networks.

- **Efficient `tf.data` Pipeline:** Asynchronous dataset processing with `.map()`, `.batch()`, and `.prefetch()` for maximum GPU utilization.
- **Learning Rate Scheduling:** Implements a **Cosine Decay** schedule to smoothly anneal the learning rate from an initial high value to a minimum value.
- **Optimizer:** Uses SGD with Momentum, with weight decay applied directly as a kernel regularizer in all convolutional layers.

#### Comprehensive Callbacks

- `ModelCheckpoint`: Saves the best model based on validation accuracy *and* validation loss, plus the latest epoch's model.
- `CSVLogger`: Logs training history to a `.csv` file.
- `TensorBoard`: Logs metrics for visualization in TensorBoard.

#### Automatic Resumption

The training script automatically detects an existing checkpoint and resumes training from the last completed epoch.

---

## Project Structure

```text
.
├── checkpoints/             # Directory for saving models, logs, and configs
├── config.yaml              # Central configuration file for experiments
├── main.py                  # Main script to run the training and evaluation pipeline
├── model.py                 # Defines the modular CNN architectures and blocks
├── data_generator.py        # Implements all data augmentation and preprocessing functions
└── dataset.py               # Loads the CIFAR-10 dataset and builds the tf.data pipelines
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.15+

### Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. **Configure Your Experiment**  
   Open `config.yaml` and adjust the parameters under the `training`, `model`, and `augmentation` sections to define your experiment.

2. **Run Training**  
   Execute the main script from your terminal. You can optionally point to a different config file.

   ```bash
   python main.py --config config.yaml
   ```

   The script will create a directory specified by `checkpoint_dir` in the config, where all artifacts (model weights, logs, visualizations) will be saved.

3. **Monitor Training (Optional)**  
   You can visualize the training progress using TensorBoard.

   ```bash
   tensorboard --logdir checkpoints/your_experiment_name/tensorboard
   ```

---

## Results: A Proof of Concept

To validate the framework's effectiveness, a model was trained on CIFAR-10 using the **`resnet_basic`** block (from ResNet) with a stage repeat configuration of `[3, 3, 3]`.

Despite the model's modest size, the powerful augmentation pipeline and robust training setup enabled it to achieve **92.5% - 93.2% accuracy** on the CIFAR-10 test set. For comparison, the original ResNet paper reported **91.25% accuracy** on CIFAR-10, meaning the current framework achieves approximately **1.25% - 2.0% higher accuracy**, largely due to the stronger data augmentation pipeline and modern training practices integrated into the framework.

These initial results demonstrate the framework's capability to produce high-performing models while remaining highly modular and configurable. Future experiments with different block types, architecture configurations, augmentation strategies, and training setups will continue to be conducted, and the README will be updated with new benchmark results over time.

---

## Future Directions and Extensibility

This framework was intentionally designed not as a final product, but as a foundation. Its modular nature allows for easy extension:

- **New Datasets:** Replace the `load_and_config_datasets` function in `dataset.py` to work with any image classification dataset (e.g., ImageNet, custom data).
- **New Model Blocks:** Add a new block function to `model.py` and register it in the `BLOCK_REGISTRY` dictionary to make it available via the config file.
- **Object Detection:** The framework is designed with extensibility in mind and can evolve beyond image classification into a full object detection framework. The modular backbone generator in `model.py` can be adapted to support modern detection architectures such as RetinaNet, SSD, and EfficientDet. Additionally, the existing augmentation pipeline provides a strong foundation for detection tasks and can be extended with bounding box transformation logic for spatial augmentations such as cropping, flipping, and scaling.
- **New Optimizers:** The optimizer in `main.py` can be switched to Adam, AdamW, or others to experiment with different training dynamics.