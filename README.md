# ResNet on CIFAR‑10 (TensorFlow 2)

This project implements a small, modular ResNet‑style image classifier for the CIFAR‑10 dataset using TensorFlow 2.
The codebase is intended for learning, experimentation, and easy extension.

---

## Overview

The repository provides:

- A compact configurable ResNet architecture
- A modern `tf.data` pipeline with strong augmentations
- Multiple normalization strategies
- A clean training loop with callbacks, LR scheduling, and evaluation

This makes the project useful for students, researchers, and anyone practicing end‑to‑end image classification pipelines.

---

## Key Features

### 1. Modular Code Structure

- `dataset.py` – load cifar10 - config tf.data
- `data_generator.py` – augmentations, normalization
- `model.py` – residual blocks + ResNet builder
- `main.py` – training loop, optimizer, LR schedule, evaluation

### 2. Flexible CIFAR‑10 Augmentation Pipeline

The augmentation system (`dataset.py`) includes:

- Padding → random crop → horizontal flip
- Cutout (configurable size/probability)
- Photometric distortion:
  - random brightness
  - random contrast (before or after HSV transforms)
  - random saturation
  - random hue
- Normalization modes:
  - `"0_1"` — scale to [0,1]
  - `"minus1_1"` — scale to [-1,1]
  - `"cifar10"` — dataset mean/std normalization

All augmentations run in **0–255 pixel space** before normalization.

### 3. Configurable ResNet Model

The architecture is controlled by:

- `STAGE_BLOCKS`
- `STAGE_FILTERS`
- `TODO: BASIC_BLOCK (to be added)`

Default configuration:

```python
STAGE_BLOCKS = [3, 3, 3]
STAGE_FILTERS = [16, 32, 64]
```

This produces a ResNet‑20‑like model with ~0.28M parameters.

## 4. Training Setup

- SGD + Momentum
- L2 weight decay
- Cosine decay learning rate
- ModelCheckpoint saving best validation accuracy and last epoch
- Validation split: **45k train / 5k val / 10k test**

---

## Requirements

- Python 3.8+
- TensorFlow 2.13+ (compatible with Keras 3 / TF 2.16+)

### Install

```bash
pip install tensorflow keras pylint PyYaml
```

## Usage

Run training:

```bash
python main.py
```

Common hyperparameters (main.py):

```python
BATCH_SIZE = 128
EPOCHS = 100
STAGE_BLOCKS = [3, 3, 3]
STAGE_FILTERS = [16, 32, 64]
INITIAL_LR = 0.1
```

## Results

Using the default small ResNet configuration (`[3, 3, 3]`, ~0.28M params) with the augmentation pipeline in `dataset.py`:

Achieved **Test Accuracy: 93.26%** with new image classification general data augmentations like random crop and random padding

---

## Notes

This project focuses on clarity and experimentation rather than production engineering. It is ideal for:

- learning `tf.data` pipelines
- trying out different augmentation strategies
- building or extending small ResNet variants
- running CIFAR‑10 benchmarks
