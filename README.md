# ResNet on CIFAR-10

A modular, production-ready implementation of a custom ResNet architecture trained on the CIFAR-10 dataset using TensorFlow 2. 

## Project Overview

This project implements a customizable Residual Network (ResNet) to classify $32 \times 32$ RGB images into $10$ classes. The codebase is fully modularized, featuring a custom `tf.data` pipeline with robust data augmentations, L2 weight decay regularized layers, and a Cosine Decay learning rate schedule.

## Features

* **Modular Architecture**: Code is split into logical components (`dataset.py`, `model.py`, `main.py`) for readability and reusability.
* **Custom ResNet Builder**: Easily configure the depth and width of the network using `STAGE_BLOCKS` and `STAGE_FILTERS`.
* **Advanced `tf.data` Pipeline**:
  * $45,000$ / $5,000$ train/validation split.
  * Spatial augmentations (random crop with padding, horizontal flip).
  * Color augmentations (random brightness and contrast) applied safely before normalization.
  * Inputs are normalized to the $[-1, 1]$ range.
* **Optimized Training**: Uses SGD with Momentum, an L2 Weight Decay of $0.0001$ applied via `kernel_regularizer`, and a Cosine Decay learning rate scheduler ($0.1$ down to $0.001$).
* **Model Checkpointing**: Automatically saves the best weights (`.weights.h5` format) based on validation accuracy.

## Project Structure
```text
├── dataset.py   # Data loading, splitting, augmentation, and tf.data pipelines
├── model.py     # Residual block and ResNet model builder functions
├── main.py      # Entry point: hyperparameters, training loop, and evaluation
└── README.md    # Project documentation

## Requirements

* Python $3.8+$
* TensorFlow $2.13+$ (Compatible with Keras 3 / TF 2.16+)

bash
pip install tensorflow

## Usage

To train the model from scratch and evaluate it on the test set, simply run the main script:

bash
python main.py

### Hyperparameters
You can easily modify the training hyperparameters at the top of `main.py`:
* `BATCH_SIZE = 128`
* `EPOCHS = 100`
* `STAGE_BLOCKS = [3, 3, 3]`
* `STAGE_FILTERS = [16, 32, 64]`
* `INITIAL_LR = 0.1`

## Results

During training, the model will evaluate itself against a $5,000$-sample validation set at the end of each epoch. The `ModelCheckpoint` callback saves the weights of the best performing epoch. After training concludes, the script loads these optimal weights and evaluates the model against the standard CIFAR-10 test set ($10,000$ samples).
