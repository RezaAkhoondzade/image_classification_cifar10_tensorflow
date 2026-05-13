"""
dataset.py

Loading CIFAR-10 data and constructing optimized tf.data pipelines.

Design principles:
Separates data loading and pipeline configuration from preprocessing logic.
Utilizes an injected data generator for modular augmentation and normalization.
Implements efficient tf.data operations including parallel mapping and prefetching.
Calculates explicit dataset repeats based on training requirements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10


# CIFAR-10 Static Labels
CIFAR_10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# TODO: save cifar-10 data in numpy arrays and load them instead of using tf.datasets
# TODO: test that the order of color augmentations matter or not
def visualize_batch(dataset, data_generator, checkpoint_dir, num_batches=5,
        num_images=9, imshow=False):
    """
    Visualizes a specified number of image batches from a tf.data.Dataset.

    This function iterates through a given number of batches, un-normalizes
    the images using the provided data_generator, displays them in a grid,
    and adds their corresponding CIFAR-10 class labels as titles.
    The resulting visualizations are saved as PNG files in the specified
    checkpoint directory.

    Inputs:
        dataset: A tf.data.Dataset containing batches of images and labels.
                 Images are expected to be in a normalized float format,
                 and labels should be integer class indices.
        data_generator: An object (e.g., DataGenerator instance) that
                        provides an `unnormalize_image` method to convert
                        normalized images back to the $[0, 1]$ range for
                        visualization.
        checkpoint_dir: String, the path to the directory where the
                        visualization images will be saved.
        num_batches: Integer, the number of batches to visualize from the
                     dataset. Default is $3$.
        num_images: Integer, the maximum number of images to display per
                    batch visualization. Default is $9$.

    Outputs:
        None: The function saves image files to disk and optionally displays
              them using `plt.show()` (if uncommented).
    """
    dataset = iter(dataset.take(num_batches))
    for i in range(num_batches):
        images, labels = next(dataset)
        images = images.numpy()
        labels = labels.numpy()

        grid_size = int(np.ceil(np.sqrt(num_images)))
        plt.figure(figsize=(9, 9))

        for j in range(min(len(images), num_images)):
            # Un-normalize back to [0, 1] for matplotlib
            img = data_generator.unnormalize_image(images[j])

            plt.subplot(grid_size, grid_size, j + 1)
            plt.imshow(img)

            # Add label name as title
            label_idx = labels[j][0]
            plt.title(CIFAR_10_CLASSES[label_idx])

            plt.axis('off')

        plt.tight_layout()

        # Save the figure to the checkpoint directory
        save_path = os.path.join(checkpoint_dir, f"batch_visual_{i}.png")
        plt.savefig(save_path)
        if imshow:
            plt.show()


def load_and_config_datasets(batch_size, epochs, steps_per_epoch, data_generator, checkpoint_dir):
    """
    Loads CIFAR-10 and constructs optimized tf.data pipelines.
    Splits out a validation set and applies parallel mapping and prefetching.

    Inputs:
        batch_size: Integer size of batches.
        epochs: Integer number of training epochs.
        steps_per_epoch: Integer number of steps per epoch.
        data_generator: Instance of DataGenerator containing preprocessing methods.
    Outputs:
        train_dataset: tf.data.Dataset
        val_dataset: tf.data.Dataset
        test_dataset: tf.data.Dataset
    """
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Split validation subset
    val_split_idx = -5000
    x_train, y_train = x_train_full[:val_split_idx], y_train_full[:val_split_idx]
    x_val, y_val = x_train_full[val_split_idx:], y_train_full[val_split_idx:]

    # Calculate required repeats for train dataset
    total_samples_needed = epochs * steps_per_epoch * batch_size
    required_repeats = int(total_samples_needed / len(x_train)) + 1

    # Setup training pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.repeat(required_repeats)
    train_dataset = train_dataset.shuffle(buffer_size=batch_size * 64)
    train_dataset = train_dataset.map(data_generator.preprocess_train_data,
        num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Visualize and write the first N batches of train
    visualize_batch(train_dataset, data_generator, checkpoint_dir)

    # Setup validation pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.map(data_generator.preprocess_val_data,
        num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Setup testing pipeline
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(data_generator.preprocess_val_data,
        num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
