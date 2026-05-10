"""
dataset.py

Loading CIFAR-10 data and constructing optimized tf.data pipelines.

Design principles:
Separates data loading and pipeline configuration from preprocessing logic.
Utilizes an injected data generator for modular augmentation and normalization.
Implements efficient tf.data operations including parallel mapping and prefetching.
Calculates explicit dataset repeats based on training requirements.
"""

import tensorflow as tf
from keras.datasets import cifar10


def load_and_config_datasets(batch_size, epochs, steps_per_epoch, data_generator):
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
