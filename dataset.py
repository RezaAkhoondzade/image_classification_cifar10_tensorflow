import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def preprocess_train_data(image, label):
    # Cast to float32 (values are still 0-255)
    image = tf.cast(image, tf.float32)
    
    # 1. Spatial Augmentations
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    
    # 2. Color Augmentations (applied while values are 0-255)
    image = tf.image.random_brightness(image, max_delta=25.5)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # 3. Normalization to [-1, 1]
    image = (image / 127.5) - 1.0
    
    # 4. Clipping to ensure values stay within valid bounds
    image = tf.clip_by_value(image, -1.0, 1.0)
    
    return image, label

def preprocess_val_data(image, label):
    # Cast to float32
    image = tf.cast(image, tf.float32)
    
    # Normalization to [-1, 1]
    image = (image / 127.5) - 1.0
    
    # Clipping to ensure bounds
    image = tf.clip_by_value(image, -1.0, 1.0)
    
    return image, label

def get_datasets(batch_size=128):
    # Load data
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Split the last 5,000 samples for validation
    val_split_idx = -5000
    x_train, y_train = x_train_full[:val_split_idx], y_train_full[:val_split_idx]
    x_val, y_val = x_train_full[val_split_idx:], y_train_full[val_split_idx:]

    # Create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Build pipelines
    train_dataset = (train_dataset
                     .shuffle(buffer_size=10000)
                     .map(preprocess_train_data, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (val_dataset
                   .map(preprocess_val_data, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    test_dataset = (test_dataset
                    .map(preprocess_val_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

    return train_dataset, val_dataset, test_dataset
