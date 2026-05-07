"""
dataset.py

Loading and preprocessing CIFAR-10 data.

Design principles
-----------------
1. All augmentations are applied BEFORE normalization.
   This follows common practice in detection and classification pipelines
   (e.g., SSD, YOLO, ResNet CIFAR training).

2. Images remain in the original 0–255 pixel space during augmentation.
   Normalization is applied only as the final preprocessing step before
   feeding data to the model.

3. Color augmentations follow an SSD-style "photometric distortion"
   pipeline where brightness, contrast, saturation, and hue are applied
   with configurable probabilities. Contrast order is randomized
   (before or after HSV transforms) to increase diversity.

4. Spatial augmentation follows the classic CIFAR strategy:
   pad → random crop → optional horizontal flip.

5. Normalization supports three modes:
      "0_1"      → scale to [0, 1]
      "minus1_1" → scale to [-1, 1]
      "cifar10"  → dataset mean/std normalization
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10


# CIFAR-10 channel statistics computed from the training set
CIFAR10_MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
CIFAR10_STD  = tf.constant([0.2470, 0.2435, 0.2616], dtype=tf.float32)


# Augmentation configuration
AUG_CONFIG = {
    "flip_prob": 0.5,

    "brightness_prob": 0.5,
    "brightness_delta": 32.0,  # brightness change in 0–255 image space

    "contrast_prob": 0.5,
    "contrast_lower": 0.5,
    "contrast_upper": 1.5,

    "saturation_prob": 0.5,
    "saturation_lower": 0.5,
    "saturation_upper": 1.5,

    "hue_prob": 0.5,
    "hue_delta": 0.1,

    # Padding used for translation augmentation
    "pad_pixels": 4,

    # Cutout / Random Erasing
    "cutout_prob": 0.5,          # probability to apply cutout
    "cutout_min_size": 4,
    "cutout_max_size": 8,

    # Normalization strategy applied after augmentations
    # options: "0_1", "minus1_1", "cifar10"
    "normalization": "minus1_1"
}


def random_photometric_distort(image):
    """
    Apply SSD-style photometric distortions.

    The order loosely follows the SSD PhotometricDistort pipeline.
    Brightness is applied first, then contrast may occur either
    before or after HSV-based transforms (saturation and hue).
    This randomized ordering increases color diversity.

    Each transformation is applied with an independent probability
    defined in AUG_CONFIG.
    """

    # Random brightness adjustment
    if tf.random.uniform(()) < AUG_CONFIG["brightness_prob"]:
        image = tf.image.random_brightness(
            image,
            max_delta=AUG_CONFIG["brightness_delta"]
        )

    # Randomly decide whether contrast happens before or after HSV transforms
    contrast_first = tf.random.uniform(()) < 0.5

    if contrast_first:
        if tf.random.uniform(()) < AUG_CONFIG["contrast_prob"]:
            image = tf.image.random_contrast(
                image,
                AUG_CONFIG["contrast_lower"],
                AUG_CONFIG["contrast_upper"]
            )

    # Saturation adjustment (HSV space internally)
    if tf.random.uniform(()) < AUG_CONFIG["saturation_prob"]:
        image = tf.image.random_saturation(
            image,
            AUG_CONFIG["saturation_lower"],
            AUG_CONFIG["saturation_upper"]
        )

    # Hue shift
    if tf.random.uniform(()) < AUG_CONFIG["hue_prob"]:
        image = tf.image.random_hue(
            image,
            AUG_CONFIG["hue_delta"]
        )

    # Apply contrast after HSV transforms if it wasn't applied earlier
    if not contrast_first:
        if tf.random.uniform(()) < AUG_CONFIG["contrast_prob"]:
            image = tf.image.random_contrast(
                image,
                AUG_CONFIG["contrast_lower"],
                AUG_CONFIG["contrast_upper"]
            )

    # Clip to make sure there are not any outbound values
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image


def random_cutout(image):
    """
    Apply a Cutout-style augmentation: randomly zero out a square patch.

    Features:
      • Probability-controlled application
      • Random side length in [cutout_min_size, cutout_max_size]
      • Works in float [0,255] pixel space

    Args:
        image: Tensor [H, W, C], dtype float32, range [0,255]

    Returns:
        Tensor of same shape and dtype
    """

    # 1. Maybe skip entirely
    if tf.random.uniform(()) >= AUG_CONFIG["cutout_prob"]:
        return image

    # 2. Infer image size
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    # 3. Randomly choose a side length between min and max (inclusive)
    cutout_min = AUG_CONFIG.get("cutout_min_size", 8)
    cutout_max = AUG_CONFIG.get("cutout_max_size", 8)
    cutout_size = tf.random.uniform(
        shape=(),
        minval=cutout_min,
        maxval=cutout_max + 1,   # +1 because upper bound is exclusive
        dtype=tf.int32
    )

    # 4. Randomly choose the center coordinates (cy, cx)
    cy = tf.random.uniform((), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((), 0, w, dtype=tf.int32)

    # 5. Compute box bounds, clipped to the valid image region
    y1 = tf.clip_by_value(cy - cutout_size // 2, 0, h)
    y2 = tf.clip_by_value(cy + cutout_size // 2, 0, h)
    x1 = tf.clip_by_value(cx - cutout_size // 2, 0, w)
    x2 = tf.clip_by_value(cx + cutout_size // 2, 0, w)

    # 6. Build a per‑pixel mask that is 0 inside the box, 1 outside.
    #    We do this by comparing coordinate grids.
    #
    # yy → vertical coordinates of shape [h,1]
    # xx → horizontal coordinates of shape [1,w]
    #
    # inside_y[i] is True when y coordinate of row i is inside [y1, y2)
    # inside_x[j] is True when x coordinate of column j is inside [x1, x2)
    # The 2D inside mask is the logical AND across height and width.
    yy = tf.range(h)[:, None]       # shape [h,1]
    xx = tf.range(w)[None, :]       # shape [1,w]
    inside_y = (yy >= y1) & (yy < y2)
    inside_x = (xx >= x1) & (xx < x2)
    inside = inside_y & inside_x    # shape [h,w], True where the square lies

    # 7. Convert boolean mask to {0,1} float mask
    mask = tf.where(
        inside,
        tf.zeros_like(inside, dtype=image.dtype),
        tf.ones_like(inside, dtype=image.dtype)
    )

    # 8. Expand to [h,w,1] so it multiplies all 3 channels equally
    mask = tf.expand_dims(mask, axis=-1)

    # 9. Apply mask: multiply with image → zero rectangle area
    image = image * mask

    return image


def normalize_image(image):
    """
    Normalize image according to AUG_CONFIG.

    Supported modes:
        0_1      → scale to [0, 1]
        minus1_1 → scale to [-1, 1]
        cifar10  → mean/std normalization

    Clipping is only applied for bounded normalization modes.
    """

    mode = AUG_CONFIG["normalization"]

    if mode == "0_1":
        image = image / 255.0

    elif mode == "minus1_1":
        image = (image / 127.5) - 1.0

    elif mode == "cifar10":
        image = image / 255.0
        image = (image - CIFAR10_MEAN) / CIFAR10_STD

    else:
        raise ValueError(
            f"Unknown normalization mode: {mode}. "
            "Valid options: '0_1', 'minus1_1', 'cifar10'"
        )

    return image


def preprocess_train_data(image, label):
    """
    Training preprocessing pipeline.

    Steps:
        1. Cast to float32
        2. Spatial augmentation (pad → random crop → horizontal flip)
        3. Color augmentation (photometric distortions)
        4. Normalize
    """

    image = tf.cast(image, tf.float32)

    # Spatial augmentations
    # Padding allows random translation via cropping
    pad = AUG_CONFIG["pad_pixels"]

    image = tf.image.resize_with_crop_or_pad(
        image,
        32 + pad * 2,
        32 + pad * 2
    )

    # Random crop back to CIFAR size
    image = tf.image.random_crop(image, [32, 32, 3])

    # Horizontal flip
    if tf.random.uniform(()) < AUG_CONFIG["flip_prob"]:
        image = tf.image.flip_left_right(image)

    # TEST: Cutout before color jitter
    image = random_cutout(image)

    # Color augmentations
    image = random_photometric_distort(image)

    # Normalize for model input
    image = normalize_image(image)

    return image, label


def preprocess_val_data(image, label):
    """
    Validation/test preprocessing.

    Only normalization is applied. No augmentation.
    """

    image = tf.cast(image, tf.float32)
    image = normalize_image(image)

    return image, label


def get_datasets(batch_size=128):
    """
    Load CIFAR-10 and create tf.data pipelines.

    The original training set is split into:
        45,000 training samples
         5,000 validation samples
    """

    # Load dataset
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Split last 5k samples for validation
    val_split_idx = -5000
    x_train, y_train = x_train_full[:val_split_idx], y_train_full[:val_split_idx]
    x_val, y_val = x_train_full[val_split_idx:], y_train_full[val_split_idx:]

    # Training pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=batch_size * 128)
    train_dataset = train_dataset.map(preprocess_train_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.map(preprocess_val_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Test pipeline
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(preprocess_val_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
