"""
dataset.py

Loading and preprocessing CIFAR-10 data.

Design principles:
All augmentations are applied before normalization.
Images remain in the original 0-255 (float) pixel space during augmentation.
Normalization is applied as the final preprocessing step.
Color augmentations use SSD-style photometric distortion with randomized order.
Spatial augmentation uses random padding, random cropping with preserving aspect ratio, and flipping.
Normalization supports scaling to [0, 1], [-1, 1], or by dataset statistics.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10


# CIFAR-10 channel statistics computed from training set
CIFAR10_MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
CIFAR10_STD  = tf.constant([0.2470, 0.2435, 0.2616], dtype=tf.float32)


# Augmentation configuration
AUG_CONFIG = {
    "flip_prob": 0.5,                # Probability of horizontal flip
    "brightness_prob": 0.5,          # Probability to apply random brightness
    "brightness_delta": 32.0,        # Max delta for brightness adjustment
    "contrast_prob": 0.5,            # Probability to apply random contrast
    "contrast_lower": 0.5,           # Lower bound for contrast factor
    "contrast_upper": 1.5,           # Upper bound for contrast factor
    "saturation_prob": 0.5,          # Probability to apply random saturation
    "saturation_lower": 0.5,         # Lower bound for saturation factor
    "saturation_upper": 1.5,         # Upper bound for saturation factor
    "hue_prob": 0.5,                 # Probability to apply random hue
    "hue_delta": 0.1,                # Max delta for hue adjustment
    "pad_pixels": 4,                 # ENHANCEMENT: Fixed pixel pad option
    "target_size": (32, 32),         # Target output resolution
    "pad_prob": 0.5,                 # Probability to apply random padding
    "pad_ratio_range": (0.0, 0.10),  # Range for dynamic ratio padding
    "pad_mode": "REFLECT",           # Padding mode (e.g., REFLECT, CONSTANT)
    "crop_prob": 1.0,                # Probability to apply random crop
    "crop_area_range": (0.6, 1.0),   # Range for crop area relative to original
    "aspect_ratio_range": (0.75, 1.33), # Range for crop aspect ratio
    "use_aspect_ratio": True,        # Whether to use aspect ratio during crop
    "resize_method": "bilinear",     # Interpolation method for resize
    "cutout_prob": 0.5,              # Probability to apply cutout
    "cutout_min_size": 4,            # Minimum size of the cutout square
    "cutout_max_size": 8,            # Maximum size of the cutout square
    "normalization": "minus1_1"      # Mode: "0_1", "minus1_1", or "cifar10"
}


def random_pad(image):
    """
    Applies random ratio-based padding to the input image.
    Uses reflection padding by default to avoid introducing edge artifacts.
    
    Inputs: 
        image: Input image tensor. Shape: (H, W, C), Dtype: tf.float32
    Outputs: 
        padded image: Padded tensor. Shape: (H', W', C), Dtype: tf.float32
    """
    if tf.random.uniform(()) > AUG_CONFIG["pad_prob"]:
        return image

    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    rmin, rmax = AUG_CONFIG["pad_ratio_range"]
    ratio = tf.random.uniform([], rmin, rmax)

    pad_h = tf.cast(h * ratio, tf.int32)
    pad_w = tf.cast(w * ratio, tf.int32)

    # Compute random padding offsets
    pad_top = tf.random.uniform([], 0, pad_h + 1, dtype=tf.int32)
    pad_bottom = pad_h - pad_top

    pad_left = tf.random.uniform([], 0, pad_w + 1, dtype=tf.int32)
    pad_right = pad_w - pad_left

    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    mode = AUG_CONFIG["pad_mode"]
    return tf.pad(image, paddings, mode=mode)


def random_crop_like_imagenet(image):
    """
    Applies random cropping similar to standard ImageNet training strategies.
    Calculates random target area and aspect ratio, converts them to width/height, 
    and extracts the corresponding bounding box.
    
    Inputs: 
        image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
    Outputs: 
        cropped image: Tensor. Shape: (H_crop, W_crop, C), Dtype: tf.float32
    """
    if tf.random.uniform(()) > AUG_CONFIG["crop_prob"]:
        return image

    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    area = h * w

    area_min, area_max = AUG_CONFIG["crop_area_range"]
    target_area = tf.random.uniform([], area_min, area_max) * area

    # Determine aspect ratio
    if AUG_CONFIG["use_aspect_ratio"]:
        ar_min, ar_max = AUG_CONFIG["aspect_ratio_range"]
        aspect = tf.random.uniform([], ar_min, ar_max)
    else:
        aspect = w / h

    # Compute dimensions from area and aspect ratio
    crop_w = tf.sqrt(target_area * aspect)
    crop_h = tf.sqrt(target_area / aspect)

    crop_w = tf.cast(tf.minimum(crop_w, w), tf.int32)
    crop_h = tf.cast(tf.minimum(crop_h, h), tf.int32)

    # Calculate offsets for crop
    h = tf.cast(h, tf.int32)
    w = tf.cast(w, tf.int32)
    offset_h = tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)
    return image


def resize_image(image):
    """
    Resizes the image to the fixed target size defined in config.
    
    Inputs: 
        image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
    Outputs: 
        resized image: Tensor. Shape: (target_h, target_w, C), Dtype: tf.float32
    """
    method = AUG_CONFIG["resize_method"]
    
    # Select interpolation method
    if method == "bilinear":
        method = tf.image.ResizeMethod.BILINEAR
    elif method == "bicubic":
        method = tf.image.ResizeMethod.BICUBIC
    else:
        method = tf.image.ResizeMethod.BILINEAR

    return tf.image.resize(image, AUG_CONFIG["target_size"], method=method)


def random_photometric_distort(image):
    """
    Applies random photometric distortions (SSD-style).
    Randomizes the order of contrast application relative to saturation/hue 
    to provide greater color diversity. Keeps values bounded in [0, 255].
    
    Inputs: 
        image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
    Outputs: 
        distorted image: Tensor. Shape: (H, W, C), Dtype: tf.float32
    """
    # Adjust brightness
    if tf.random.uniform(()) < AUG_CONFIG["brightness_prob"]:
        image = tf.image.random_brightness(image, 
            max_delta=AUG_CONFIG["brightness_delta"])

    # Determine contrast application order
    contrast_first = tf.random.uniform(()) < 0.5

    if contrast_first:
        if tf.random.uniform(()) < AUG_CONFIG["contrast_prob"]:
            image = tf.image.random_contrast(image, AUG_CONFIG["contrast_lower"], 
                AUG_CONFIG["contrast_upper"])

    # Adjust saturation
    if tf.random.uniform(()) < AUG_CONFIG["saturation_prob"]:
        image = tf.image.random_saturation(image, AUG_CONFIG["saturation_lower"], 
            AUG_CONFIG["saturation_upper"])

    # Shift hue
    if tf.random.uniform(()) < AUG_CONFIG["hue_prob"]:
        image = tf.image.random_hue(image, AUG_CONFIG["hue_delta"])

    # Apply contrast if not applied earlier
    if not contrast_first:
        if tf.random.uniform(()) < AUG_CONFIG["contrast_prob"]:
            image = tf.image.random_contrast(image, AUG_CONFIG["contrast_lower"], 
                AUG_CONFIG["contrast_upper"])

    # Bound pixel values
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def random_cutout(image):
    """
    Randomly zeros out a square patch in the image.
    Generates coordinate masks to create a boolean mask for the cutout region.
    
    Inputs: 
        image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
    Outputs: 
        modified image: Tensor. Shape: (H, W, C), Dtype: tf.float32
    """
    # Check probability skip
    if tf.random.uniform(()) >= AUG_CONFIG["cutout_prob"]:
        return image

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    # Choose cutout dimensions
    cutout_min = AUG_CONFIG.get("cutout_min_size", 8)
    cutout_max = AUG_CONFIG.get("cutout_max_size", 8)
    cutout_size = tf.random.uniform(shape=(), minval=cutout_min, 
        maxval=cutout_max + 1, dtype=tf.int32)

    # Determine center coordinates
    cy = tf.random.uniform((), 0, h, dtype=tf.int32)
    cx = tf.random.uniform((), 0, w, dtype=tf.int32)

    # Calculate box bounds
    y1 = tf.clip_by_value(cy - cutout_size // 2, 0, h)
    y2 = tf.clip_by_value(cy + cutout_size // 2, 0, h)
    x1 = tf.clip_by_value(cx - cutout_size // 2, 0, w)
    x2 = tf.clip_by_value(cx + cutout_size // 2, 0, w)

    # Create coordinate masks
    yy = tf.range(h)[:, None]
    xx = tf.range(w)[None, :]
    inside_y = (yy >= y1) & (yy < y2)
    inside_x = (xx >= x1) & (xx < x2)
    inside = inside_y & inside_x

    # Generate float mask
    mask = tf.where(inside, tf.zeros_like(inside, dtype=image.dtype), 
        tf.ones_like(inside, dtype=image.dtype))

    # Expand and apply mask
    mask = tf.expand_dims(mask, axis=-1)
    image = image * mask

    return image


def normalize_image(image):
    """
    Normalizes the image based on the selected configuration mode.
    Modes include [0, 1] scaling, [-1, 1] scaling, or CIFAR-10 standardization.
    
    Inputs: 
        image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
    Outputs: 
        normalized image: Tensor. Shape: (H, W, C), Dtype: tf.float32
    """
    mode = AUG_CONFIG["normalization"]

    # Apply scaling logic
    if mode == "0_1":
        image = image / 255.0
    elif mode == "minus1_1":
        image = (image / 127.5) - 1.0
    elif mode == "cifar10":
        image = image / 255.0
        image = (image - CIFAR10_MEAN) / CIFAR10_STD
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return image


def preprocess_train_data(image, label):
    """
    Full preprocessing pipeline for training data.
    Applies spatial augmentations, geometric changes, photometric 
    distortions, and finally normalizes the image.
    
    Inputs: 
        image: Original tensor. Shape: (H, W, C), Dtype: tf.uint8
        label: Label tensor. Shape: (1,), Dtype: tf.int64
    Outputs: 
        processed image: Tensor. Shape: (target_h, target_w, C), Dtype: tf.float32
        label: Label tensor. Shape: (1,), Dtype: tf.int64
    """
    image = tf.cast(image, tf.float32)

    # Apply spatial augmentations
    image = random_pad(image)
    image = random_crop_like_imagenet(image)
    image = resize_image(image)
    image = random_cutout(image)

    # Handle random flipping
    if tf.random.uniform(()) < AUG_CONFIG["flip_prob"]:
        image = tf.image.flip_left_right(image)

    # Apply color augmentations
    image = random_photometric_distort(image)
    image = normalize_image(image)

    return image, label


def preprocess_val_data(image, label):
    """
    Preprocessing pipeline for validation and test data.
    Only applies type casting and normalization.
    
    Inputs: 
        image: Original tensor. Shape: (H, W, C), Dtype: tf.uint8
        label: Label tensor. Shape: (1,), Dtype: tf.int64
    Outputs: 
        normalized image: Tensor. Shape: (H, W, C), Dtype: tf.float32
        label: Label tensor. Shape: (1,), Dtype: tf.int64
    """
    image = tf.cast(image, tf.float32)
    image = normalize_image(image)

    return image, label


def get_datasets(batch_size=128):
    """
    Loads CIFAR-10 and constructs optimized tf.data pipelines.
    Splits out a validation set and applies parallel mapping and prefetching.
    
    Inputs: 
        batch_size: Integer size of batches. Default: 128
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

    # Setup training pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=batch_size * 128)
    train_dataset = train_dataset.map(preprocess_train_data, 
        num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Setup validation pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.map(preprocess_val_data, 
        num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Setup testing pipeline
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(preprocess_val_data, 
        num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
