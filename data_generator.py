"""
data_generator.py

Preprocessing and augmenting CIFAR-10 data.

Design principles:
All augmentations are applied before normalization.
Images remain in the original 0-255 (float) pixel space during augmentation.
Normalization is applied as the final preprocessing step.
Color augmentations use SSD-style photometric distortion with randomized order.
Spatial augmentation uses random padding, random cropping and flipping.
Random cropping will preserve aspect ratio to a minimum/maximum ratio
Normalization supports scaling to [0, 1], [-1, 1], or by dataset statistics.
"""

import tensorflow as tf


# CIFAR-10 channel statistics computed from training set
CIFAR10_MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
CIFAR10_STD  = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)


class DataGenerator:
    """
    Constructs and applies a configurable pipeline of image preprocessing and data augmentations.
    Encapsulates spatial transformations (such as random padding, cropping, and resizing)
    and photometric distortions
    Finally create preprocess_train_data and preprocess_val_data to be mapped on `tf.data.Dataset`.
    """

    def __init__(self, augment_cfg):
        """
        Constructor

        Inputs:
            augment_cfg: Python dictionary including all configs for data preprocess
        """
        self.augment_cfg = augment_cfg

    def random_pad(self, image):
        """
        Applies random ratio-based padding to the input image.
        Uses reflection padding by default to avoid introducing edge artifacts.

        Inputs:
            image: Input image tensor. Shape: (H, W, C), Dtype: tf.float32
        Outputs:
            padded image: Padded tensor. Shape: (H_padd, W_padd, C), Dtype: tf.float32
        """
        shape = tf.shape(image)
        h = tf.cast(shape[0], tf.float32)
        w = tf.cast(shape[1], tf.float32)

        # Computer random pad_h and pad_w in both dimensions
        rmax = self.augment_cfg["pad_max_range"]
        ratio = tf.random.uniform([2, ], 0., rmax)
        pad_h = tf.cast(h * ratio[0], tf.int32)
        pad_w = tf.cast(w * ratio[1], tf.int32)

        # Compute random padding offsets
        pad_top = tf.random.uniform([], 0, pad_h + 1, dtype=tf.int32)
        pad_bottom = pad_h - pad_top
        pad_left = tf.random.uniform([], 0, pad_w + 1, dtype=tf.int32)
        pad_right = pad_w - pad_left

        paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

        padded_image = tf.cond(
            pred=tf.random.uniform(()) < 0.5,
            true_fn=lambda: tf.pad(image, paddings, mode="CONSTANT"),
            false_fn=lambda: tf.pad(image, paddings, mode="REFLECT")
        )
        return padded_image

    def random_crop_with_traget_ratio(self, image):
        """
        Applies random cropping similar to standard ImageNet training strategies.
        Calculates random target area and aspect ratio, converts them to width/height,
        and extracts the corresponding bounding box.

        Inputs:
            image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
        Outputs:
            cropped image: Tensor. Shape: (H_crop, W_crop, C), Dtype: tf.float32
        """
        shape = tf.shape(image)
        h = tf.cast(shape[0], tf.float32)
        w = tf.cast(shape[1], tf.float32)
        area = h * w

        area_min = self.augment_cfg["crop_min_range"]
        scale = tf.random.uniform([], area_min, 1.)
        target_area = tf.square(scale) * area

        # Determine aspect ratio
        ar_min, ar_max = self.augment_cfg["aspect_ratio_range"]
        aspect = tf.random.uniform([], ar_min, ar_max)

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

    def resize_image(self, image):
        """
        Resizes the image to the fixed target size defined in config.

        Inputs:
            image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
        Outputs:
            resized image: Tensor. Shape: (target_h, target_w, C), Dtype: tf.float32
        """
        method = self.augment_cfg["resize_method"]

        # Select interpolation method
        if method == "bilinear":
            method = tf.image.ResizeMethod.BILINEAR
        elif method == "bicubic":
            method = tf.image.ResizeMethod.BICUBIC
        else:
            method = tf.image.ResizeMethod.BILINEAR

        return tf.image.resize(image, self.augment_cfg["target_size"], method=method)

    def random_cutout(self, image):
        """
        Randomly zeros out a square patch in the image.
        Generates coordinate masks to create a boolean mask for the cutout region.

        Inputs:
            image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
        Outputs:
            modified image: Tensor. Shape: (H, W, C), Dtype: tf.float32
        """
        # Check probability skip
        if tf.random.uniform(()) >= self.augment_cfg["cutout_prob"]:
            return image

        h = tf.shape(image)[0]
        w = tf.shape(image)[1]

        # Choose cutout dimensions
        cutout_min = self.augment_cfg.get("cutout_min_size", 8)
        cutout_max = self.augment_cfg.get("cutout_max_size", 8)
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

    def random_photometric_distort(self, image):
        """
        Applies random photometric distortions (SSD-style).
        Randomizes the order of contrast application relative to saturation/hue
        to provide greater color diversity. Keeps values bounded in [0, 255].

        Inputs:
            image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
        Outputs:
            distorted image: Tensor. Shape: (H, W, C), Dtype: tf.float32
        """
        # This is the base range for tf.image color augmentations.
        image = tf.cast(image, tf.float32) / 255.0

        # Adjust brightness
        if tf.random.uniform(()) < self.augment_cfg["brightness_prob"]:
            image = tf.image.random_brightness(image, self.augment_cfg["brightness_delta"])

        # Determine contrast application order
        contrast_first = tf.random.uniform(()) < 0.5

        if contrast_first:
            if tf.random.uniform(()) < self.augment_cfg["contrast_prob"]:
                image = tf.image.random_contrast(image, self.augment_cfg["contrast_lower"],
                    self.augment_cfg["contrast_upper"])

        # Adjust saturation
        if tf.random.uniform(()) < self.augment_cfg["saturation_prob"]:
            image = tf.image.random_saturation(image, self.augment_cfg["saturation_lower"],
                self.augment_cfg["saturation_upper"])

        # Shift hue
        if tf.random.uniform(()) < self.augment_cfg["hue_prob"]:
            image = tf.image.random_hue(image, self.augment_cfg["hue_delta"])

        # Apply contrast if not applied earlier
        if not contrast_first:
            if tf.random.uniform(()) < self.augment_cfg["contrast_prob"]:
                image = tf.image.random_contrast(image, self.augment_cfg["contrast_lower"],
                    self.augment_cfg["contrast_upper"])

        # Bound pixel values
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def normalize_image(self, image):
        """
        Applies the final normalization/standardization step.
        Assumes input image is already scaled to the [0, 1] range.
        Modes include [0, 1] scaling, [-1, 1] scaling, or CIFAR-10 standardization.

        Inputs:
            image: Image tensor. Shape: (H, W, C), Dtype: tf.float32
        Outputs:
            normalized image: Tensor. Shape: (H, W, C), Dtype: tf.float32
        """
        mode = self.augment_cfg["normalization"]

        # Apply scaling logic
        if mode == "0_1":
            # Already in [0, 1], do nothing.
            return image
        elif mode == "minus1_1":
            image = image * 2.0 - 1.0
        elif mode == "cifar10":
            image = (image - CIFAR10_MEAN) / CIFAR10_STD
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        return image

    def unnormalize_image(self, image):
        """
        Reverses normalize_image to bring image back to [0, 1].
        """
        mode = self.augment_cfg["normalization"]
        if mode == "0_1":
            # Already [0, 1]
            return image
        elif mode == "minus1_1":
            # Inverse of (x * 2.0 - 1.0) is (x + 1.0) / 2.0
            return (image + 1.0) / 2.0
        elif mode == "cifar10":
            # Inverse of (x - mean) / std is (x * std) + mean
            return (image * CIFAR10_STD) + CIFAR10_MEAN

        return image

    def preprocess_train_data(self, image, label):
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
        # Apply spatial augmentations
        # tf.print("original", tf.shape(image))
        if tf.random.uniform(()) < self.augment_cfg["pad_crop_prob"]:
            if self.augment_cfg["pad_crop_style"] == "imagenet":
                image = tf.cast(image, tf.float32)
                image = self.random_pad(image)
                image = self.random_crop_with_traget_ratio(image)
                image = self.resize_image(image)
            elif self.augment_cfg["pad_crop_style"] == "cifar10":
                image = tf.image.resize_with_crop_or_pad(image, 40, 40)
                image = tf.image.random_crop(image, size=[32, 32, 3])

        image = self.random_cutout(image)

        # Handle random flipping
        if tf.random.uniform(()) < self.augment_cfg["flip_prob"]:
            image = tf.image.flip_left_right(image)

        # Apply color augmentations and normalize image
        image = self.random_photometric_distort(image)
        image = self.normalize_image(image)

        return image, label


    def preprocess_val_data(self, image, label):
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
        # Cast and scale to the [0, 1] range first
        image = tf.cast(image, tf.float32) / 255.0
        image = self.normalize_image(image)

        return image, label
