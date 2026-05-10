"""
model.py

Implementation of a customizable ResNet architecture for image classification.

Design principles:
Uses a standard residual block with two $3 \times 3$ convolutions.
Shortcut connections use $1 \times 1$ convolutions when spatial dimensions or channel counts change.
Batch normalization and ReLU activations are applied throughout the network.
L2 weight decay is consistently applied to all convolutional and dense layers.
Network structure is dynamically built using tuples for stage blocks and filter configurations.
"""

from keras import layers, models, regularizers


def residual_block(x, filters, stride=1, weight_decay=0e-4):
    """
    Constructs a standard residual block with two convolutional layers and a shortcut connection.
    If the spatial dimensions or number of filters change,
        a 1x1 convolution is applied to the shortcut path to match dimensions.

    Inputs:
        x: Input tensor. Shape: (B, H, W, C_in), Dtype: tf.float32
        filters: Number of filters for the convolutional layers. Dtype: int
        stride: Stride size for the first convolutional layer. Dtype: int
        weight_decay: L2 regularization factor. Dtype: float
    Outputs:
        output tensor: The output of the residual block.
            Shape: (B, H/stride, W/stride, filters), Dtype: tf.float32
    """
    shortcut = x

    # First convolution layer
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
        use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
        use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
            padding='same', use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut to the main path
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet(input_shape=(32, 32, 3), num_classes=10, stage_blocks=(3, 3, 3),
        stage_filters=(16, 32, 64), weight_decay=1e-4):
    """
    Builds a customizable ResNet model based on the provided stage configurations.
    The network begins with an initial convolution, followed by a sequence of residual stages,
    and concludes with global average pooling and a dense classification layer.

    Inputs:
        input_shape: Tuple defining the shape of input images. Dtype: tuple of ints
        num_classes: Number of output classes for the final dense layer. Dtype: int
        stage_blocks: Tuple for the number of residual blocks in each stage. Dtype: tuple of ints
        stage_filters: Tuple for the number of filters for each stage. Dtype: tuple of ints
        weight_decay: L2 regularization factor applied to all learnable weights. Dtype: float
    Outputs:
        model: Keras Model instance. Dtype: tf.keras.models.Model
    """
    # Input Layer and Initial Convolution
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=stage_filters[0], kernel_size=3, strides=1,
        padding='same', use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual Stages
    for stage_idx, (num_blocks, filters) in enumerate(zip(stage_blocks, stage_filters)):
        for block_idx in range(num_blocks):
            # Downsample on the first block of stages after the first one
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1
            x = residual_block(x, filters, stride=stride, weight_decay=weight_decay)

    # Global Average Pooling and Dense output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
        kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = models.Model(inputs, outputs)
    return model
