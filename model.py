from tensorflow.keras import layers, models, regularizers


def residual_block(x, filters, stride=1, weight_decay=0e-4):
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

def build_resnet(input_shape=(32, 32, 3), num_classes=10, stage_blocks=[3, 3, 3],
        stage_filters=[16, 32, 64], weight_decay=1e-4):

    # Initial Convolution
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
