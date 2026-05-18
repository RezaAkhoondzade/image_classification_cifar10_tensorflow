"""
model.py

Implementation of a highly modular CNN architecture for image classification.

Design principles:
A Block Registry pattern dynamically selects structural building blocks.
Available blocks: ResNet (Basic, PreAct, Bottleneck), MobileNetV2 (Inverted Residual),
EfficientNet (MBConv), and ConvNeXt.
Squeeze-and-Excitation (SE) is integrated where applicable.
Shortcut connections use 1x1 convolutions when spatial dimensions or channel counts change.
Standard L2 regularization is applied consistently to learnable weights. Compatible with SGD
Network structure is built dynamically using arrays for repeats, filters, kernels, and strides.
"""

from keras import layers, models, regularizers


def get_activation(activation_name, default_name):
    """Helper to resolve the activation function."""
    if activation_name == "default" or activation_name is None:
        return default_name
    return activation_name


def _projection_shortcut(x, filters, stride, weight_decay, kernel_initializer):
    """Creates a 1x1 projection shortcut to match dimensions."""
    x = layers.Conv2D(filters, 1, stride, 'same', use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    return x


def resnet_basic(x, filters, kernel_size=3, stride=1, activation="default",
        weight_decay=1e-4, kernel_initializer="he_normal"):
    """Standard ResNet Basic Block."""
    act_fn = get_activation(activation, "relu")
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = _projection_shortcut(shortcut, filters, stride,
            weight_decay, kernel_initializer)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(act_fn)(x)
    return x


def preact_resnet(x, filters, kernel_size=3, stride=1, activation="default",
        weight_decay=1e-4, kernel_initializer="he_normal"):
    """Pre-Activation ResNet Block."""
    act_fn = get_activation(activation, "relu")
    shortcut = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
            padding='same', use_bias=False, kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = layers.Add()([x, shortcut])
    return x


def resnet_bottleneck(x, filters, kernel_size=3, stride=1, activation="default",
        weight_decay=1e-4, kernel_initializer="he_normal"):
    """ResNet Bottleneck Block with expansion factor of 4."""
    act_fn = get_activation(activation, "relu")
    shortcut = x
    expansion = 4
    expanded_filters = filters * expansion

    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    x = layers.Conv2D(expanded_filters, kernel_size=1, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != expanded_filters:
        shortcut = _projection_shortcut(shortcut, expanded_filters, stride,
            weight_decay, kernel_initializer)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(act_fn)(x)
    return x


def inverted_residual(x, filters, kernel_size=3, stride=1, activation="default",
        weight_decay=1e-4, kernel_initializer="he_normal"):
    """MobileNetV2 Inverted Residual Block with expansion factor of 6."""
    act_fn = get_activation(activation, "relu6")  # MobileNetV2 uses relu6
    shortcut = x
    expansion = 6
    expanded_filters = x.shape[-1] * expansion

    x = layers.Conv2D(expanded_filters, kernel_size=1, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same',
        use_bias=False, depthwise_initializer=kernel_initializer,
        depthwise_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and x.shape[-1] == filters:
        x = layers.Add()([x, shortcut])

    return x


def mbconv(x, filters, kernel_size=3, stride=1, activation="default",
        weight_decay=1e-4, kernel_initializer="he_normal"):
    """EfficientNet MBConv Block with SE mechanism."""
    act_fn = get_activation(activation, "swish")  # EfficientNet uses swish
    shortcut = x
    expansion = 6
    se_ratio = 0.25
    input_filters = shortcut.shape[-1]
    expanded_filters = input_filters * expansion

    x = layers.Conv2D(expanded_filters, kernel_size=1, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same',
        use_bias=False, depthwise_initializer=kernel_initializer,
        depthwise_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_fn)(x)

    # Squeeze-and-Excitation block. Activations are strictly relu and sigmoid
    reduced_filters = max(4, int(input_filters * se_ratio))
    se = layers.GlobalAveragePooling2D(keepdims=True)(x)
    se = layers.Conv2D(reduced_filters, kernel_size=1, activation='relu',
        use_bias=True, kernel_initializer=kernel_initializer)(se)
    se = layers.Conv2D(expanded_filters, kernel_size=1, activation='sigmoid',
        use_bias=True, kernel_initializer=kernel_initializer)(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
        use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and input_filters == filters:
        x = layers.Add()([x, shortcut])

    return x


def convnext(x, filters, kernel_size=5, stride=1, activation="default",
        weight_decay=1e-4, kernel_initializer="he_normal"):
    """ConvNeXt Block adapted for CIFAR."""
    act_fn = get_activation(activation, "gelu") # ConvNeXt uses gelu
    shortcut = x
    expansion = 4
    expanded_filters = filters * expansion

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same',
        use_bias=True, depthwise_initializer=kernel_initializer,
        depthwise_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.Conv2D(expanded_filters, kernel_size=1, strides=1, padding='same',
        use_bias=True, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Activation(act_fn)(x)

    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
        use_bias=True, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
            padding='same', use_bias=False, kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(weight_decay))(shortcut)
        shortcut = layers.LayerNormalization(epsilon=1e-6)(shortcut)

    x = layers.Add()([x, shortcut])
    return x


BLOCK_REGISTRY = {
    "resnet_basic": resnet_basic,
    "preact_resnet": preact_resnet,
    "resnet_bottleneck": resnet_bottleneck,
    "inverted_residual": inverted_residual,
    "mbconv": mbconv,
    "convnext": convnext
}


def build_model(model_cfg, input_shape=(32, 32, 3), num_classes=10):

    stem_conv_filters = model_cfg["stem_conv_filters"]
    stem_conv_stride = model_cfg["stem_conv_stride"]
    block_type = model_cfg["block_type"]
    repeats = model_cfg["repeats"]
    filters = model_cfg["filters"]
    kernels = model_cfg["kernels"]
    strides = model_cfg["strides"]
    activation = model_cfg["activation"]
    kernel_initializer = model_cfg["kernel_initializer"]
    top_conv_filters = model_cfg["top_conv_filters"]
    classifier_dropout_rate = model_cfg["classifier_dropout_rate"]
    weight_decay = model_cfg["weight_decay"]

    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"block_type '{block_type}' not found in registry.")

    block_fn = BLOCK_REGISTRY[block_type]

    # Resolve the stem activation based on the default block type used
    stem_act_fn = get_activation(activation, "relu")

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=stem_conv_filters, kernel_size=3, strides=stem_conv_stride,
        padding='same', use_bias=False, kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    if block_type != "preact_resnet":
        x = layers.BatchNormalization()(x)
        x = layers.Activation(stem_act_fn)(x)

    num_stages = len(repeats)
    for stage_idx in range(num_stages):
        stage_repeats = repeats[stage_idx]
        stage_filter = filters[stage_idx]
        stage_kernel = kernels[stage_idx]
        stage_stride = strides[stage_idx]

        for block_idx in range(stage_repeats):
            current_stride = stage_stride if block_idx == 0 else 1

            x = block_fn(x, filters=stage_filter, kernel_size=stage_kernel,
                stride=current_stride, activation=activation,
                weight_decay=weight_decay, kernel_initializer=kernel_initializer)

    if block_type == "mbconv" or block_type == "inverted_residual":
        x = layers.Conv2D(filters=top_conv_filters, kernel_size=3, strides=stem_conv_stride,
            padding='same', use_bias=False, kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(stem_act_fn)(x)

    if block_type == "preact_resnet":
        x = layers.BatchNormalization()(x)
        x = layers.Activation(stem_act_fn)(x)

    x = layers.GlobalAveragePooling2D()(x)
    if classifier_dropout_rate > 0.:
        x = layers.Dropout(classifier_dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = models.Model(inputs, outputs)
    return model
