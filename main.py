import tensorflow as tf
from dataset import get_datasets
from model import build_resnet

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
STAGE_BLOCKS = [3, 3, 3]
STAGE_FILTERS = [16, 32, 64]
INITIAL_LR = 0.1
MIN_LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

def main():
    # 1. Get Datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(batch_size=BATCH_SIZE)

    # 2. Build Model
    print("Building model...")
    model = build_resnet(
        input_shape=(32, 32, 3), 
        num_classes=10, 
        stage_blocks=STAGE_BLOCKS, 
        stage_filters=STAGE_FILTERS,
        weight_decay=WEIGHT_DECAY
    )
    
    # 3. Learning Rate Scheduler
    # Calculate decay steps based on training data size (45,000 samples)
    steps_per_epoch = 45000 // BATCH_SIZE
    decay_steps = steps_per_epoch * EPOCHS
    
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=decay_steps,
        alpha=MIN_LR / INITIAL_LR  # Ensures the lowest LR is MIN_LR
    )

    # 4. Optimizer and Compilation
    # Note: Weight decay is handled via kernel_regularizer in the model layers
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_scheduler,
        momentum=MOMENTUM
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_resnet_weights.weights.h5',
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # 6. Training
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb]
    )

    # 7. Evaluation
    print("Evaluating on test set...")
    # Load the best weights before testing
    model.load_weights('best_resnet_weights.weights.h5')
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
