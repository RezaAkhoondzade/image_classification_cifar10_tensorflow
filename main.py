import os
import json
import tensorflow as tf
from dataset import get_datasets
from model import build_resnet

# Hyperparameters
CHECKPOINT_DIR = "checkpoints/resnet333_try7"
BATCH_SIZE = 64
EPOCHS = 200
STEPS_PER_EPOCH = 700
STAGE_BLOCKS = [3, 3, 3]
STAGE_FILTERS = [16, 32, 64]
INITIAL_LR = 0.1
MIN_LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001


def main():

    # Prepare checkpoint folder and resume variables
    resume = False
    initial_epoch = 0
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    else:
        resume = True

    # Get datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(batch_size=BATCH_SIZE,
        epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

    # Build model
    print("Building model...")
    model = build_resnet(
        input_shape=(32, 32, 3),
        num_classes=10,
        stage_blocks=STAGE_BLOCKS,
        stage_filters=STAGE_FILTERS,
        weight_decay=WEIGHT_DECAY
    )
    model.summary()

    # Learning rate scheduler
    decay_steps = STEPS_PER_EPOCH * EPOCHS
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=decay_steps,
        alpha=MIN_LR / INITIAL_LR  # Ensures the lowest LR is MIN_LR
    )

    # Optimizer and Compilation
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

    if resume:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.keras")
        print("Loading existing model checkpoint...")
        model = tf.keras.models.load_model(checkpoint_path)
        initial_epoch = int(model.optimizer.iterations.numpy() // STEPS_PER_EPOCH)
        print(f"Continue training from epoch {initial_epoch}")

    # Callbacks
    checkpoint_cb_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, 'best_model.keras'),
        save_best_only=True,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_DIR,
            'last_checkpoint.keras'
        ),
        save_weights_only=False,
        save_freq='epoch'
    )

    # Training
    print("Starting training...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, checkpoint_cb_best]
    )

    history_path = os.path.join(CHECKPOINT_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(model.history.history, f)

    # Evaluation
    print("Evaluating on test set...")
    # Load the best weights before testing
    model = tf.keras.models.load_model(os.path.join(CHECKPOINT_DIR, 'best_model.keras'))
    _, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save evaluation results on test set
    evaluation_dict = {"test_accuracy": test_acc}
    evaluation_path = os.path.join(CHECKPOINT_DIR, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation_dict, f)


if __name__ == '__main__':
    main()
