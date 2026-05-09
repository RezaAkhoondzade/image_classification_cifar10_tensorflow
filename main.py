import os
import json
import yaml
import shutil
import tensorflow as tf
from data_generator import DataGenerator
from dataset import load_and_config_datasets
from model import build_resnet


def main():
    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configs for easier access
    train_cfg = config['training']
    model_cfg = config['model']
    augment_cfg = config['augmentation']

    # Prepare checkpoint folder and resume variables
    resume = False
    initial_epoch = 0
    if not os.path.exists(train_cfg['checkpoint_dir']):
        os.mkdir(train_cfg['checkpoint_dir'])
    else:
        resume = True

    # Copy config file into checkpoint folder
    destination_path = os.path.join(train_cfg['checkpoint_dir'], 'config.yaml')
    shutil.copy2("config.yaml", destination_path)

    # Create data generator
    data_generator = DataGenerator(augment_cfg)

    # Get datasets and config tf.data.Dataset
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = load_and_config_datasets(
        batch_size=train_cfg["batch_size"], epochs=train_cfg["epochs"],
        steps_per_epoch=train_cfg["steps_per_epoch"], data_generator=data_generator)

    # Build model
    print("Building model...")
    model = build_resnet(
        input_shape=(32, 32, 3),
        num_classes=10,
        stage_blocks=model_cfg["stage_blocks"],
        stage_filters=model_cfg["stage_filters"],
        weight_decay=train_cfg["weight_decay"]
    )
    model.summary()

    # Learning rate scheduler
    decay_steps = train_cfg["steps_per_epoch"] * train_cfg["epochs"]
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=train_cfg["initial_lr"],
        decay_steps=decay_steps,
        alpha=train_cfg["min_lr"] / train_cfg["initial_lr"]  # Ensures the lowest LR is MIN_LR
    )

    # Optimizer and Compilation
    # Note: Weight decay is handled via kernel_regularizer in the model layers
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_scheduler,
        momentum=train_cfg["momentum"]
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    if resume:
        checkpoint_path = os.path.join(train_cfg['checkpoint_dir'], "last_checkpoint.keras")
        print("Loading existing model checkpoint...")
        model = tf.keras.models.load_model(checkpoint_path)
        initial_epoch = int(model.optimizer.iterations.numpy() // train_cfg["steps_per_epoch"])
        print(f"Continue training from epoch {initial_epoch}")

    # Callbacks
    checkpoint_cb_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(train_cfg['checkpoint_dir'], 'best_model.keras'),
        save_best_only=True,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            train_cfg['checkpoint_dir'],
            'last_checkpoint.keras'
        ),
        save_weights_only=False,
        save_freq='epoch'
    )

    # Training
    print("Starting training...")
    model.fit(
        train_dataset,
        epochs=train_cfg["epochs"],
        initial_epoch=initial_epoch,
        steps_per_epoch=train_cfg["steps_per_epoch"],
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, checkpoint_cb_best]
    )

    history_path = os.path.join(train_cfg['checkpoint_dir'], "history.json")
    with open(history_path, "w") as f:
        json.dump(model.history.history, f)

    # Evaluation
    print("Evaluating on test set...")
    # Load the best weights before testing
    model = tf.keras.models.load_model(os.path.join(train_cfg['checkpoint_dir'], 'best_model.keras'))
    _, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save evaluation results on test set
    evaluation_dict = {"test_accuracy": test_acc}
    evaluation_path = os.path.join(train_cfg['checkpoint_dir'], "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(evaluation_dict, f)


if __name__ == '__main__':
    main()
