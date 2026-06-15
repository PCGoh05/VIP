import argparse
import json

import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    BATCH_SIZE,
    FIGURES_DIR,
    IMAGE_SIZE,
    MODELS_DIR,
    PROCESSED_DIR,
    SEED,
    SIMPLE_CNN_EPOCHS,
)
from utils import make_dir


def load_datasets(max_train_batches=None, max_valid_batches=None):
    train_dir = PROCESSED_DIR / "train"
    valid_dir = PROCESSED_DIR / "valid"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    if max_train_batches is not None:
        train_ds = train_ds.take(max_train_batches)
    if max_valid_batches is not None:
        valid_ds = valid_ds.take(max_valid_batches)

    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    return train_ds, valid_ds, class_names


def build_simple_cnn(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="limited_augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="simple_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Simple CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Simple CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    output_path = FIGURES_DIR / output_name
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Simple CNN baseline model.")
    parser.add_argument("--epochs", type=int, default=SIMPLE_CNN_EPOCHS)
    parser.add_argument("--quick-test", action="store_true", help="Run a tiny training job for pipeline checking.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    make_dir(MODELS_DIR)
    make_dir(FIGURES_DIR)

    if args.quick_test:
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = args.max_train_batches or 4
        args.max_valid_batches = args.max_valid_batches or 2
        print("Running Simple CNN quick test. This is not a final model.")

    train_ds, valid_ds, class_names = load_datasets(
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
    )
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    model = build_simple_cnn(num_classes=len(class_names))
    model.summary()

    model_filename = "simple_cnn_quick.keras" if args.quick_test else "simple_cnn.keras"
    figure_filename = "simple_cnn_quick_training_curves.png" if args.quick_test else "simple_cnn_training_curves.png"
    model_path = MODELS_DIR / model_filename
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    plot_history(history, figure_filename)
    print(f"Best Simple CNN model saved to: {model_path}")


if __name__ == "__main__":
    main()
