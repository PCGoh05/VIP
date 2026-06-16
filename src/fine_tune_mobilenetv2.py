import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

from config import BATCH_SIZE, FIGURES_DIR, IMAGE_SIZE, MODELS_DIR, PROCESSED_DIR, SEED
from utils import make_dir


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "valid",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), valid_ds.prefetch(autotune)


def find_mobilenet_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("MobileNetV2 base model was not found.")


def prepare_for_fine_tuning(model, trainable_layers):
    base_model = find_mobilenet_base(model)
    base_model.trainable = True

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Batch normalization layers are kept frozen because small fine-tuning runs
    # can make their moving statistics unstable.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Fine-tuned MobileNetV2 Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Fine-tuned MobileNetV2 Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    output_path = FIGURES_DIR / "mobilenetv2_finetuned_training_curves.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved fine-tuning curves: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the trained MobileNetV2 model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--trainable-layers", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    make_dir(MODELS_DIR)
    make_dir(FIGURES_DIR)

    source_model_path = MODELS_DIR / "mobilenetv2.keras"
    if not source_model_path.exists():
        raise FileNotFoundError(f"Base MobileNetV2 model not found: {source_model_path}")

    train_ds, valid_ds = load_datasets()
    model = tf.keras.models.load_model(str(source_model_path))
    model = prepare_for_fine_tuning(model, trainable_layers=args.trainable_layers)
    model.summary()

    output_model_path = MODELS_DIR / "mobilenetv2_finetuned.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    plot_history(history)
    print(f"Best fine-tuned MobileNetV2 model saved to: {output_model_path}")


if __name__ == "__main__":
    main()
