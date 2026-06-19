import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from config import FIGURES_DIR, IMAGE_SIZE, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from utils import list_image_files, make_dir


def load_class_names():
    class_file = MODELS_DIR / "class_names.json"
    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))
    return sorted([p.name for p in (PROCESSED_DIR / "train").iterdir() if p.is_dir()])


def select_gradcam_model():
    candidates = [
        MODELS_DIR / "mobilenetv2_finetuned.keras",
        MODELS_DIR / "mobilenetv2.keras",
    ]
    for model_path in candidates:
        if model_path.exists():
            return model_path
    raise FileNotFoundError("No MobileNetV2 model found for Grad-CAM.")


def read_image_array(path):
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(img).astype("float32")


def find_feature_layer(model):
    """Find the final feature map layer used for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) and len(layer.output.shape) == 4:
            return layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Could not find a 4D feature layer for Grad-CAM.")


def call_layer_for_gradcam(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def make_gradcam_heatmap(model, image_batch, class_index=None):
    feature_layer = find_feature_layer(model)
    feature_layer_index = model.layers.index(feature_layer)
    layers_before_feature = model.layers[:feature_layer_index]
    layers_after_feature = model.layers[feature_layer_index + 1:]

    with tf.GradientTape() as tape:
        x = image_batch
        for layer in layers_before_feature:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = call_layer_for_gradcam(layer, x)

        feature_maps = call_layer_for_gradcam(feature_layer, x)
        tape.watch(feature_maps)

        x = feature_maps
        for layer in layers_after_feature:
            x = call_layer_for_gradcam(layer, x)
        predictions = x

        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = feature_maps[0]

    heatmap = feature_maps @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def create_gradcam_overlay(image_array, heatmap, alpha=0.45):
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    original = np.uint8(image_array)
    overlay = cv2.addWeighted(original, 1 - alpha, color_heatmap, alpha, 0)
    return overlay


def save_gradcam_figure(image_array, overlay, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.uint8(image_array))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def collect_test_images():
    rows = []
    test_dir = PROCESSED_DIR / "test"
    for class_dir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
        for image_path in list_image_files(class_dir):
            rows.append({"path": image_path, "true_class": class_dir.name})
    return rows


def main():
    make_dir(FIGURES_DIR)
    make_dir(REPORTS_DIR)
    output_dir = FIGURES_DIR / "gradcam_examples"
    make_dir(output_dir)

    model_path = select_gradcam_model()
    print(f"Using model for Grad-CAM: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    class_names = load_class_names()

    correct_saved = 0
    incorrect_saved = 0
    rows = []

    for item in collect_test_images():
        image_array = read_image_array(item["path"])
        image_batch = np.expand_dims(image_array, axis=0)
        probabilities = model.predict(image_batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        is_correct = predicted_class == item["true_class"]

        should_save = (is_correct and correct_saved < 3) or ((not is_correct) and incorrect_saved < 3)
        if should_save:
            heatmap = make_gradcam_heatmap(model, image_batch, predicted_index)
            overlay = create_gradcam_overlay(image_array, heatmap)

            group = "correct" if is_correct else "incorrect"
            number = correct_saved + 1 if is_correct else incorrect_saved + 1
            output_path = output_dir / f"{group}_{number}.png"
            title = f"True: {item['true_class']} | Predicted: {predicted_class}"
            save_gradcam_figure(image_array, overlay, title, output_path)

            if is_correct:
                correct_saved += 1
            else:
                incorrect_saved += 1

            rows.append({
                "image_path": str(item["path"]),
                "true_class": item["true_class"],
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": is_correct,
                "gradcam_file": str(output_path),
                "visual_note": "Check whether the bright heatmap area is on leaf symptom regions instead of the background.",
            })

        if correct_saved >= 3 and incorrect_saved >= 3:
            break

    summary = pd.DataFrame(rows)
    summary_path = REPORTS_DIR / "gradcam_examples_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved Grad-CAM examples to: {output_dir}")
    print(f"Saved Grad-CAM summary: {summary_path}")


if __name__ == "__main__":
    main()
