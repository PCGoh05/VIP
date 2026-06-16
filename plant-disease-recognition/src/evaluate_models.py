import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import BATCH_SIZE, FIGURES_DIR, IMAGE_SIZE, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from utils import make_dir


def load_test_dataset():
    test_dir = PROCESSED_DIR / "test"
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    return test_ds, test_ds.class_names


def collect_true_labels(test_ds):
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    return np.array(y_true)


def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_one_model(model_name, model_path, test_ds, y_true, class_names):
    print(f"\nEvaluating {model_name}")
    model = tf.keras.models.load_model(str(model_path))

    start = time.perf_counter()
    y_prob = model.predict(test_ds, verbose=0)
    elapsed = time.perf_counter() - start
    avg_inference_time = elapsed / len(y_true)

    y_pred = np.argmax(y_prob, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_path = REPORTS_DIR / f"{model_name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    cm_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, f"{model_name} Confusion Matrix", cm_path)

    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "weighted_f1_score": weighted_f1,
        "model_size_mb": model_size_mb,
        "avg_inference_time_seconds": avg_inference_time,
    }


def main():
    make_dir(FIGURES_DIR)
    make_dir(REPORTS_DIR)

    test_ds, class_names = load_test_dataset()
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    y_true = collect_true_labels(test_ds)

    models_to_evaluate = {
        "simple_cnn": MODELS_DIR / "simple_cnn.keras",
        "mobilenetv2": MODELS_DIR / "mobilenetv2.keras",
        "mobilenetv2_finetuned": MODELS_DIR / "mobilenetv2_finetuned.keras",
    }

    rows = []
    for model_name, model_path in models_to_evaluate.items():
        if not model_path.exists():
            print(f"Skipped {model_name}: model file not found at {model_path}")
            continue
        rows.append(evaluate_one_model(model_name, model_path, test_ds, y_true, class_names))

    comparison = pd.DataFrame(rows)
    output_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print("\nModel comparison:")
    print(comparison)
    print(f"\nSaved model comparison table: {output_path}")


if __name__ == "__main__":
    main()
