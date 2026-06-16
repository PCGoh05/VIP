import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import IMAGE_SIZE, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from gradcam import create_gradcam_overlay, make_gradcam_heatmap  # noqa: E402


DISCLAIMER = (
    "This prototype is for educational and preliminary screening purposes only "
    "and does not replace expert agricultural diagnosis."
)


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(str(model_path))


@st.cache_data
def load_model_comparison():
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        return pd.read_csv(comparison_path)
    return pd.DataFrame()


def load_class_names():
    class_file = MODELS_DIR / "class_names.json"
    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))
    st.error("Class names file not found. Please train or evaluate the model first.")
    st.stop()


def make_class_label(class_name):
    label = class_name.replace("___", " - ")
    label = label.replace("_", " ")
    label = label.replace("(maize)", "(maize)")
    return label


def split_crop_and_condition(class_name):
    if "___" not in class_name:
        return "Unknown crop", make_class_label(class_name)
    crop, condition = class_name.split("___", 1)
    return crop.replace("_", " "), condition.replace("_", " ")


def get_available_models():
    candidates = {
        "MobileNetV2 fine-tuned": MODELS_DIR / "mobilenetv2_finetuned.keras",
        "MobileNetV2": MODELS_DIR / "mobilenetv2.keras",
    }
    comparison = load_model_comparison()

    available = []
    for display_name, path in candidates.items():
        if not path.exists():
            continue

        model_key = path.stem
        accuracy = None
        macro_f1 = None
        inference_time = None
        if not comparison.empty and "model" in comparison:
            row = comparison[comparison["model"] == model_key]
            if not row.empty:
                accuracy = float(row.iloc[0]["accuracy"])
                macro_f1 = float(row.iloc[0]["macro_f1_score"])
                inference_time = float(row.iloc[0]["avg_inference_time_seconds"])

        available.append({
            "display_name": display_name,
            "path": path,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "inference_time": inference_time,
        })

    return available


def get_default_model_index(available_models):
    if not available_models:
        return 0

    available_with_scores = [
        (index, item)
        for index, item in enumerate(available_models)
        if item["accuracy"] is not None
    ]
    if available_with_scores:
        best_index, _ = max(available_with_scores, key=lambda pair: pair[1]["accuracy"])
        return best_index
    return 0


@st.cache_data
def load_sample_images():
    test_dir = PROCESSED_DIR / "test"
    if not test_dir.exists():
        return []

    samples = []
    for class_dir in sorted([path for path in test_dir.iterdir() if path.is_dir()]):
        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        for image_path in image_paths[:3]:
            samples.append({
                "label": f"{make_class_label(class_dir.name)} / {image_path.name}",
                "path": str(image_path),
                "true_class": class_dir.name,
            })
    return samples


def prepare_image(image_source):
    image = Image.open(image_source).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized).astype("float32")
    image_batch = np.expand_dims(image_array, axis=0)
    return image, image_array, image_batch


def confidence_message(confidence):
    if confidence >= 0.80:
        return "High confidence", "normal"
    if confidence >= 0.55:
        return "Medium confidence. Please inspect the Grad-CAM carefully.", "warning"
    return "Low confidence. This image may be unclear or outside the trained dataset style.", "error"


def render_prediction_panel(predicted_class, confidence, inference_time):
    crop, condition = split_crop_and_condition(predicted_class)
    message, level = confidence_message(confidence)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction result")
    st.metric("Predicted class", make_class_label(predicted_class))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Crop", crop)
    col_b.metric("Condition", condition)
    col_c.metric("Confidence", f"{confidence:.2%}")
    st.metric("Inference time", f"{inference_time:.4f} seconds")

    if level == "normal":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_metrics(selected_model):
    cols = st.columns(3)
    if selected_model["accuracy"] is not None:
        cols[0].metric("Test accuracy", f"{selected_model['accuracy']:.2%}")
    else:
        cols[0].metric("Test accuracy", "Not evaluated")

    if selected_model["macro_f1"] is not None:
        cols[1].metric("Macro F1-score", f"{selected_model['macro_f1']:.2%}")
    else:
        cols[1].metric("Macro F1-score", "Not evaluated")

    if selected_model["inference_time"] is not None:
        cols[2].metric("Avg test inference", f"{selected_model['inference_time']:.4f} s")
    else:
        cols[2].metric("Avg test inference", "Not evaluated")


def render_top_predictions(probabilities, class_names):
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_three = pd.DataFrame(
        {
            "Class": [make_class_label(class_names[i]) for i in top_indices],
            "Confidence": [float(probabilities[i]) for i in top_indices],
        }
    )

    st.subheader("Top three predictions")
    st.bar_chart(top_three.set_index("Class"))
    table = top_three.copy()
    table["Confidence"] = table["Confidence"].map(lambda value: f"{value:.2%}")
    st.dataframe(table, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Plant Disease Recognition", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; max-width: 1180px; }
        .hero {
            border: 1px solid #dfe7df;
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            background: #f7fbf7;
            margin-bottom: 1rem;
        }
        .hero h1 { margin-bottom: 0.25rem; }
        .panel {
            border: 1px solid #e2e8e2;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
        }
        .small-note { color: #5f6f5f; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>AI-Powered Plant Disease Recognition Assistant</h1>
            <p class="small-note">Upload one tomato, potato or corn leaf image to classify disease symptoms and inspect the Grad-CAM visual explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_models = get_available_models()
    if not available_models:
        st.warning("No trained MobileNetV2 model found. Run src/train_mobilenetv2.py first.")
        st.stop()

    model_names = [item["display_name"] for item in available_models]
    selected_model = available_models[get_default_model_index(available_models)]
    model = load_model(selected_model["path"])
    class_names = load_class_names()
    comparison = load_model_comparison()
    sample_images = load_sample_images()

    with st.sidebar:
        st.header("Demo settings")
        selected_name = st.selectbox(
            "Model",
            model_names,
            index=get_default_model_index(available_models),
        )
        selected_model = available_models[model_names.index(selected_name)]
        model = load_model(selected_model["path"])

        if selected_model["accuracy"] is not None:
            st.caption(f"Final test accuracy: {selected_model['accuracy']:.2%}")
        if selected_model["macro_f1"] is not None:
            st.caption(f"Macro F1-score: {selected_model['macro_f1']:.2%}")

        st.divider()
        st.subheader("Final test sample")
        sample_labels = ["No sample selected"] + [item["label"] for item in sample_images]
        sample_label = st.selectbox("Optional sample image", sample_labels)
        selected_sample = None
        if sample_label != "No sample selected":
            selected_sample = sample_images[sample_labels.index(sample_label) - 1]

        st.divider()
        st.subheader("Accepted classes")
        for class_name in class_names:
            st.caption(make_class_label(class_name))
        st.divider()
        st.warning(DISCLAIMER)

    render_model_metrics(selected_model)

    uploaded_file = st.file_uploader("Upload one leaf image", type=["jpg", "jpeg", "png"])
    true_class = None
    image_source = None
    source_title = "Uploaded image"

    if uploaded_file is not None:
        image_source = uploaded_file
    elif selected_sample is not None:
        image_source = selected_sample["path"]
        true_class = selected_sample["true_class"]
        source_title = "Selected final test sample"

    if image_source is None:
        st.info("Upload a leaf image or choose a final test sample from the sidebar.")
        if not comparison.empty:
            st.subheader("Model comparison")
            display = comparison.copy()
            numeric_columns = [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1_score",
                "weighted_f1_score",
            ]
            for column in numeric_columns:
                if column in display:
                    display[column] = display[column].map(lambda value: f"{value:.2%}")
            if "model_size_mb" in display:
                display["model_size_mb"] = display["model_size_mb"].map(lambda value: f"{value:.2f} MB")
            if "avg_inference_time_seconds" in display:
                display["avg_inference_time_seconds"] = display["avg_inference_time_seconds"].map(
                    lambda value: f"{value:.4f} s"
                )
            st.dataframe(display, use_container_width=True, hide_index=True)
        return

    original_image, image_array, image_batch = prepare_image(image_source)

    start = time.perf_counter()
    probabilities = model.predict(image_batch, verbose=0)[0]
    inference_time = time.perf_counter() - start

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.subheader(source_title)
        st.image(original_image, use_container_width=True)
        if true_class is not None:
            st.metric("Ground truth", make_class_label(true_class))

    with right:
        render_prediction_panel(predicted_class, confidence, inference_time)
        if true_class is not None:
            if predicted_class == true_class:
                st.success("Prediction matches the ground-truth label.")
            else:
                st.error("Prediction does not match the ground-truth label.")
        render_top_predictions(probabilities, class_names)

    st.subheader("Grad-CAM visual explanation")
    heatmap = make_gradcam_heatmap(model, image_batch, predicted_index)
    overlay = create_gradcam_overlay(image_array, heatmap)
    col_original, col_overlay = st.columns(2)
    col_original.image(original_image, caption="Original image", use_container_width=True)
    col_overlay.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
    st.caption(
        "For qualitative analysis, check whether the bright heatmap area is on visible leaf symptom regions rather than the background."
    )


if __name__ == "__main__":
    main()
