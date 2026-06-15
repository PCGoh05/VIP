import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from config import IMAGE_SIZE, MODELS_DIR  # noqa: E402
from gradcam import create_gradcam_overlay, make_gradcam_heatmap  # noqa: E402


DISCLAIMER = (
    "This prototype is for educational and preliminary screening purposes only "
    "and does not replace expert agricultural diagnosis."
)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "mobilenetv2.keras"
    return tf.keras.models.load_model(str(model_path))


def load_class_names():
    class_file = MODELS_DIR / "class_names.json"
    if class_file.exists():
        return json.loads(class_file.read_text(encoding="utf-8"))
    st.error("Class names file not found. Please train or evaluate the model first.")
    st.stop()


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized).astype("float32")
    image_batch = np.expand_dims(image_array, axis=0)
    return image, image_array, image_batch


def main():
    st.set_page_config(page_title="Plant Disease Recognition", layout="centered")
    st.title("Plant Disease Recognition Assistant")
    st.caption(DISCLAIMER)

    model_path = MODELS_DIR / "mobilenetv2.keras"
    if not model_path.exists():
        st.warning("MobileNetV2 model not found. Run src/train_mobilenetv2.py first.")
        st.stop()

    model = load_model()
    class_names = load_class_names()

    uploaded_file = st.file_uploader("Upload one leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    original_image, image_array, image_batch = prepare_image(uploaded_file)
    st.image(original_image, caption="Uploaded image", use_container_width=True)

    start = time.perf_counter()
    probabilities = model.predict(image_batch, verbose=0)[0]
    inference_time = time.perf_counter() - start

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction")
    st.write(f"Predicted class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2%}**")
    st.write(f"Inference time: **{inference_time:.4f} seconds**")

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_three = [
        {"Class": class_names[i], "Confidence": f"{probabilities[i]:.2%}"}
        for i in top_indices
    ]
    st.subheader("Top three predictions")
    st.table(top_three)

    st.subheader("Grad-CAM visual explanation")
    heatmap = make_gradcam_heatmap(model, image_batch, predicted_index)
    overlay = create_gradcam_overlay(image_array, heatmap)
    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)


if __name__ == "__main__":
    main()
