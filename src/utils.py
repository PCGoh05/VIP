import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_image_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def is_image_readable(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def find_split_dir(base_dir, split_name):
    """Find a train or valid folder even if the Kaggle zip has nested folders."""
    base_dir = Path(base_dir)
    direct = base_dir / split_name
    if direct.exists():
        return direct

    for path in base_dir.rglob(split_name):
        if path.is_dir():
            return path
    return None


def count_images_by_class(split_dir):
    split_dir = Path(split_dir)
    counts = {}
    if not split_dir.exists():
        return counts

    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        counts[class_dir.name] = len(list_image_files(class_dir))
    return counts


def copy_class_folder(source_dir, destination_dir):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def load_image_for_model(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def measure_prediction_time(model, images):
    start = time.perf_counter()
    predictions = model.predict(images, verbose=0)
    elapsed = time.perf_counter() - start
    return predictions, elapsed / len(images)
