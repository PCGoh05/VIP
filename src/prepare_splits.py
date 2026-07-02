import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DIR, REPORTS_DIR, SEED, SELECTED_DIR, VALID_TEST_SIZE
from utils import list_image_files, make_dir


def reset_processed_dir():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    make_dir(PROCESSED_DIR)


def copy_training_set():
    source_train = SELECTED_DIR / "train"
    target_train = PROCESSED_DIR / "train"
    if not source_train.exists():
        raise FileNotFoundError(f"Selected train folder not found: {source_train}")
    shutil.copytree(source_train, target_train)


def split_valid_into_valid_and_test():
    source_valid = SELECTED_DIR / "valid"
    if not source_valid.exists():
        raise FileNotFoundError(f"Selected valid folder not found: {source_valid}")

    image_paths = []
    labels = []
    for class_dir in sorted([p for p in source_valid.iterdir() if p.is_dir()]):
        for image_path in list_image_files(class_dir):
            image_paths.append(image_path)
            labels.append(class_dir.name)

    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=VALID_TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    for split_name, paths, split_labels in [
        ("valid", valid_paths, valid_labels),
        ("test", test_paths, test_labels),
    ]:
        for image_path, label in zip(paths, split_labels):
            target_dir = PROCESSED_DIR / split_name / label
            make_dir(target_dir)
            shutil.copy2(image_path, target_dir / image_path.name)


def build_summary():
    rows = []
    for split_name in ["train", "valid", "test"]:
        split_dir = PROCESSED_DIR / split_name
        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            rows.append({
                "split": split_name,
                "class_name": class_dir.name,
                "image_count": len(list_image_files(class_dir)),
            })
    return pd.DataFrame(rows)


def main():
    make_dir(REPORTS_DIR)
    reset_processed_dir()
    copy_training_set()
    split_valid_into_valid_and_test()

    summary = build_summary()
    output_path = REPORTS_DIR / "processed_split_summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary)
    print(f"\nSaved split summary: {output_path}")


if __name__ == "__main__":
    main()
