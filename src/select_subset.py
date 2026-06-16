import pandas as pd

from config import RAW_DIR, REPORTS_DIR, SELECTED_CLASSES, SELECTED_DIR
from utils import copy_class_folder, count_images_by_class, find_split_dir, make_dir


def copy_selected_classes(split_name):
    source_split = find_split_dir(RAW_DIR, split_name)
    if source_split is None:
        raise FileNotFoundError(f"Could not find '{split_name}' folder inside {RAW_DIR}")

    target_split = SELECTED_DIR / split_name
    make_dir(target_split)

    rows = []
    print(f"\nCopying selected classes for {split_name}:")
    for class_name in SELECTED_CLASSES:
        source_class = source_split / class_name
        target_class = target_split / class_name

        if not source_class.exists():
            print(f"Missing class, skipped: {class_name}")
            continue

        copy_class_folder(source_class, target_class)
        count = count_images_by_class(target_split).get(class_name, 0)
        rows.append({"split": split_name, "class_name": class_name, "image_count": count})
        print(f"- {class_name}: {count} images")

    return rows


def main():
    make_dir(SELECTED_DIR)
    make_dir(REPORTS_DIR)

    rows = []
    rows.extend(copy_selected_classes("train"))
    rows.extend(copy_selected_classes("valid"))

    summary = pd.DataFrame(rows)
    output_path = REPORTS_DIR / "selected_class_counts.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nSaved selected class summary: {output_path}")


if __name__ == "__main__":
    main()
