from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, RAW_DIR, REPORTS_DIR
from utils import count_images_by_class, find_split_dir, is_image_readable, list_image_files, make_dir


def audit_split(split_name):
    split_dir = find_split_dir(RAW_DIR, split_name)
    if split_dir is None:
        print(f"Could not find '{split_name}' folder inside {RAW_DIR}")
        return pd.DataFrame(), []

    print(f"\nFound {split_name} folder: {split_dir}")
    counts = count_images_by_class(split_dir)
    rows = [
        {"split": split_name, "class_name": class_name, "image_count": count}
        for class_name, count in counts.items()
    ]

    unreadable = []
    for image_path in list_image_files(split_dir):
        if not is_image_readable(image_path):
            unreadable.append(str(image_path))

    print(f"Classes in {split_name}: {len(counts)}")
    print(f"Images in {split_name}: {sum(counts.values())}")
    print(f"Unreadable images in {split_name}: {len(unreadable)}")
    return pd.DataFrame(rows), unreadable


def save_distribution_chart(df):
    if df.empty:
        return

    chart_data = df.pivot(index="class_name", columns="split", values="image_count").fillna(0)
    ax = chart_data.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Class Distribution in Raw Dataset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    output_path = FIGURES_DIR / "raw_class_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved chart: {output_path}")


def main():
    make_dir(FIGURES_DIR)
    make_dir(REPORTS_DIR)

    train_df, train_bad = audit_split("train")
    valid_df, valid_bad = audit_split("valid")

    combined = pd.concat([train_df, valid_df], ignore_index=True)
    if combined.empty:
        print("\nNo dataset images found. Please place the extracted Kaggle dataset inside data/raw/.")
        return

    csv_path = REPORTS_DIR / "raw_class_distribution.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved class distribution CSV: {csv_path}")

    unreadable_path = REPORTS_DIR / "unreadable_files.txt"
    unreadable_files = train_bad + valid_bad
    unreadable_path.write_text("\n".join(unreadable_files), encoding="utf-8")
    print(f"Saved unreadable file list: {unreadable_path}")

    save_distribution_chart(combined)
    print("\nAvailable classes:")
    for class_name in sorted(combined["class_name"].unique()):
        print(f"- {class_name}")


if __name__ == "__main__":
    main()
