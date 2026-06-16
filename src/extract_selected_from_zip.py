import shutil
import zipfile
from pathlib import Path, PurePosixPath

from config import RAW_DIR, SELECTED_CLASSES
from utils import make_dir


ZIP_PATH = Path(r"C:\Users\Acer\Downloads\VIP_Dataset.zip")


def get_entry_parts(entry_name):
    parts = PurePosixPath(entry_name).parts
    if "train" in parts:
        split_index = parts.index("train")
    elif "valid" in parts:
        split_index = parts.index("valid")
    else:
        return None

    if len(parts) <= split_index + 2:
        return None

    split_name = parts[split_index]
    class_name = parts[split_index + 1]
    file_name = parts[-1]
    return split_name, class_name, file_name


def reset_selected_raw_folders():
    for split_name in ["train", "valid"]:
        for class_name in SELECTED_CLASSES:
            target_dir = RAW_DIR / split_name / class_name
            if target_dir.exists():
                shutil.rmtree(target_dir)


def extract_selected_classes():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset zip not found: {ZIP_PATH}")

    selected = set(SELECTED_CLASSES)
    counts = {
        "train": {class_name: 0 for class_name in SELECTED_CLASSES},
        "valid": {class_name: 0 for class_name in SELECTED_CLASSES},
    }
    duplicate_entries = 0
    saved_paths = set()

    make_dir(RAW_DIR)
    reset_selected_raw_folders()

    with zipfile.ZipFile(ZIP_PATH) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue

            entry_parts = get_entry_parts(entry.filename)
            if entry_parts is None:
                continue

            split_name, class_name, file_name = entry_parts
            if split_name not in ["train", "valid"] or class_name not in selected:
                continue

            target_dir = RAW_DIR / split_name / class_name
            make_dir(target_dir)
            target_path = target_dir / file_name

            if target_path in saved_paths:
                duplicate_entries += 1
                continue

            with archive.open(entry) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

            saved_paths.add(target_path)
            counts[split_name][class_name] += 1

    return counts, duplicate_entries


def main():
    counts, duplicate_entries = extract_selected_classes()

    print("Selected dataset extraction complete.\n")
    for split_name in ["train", "valid"]:
        print(f"{split_name}:")
        for class_name, count in counts[split_name].items():
            print(f"- {class_name}: {count}")
        print()

    if duplicate_entries:
        print(f"Skipped duplicate zip entries: {duplicate_entries}")


if __name__ == "__main__":
    main()
