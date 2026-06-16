# Dataset Setup for Team Members

This project does not store dataset images, trained models or large generated outputs in GitHub.

## Why the dataset is not in GitHub

The Kaggle plant disease dataset is large. Uploading it to GitHub would make the repository slow and may exceed GitHub file size limits. The repository should contain:

- source code
- notebooks
- README instructions
- lightweight result summaries

The repository should not contain:

- dataset images
- `.keras` model files
- large generated figures
- large generated CSV or log files

These files are ignored by `.gitignore`.

## Recommended team workflow

1. Keep the source code on GitHub.
2. Keep `VIP_Dataset.zip` in a shared Google Drive, OneDrive or the original Kaggle download page.
3. Each team member downloads the zip file to their own computer.
4. Each team member runs the same scripts to recreate `data/raw`, `data/selected` and `data/processed`.

This keeps the project reproducible without making the GitHub repository too large.

## Local dataset locations

The current local dataset zip on this computer is:

```text
C:\Users\Acer\Downloads\VIP_Dataset.zip
```

The latest project root is:

```text
C:\Users\Acer\Documents\VIP
```

After extraction and splitting, the dataset should appear in:

```text
data/raw/
data/selected/
data/processed/
```

## How to extract the selected classes

From the project root, run one of these commands.

Default path:

```powershell
python src/extract_selected_from_zip.py
```

Custom path:

```powershell
python src/extract_selected_from_zip.py --zip-path "D:\Datasets\VIP_Dataset.zip"
```

Environment variable:

```powershell
$env:VIP_DATASET_ZIP="D:\Datasets\VIP_Dataset.zip"
python src/extract_selected_from_zip.py
```

## How to recreate the processed dataset

Run:

```powershell
python src/dataset_audit.py
python src/select_subset.py
python src/prepare_splits.py
```

The final training, validation and test folders will be:

```text
data/processed/train/
data/processed/valid/
data/processed/test/
```

## What team members should see on GitHub

Team members should use GitHub to view:

- the full code pipeline in `src/`
- the Streamlit app in `app/`
- setup instructions in `README.md`
- dataset instructions in this file
- final result summary in `outputs/reports/final_results_summary.md`

They should not expect to see the actual dataset images on GitHub.

## Optional: sharing trained models

If the team wants to share only the selected dataset, package only:

```text
data/processed/
```

Do not package all three local folders (`data/raw`, `data/selected` and `data/processed`) because they contain duplicate copies of the selected images.

Current local sizes on this computer are approximately:

```text
data/raw/       270 MB
data/selected/  270 MB
data/processed/ 270 MB
```

Sharing all three would waste space. Sharing `data/processed/` is enough for teammates who want to train, evaluate or demo using the final split.

## Optional: sharing trained models

The final fine-tuned model is small enough for this coursework repository and is included for deployment:

```text
outputs/models/mobilenetv2_finetuned.keras
outputs/models/class_names.json
```

If the team trains a larger model later, upload the `.keras` model file to Google Drive or OneDrive and share the link in the group chat. Then each team member can place the downloaded model files in:

```text
outputs/models/
```

Do not commit very large model files to GitHub.
