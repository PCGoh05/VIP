# Streamlit Deployment Guide

This project can be deployed as a Streamlit web app so users can open a webpage without cloning the repository.

## What is included for deployment

The deployed app needs:

- source code in `app/` and `src/`
- `requirements.txt`
- `runtime.txt`
- `outputs/models/mobilenetv2_finetuned.keras`
- `outputs/models/class_names.json`
- `outputs/reports/model_comparison.csv`

The deployed app does not need the full dataset.

## What the deployed app can do

The deployed app can:

- accept one uploaded leaf image
- predict the plant disease class
- show confidence and top three predictions
- show inference time
- show Grad-CAM overlay

The deployed app may not show local final test sample choices unless the dataset is also available in the deployment environment. For public demo use, uploading one image is enough.

## Recommended deployment option

Use Streamlit Community Cloud:

1. Push the latest project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from GitHub.
4. Select this repository:

```text
PCGoh05/VIP
```

5. Select branch:

```text
master
```

6. Set the main file path:

```text
app/streamlit-app.py
```

7. Deploy the app.

## Dataset sharing

Do not commit the image dataset folders into the GitHub repository.

If the team wants to share only the selected 8-class dataset, package only one final folder:

```text
data/processed/
```

Do not package all three folders (`raw`, `selected`, `processed`) because they contain duplicate copies of the same selected images.

Recommended sharing methods:

- Google Drive or OneDrive shared link
- GitHub Release asset
- Kaggle dataset page plus project extraction script

Do not store the selected dataset directly in normal Git commits.
