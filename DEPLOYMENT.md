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

The repository includes a very small set of packaged demo images in:

```text
app/sample_images/
```

These images let users test the deployed webpage immediately without downloading the dataset.

## What the deployed app can do

The deployed app can:

- accept one uploaded leaf image
- test packaged sample images directly from the sidebar
- predict the plant disease class
- show confidence and top three predictions
- show inference time
- show Grad-CAM overlay

The deployed app is for inference only. It is not intended to train the model or run the full dataset pipeline online.

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

## Running the full dataset pipeline

To train models or run full evaluation, team members should use a local computer, WSL2 GPU environment, Google Colab or Kaggle Notebook. They need to download the dataset zip or the shared `data/processed/` zip first.

The deployed Streamlit webpage is only for testing predictions and Grad-CAM from uploaded images or packaged samples.
