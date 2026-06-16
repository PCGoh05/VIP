# AI-Powered Plant Disease Recognition Assistant

CDS6334 Visual Information Processing Final Project  
Track: Intelligent Visual Application Track

This is a simple student-level project for recognizing selected plant leaf diseases and showing Grad-CAM heatmaps as a visual explanation.

## Group Members

| Name | Role |
| --- | --- |
| Lee Yi Yang | Dataset lead |
| Goh Pei Chung | Baseline and evaluation lead |
| Saw Qi Rui | Transfer-learning lead |
| Jayy Wong Jun Vun | Explainability and application lead |

## Project Structure

```text
plant-disease-recognition/
├── data/
│   ├── raw/
│   ├── selected/
│   └── processed/
├── notebooks/
├── src/
├── app/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── requirements.txt
└── README.md
```

## Dataset

Use the Kaggle New Plant Diseases Dataset (Augmented).

Place the extracted dataset inside:

```text
data/raw/
```

The scripts will search inside `data/raw/` for the Kaggle `train` and `valid` folders, even if the zip file creates nested folders.

Example acceptable layout:

```text
data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/
data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/
```

The small Kaggle test or prediction folder is not used as the official test set.

## Selected Classes

The selected classes are listed in `src/config.py`:

```text
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Potato___Early_blight
Potato___Late_blight
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
```

If your downloaded dataset uses slightly different class names, edit `SELECTED_CLASSES` in `src/config.py`.

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install requirements:

```powershell
pip install -r requirements.txt
```

## GPU Training Note

This project can run on Windows CPU, but training is faster with an NVIDIA GPU through WSL2 Ubuntu.

On this laptop, native Windows TensorFlow can install successfully, but TensorFlow 2.11 and newer do not use CUDA directly on native Windows. For GPU training, open PowerShell as Administrator and enable WSL2:

```powershell
wsl --install
```

If Windows asks for a restart, restart the laptop. After Ubuntu is available, install the project requirements inside Ubuntu and verify GPU detection:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If a GPU device is listed, use the same training commands below inside WSL2.

## Step-by-Step Workflow

Run all commands from the `plant-disease-recognition` folder.

### 1. Extract selected classes from the dataset zip

Place `VIP_Dataset.zip` at:

```text
C:\Users\Acer\Downloads\VIP_Dataset.zip
```

Then run:

```powershell
python src/extract_selected_from_zip.py
```

This extracts only the selected tomato, potato and corn classes into:

```text
data/raw/train/
data/raw/valid/
```

### 2. Audit the raw dataset

```powershell
python src/dataset_audit.py
```

This script:

- lists available classes
- counts images per class
- detects unreadable image files
- saves `outputs/reports/raw_class_distribution.csv`
- saves `outputs/figures/raw_class_distribution.png`

### 3. Select the project subset

```powershell
python src/select_subset.py
```

This copies only the selected tomato, potato and corn classes into:

```text
data/selected/train/
data/selected/valid/
```

### 4. Prepare train, validation and test splits

```powershell
python src/prepare_splits.py
```

This script:

- uses the selected Kaggle `train` folder as the training set
- splits the selected Kaggle `valid` folder into validation and final test subsets
- uses stratified sampling so every class is represented
- saves the final dataset into `data/processed/`
- saves `outputs/reports/processed_split_summary.csv`

Final split layout:

```text
data/processed/train/
data/processed/valid/
data/processed/test/
```

### 5. Train the Simple CNN baseline

```powershell
python src/train_simple_cnn.py
```

This saves:

```text
outputs/models/simple_cnn.keras
outputs/figures/simple_cnn_training_curves.png
```

For a short smoke test that does not produce the final model:

```powershell
python src/train_simple_cnn.py --quick-test
```

### 6. Train the MobileNetV2 transfer-learning model

```powershell
python src/train_mobilenetv2.py
```

This saves:

```text
outputs/models/mobilenetv2.keras
outputs/figures/mobilenetv2_training_curves.png
```

For a short smoke test that does not produce the final model:

```powershell
python src/train_mobilenetv2.py --quick-test
```

Optional fine-tuning after the first MobileNetV2 model is working:

```powershell
python src/fine_tune_mobilenetv2.py
```

### 7. Evaluate both models

```powershell
python src/evaluate_models.py
```

This calculates:

- accuracy
- macro precision
- macro recall
- macro F1-score
- weighted F1-score
- confusion matrix
- model size
- average inference time per image

Outputs are saved in:

```text
outputs/reports/
outputs/figures/
```

### 8. Generate Grad-CAM examples

```powershell
python src/gradcam.py
```

This saves correct and incorrect prediction examples, where available, in:

```text
outputs/figures/gradcam_examples/
```

Each heatmap should be checked manually to explain whether the model focuses on visible leaf symptom regions or irrelevant background.

### 9. Run the Streamlit application

```powershell
streamlit run app/streamlit_app.py
```

The app allows users to:

- upload one leaf image
- choose the available MobileNetV2 model, with the best evaluated model selected by default
- optionally select a final test sample for demonstration and ground-truth checking
- view the uploaded image
- see the predicted class
- see the confidence score
- see the top three predictions
- see inference time
- view a Grad-CAM overlay

Disclaimer shown in the app:

```text
This prototype is for educational and preliminary screening purposes only and does not replace expert agricultural diagnosis.
```

## Notes for Presentation

- Simple CNN is the baseline model.
- MobileNetV2 is the main model because it uses transfer learning.
- EfficientNetB0 is optional and can be added later only after the first two models work.
- Since the dataset is already augmented, the training scripts use only light online augmentation.
- Validation and test sets are not manually augmented.

## Current Final Results

The completed run used 8 selected classes with 14,808 training images, 1,851 validation images and 1,851 final test images.

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Model Size | Avg Inference Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple CNN | 91.84% | 92.62% | 91.54% | 91.66% | 91.75% | 1.33 MB | 0.0012 s/image |
| MobileNetV2 | 95.30% | 95.26% | 95.27% | 95.22% | 95.26% | 9.31 MB | 0.0022 s/image |
| MobileNetV2 fine-tuned | 96.49% | 96.62% | 96.48% | 96.44% | 96.47% | 20.84 MB | 0.0021 s/image |

The fine-tuned MobileNetV2 achieved the strongest final test performance and is used as the main model for the Streamlit demo and Grad-CAM analysis.

Generated results are saved in:

```text
outputs/reports/
outputs/figures/
outputs/models/
```
