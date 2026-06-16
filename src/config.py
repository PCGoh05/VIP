from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SELECTED_DIR = DATA_DIR / "selected"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Kept small for a student project. You can increase these after confirming
# that the full pipeline works on your computer.
SIMPLE_CNN_EPOCHS = 12
MOBILENET_EPOCHS = 12
VALID_TEST_SIZE = 0.5

# Eight selected classes from tomato, potato and corn.
SELECTED_CLASSES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
]
