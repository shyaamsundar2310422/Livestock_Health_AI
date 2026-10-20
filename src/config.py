"""
Configuration and constants for the Cattle Disease Classification application.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "digital_twins.db"

# DMI Model Configuration (INRAE Dataset 2 - Pooled Expanded Features)
DMI_MODEL_PATH = MODELS_DIR / "dmi_extra_trees_pooled_expanded.joblib"
DMI_MODEL_NAME = "Pooled Expanded Extra Trees Regressor"

# Exact 9 input features required by the Pooled Expanded model
DMI_FEATURE_NAMES = [
    "age",
    "parity",
    "DIM",
    "BW",
    "BCS",
    "Milk",
    "PC_Milk",
    "Fat_Milk",
    "Lac_Milk",
]

DMI_FEATURE_METADATA = {
    "age": {"label": "Age", "unit": "months", "min": 18.0, "max": 180.0, "default": 48.0, "help": "Cow age in months"},
    "parity": {"label": "Parity", "unit": "lactations", "min": 1.0, "max": 10.0, "default": 2.0, "help": "Lactation number / parity"},
    "DIM": {"label": "Days in Milk (DIM)", "unit": "days", "min": 1.0, "max": 500.0, "default": 120.0, "help": "Days elapsed in current lactation cycle"},
    "BW": {"label": "Body Weight (BW)", "unit": "kg", "min": 350.0, "max": 1000.0, "default": 620.0, "help": "Live animal body weight in kilograms"},
    "BCS": {"label": "Body Condition Score (BCS)", "unit": "1-5 scale", "min": 1.0, "max": 5.0, "default": 3.0, "help": "Body condition score on 1.0 to 5.0 scale"},
    "Milk": {"label": "Daily Milk Yield", "unit": "kg/day", "min": 0.0, "max": 75.0, "default": 28.5, "help": "Daily milk production in kg/day"},
    "PC_Milk": {"label": "Milk Protein Content (PC)", "unit": "g/kg", "min": 20.0, "max": 55.0, "default": 32.5, "help": "Milk protein content in g/kg"},
    "Fat_Milk": {"label": "Milk Fat Content", "unit": "g/kg", "min": 25.0, "max": 65.0, "default": 38.0, "help": "Milk fat content in g/kg"},
    "Lac_Milk": {"label": "Milk Lactose Content", "unit": "g/kg", "min": 35.0, "max": 60.0, "default": 48.0, "help": "Milk lactose content in g/kg"},
}

DMI_DISCLAIMER = (
    "**⚠️ Dry Matter Intake (DMI) Research Disclaimer:** This module provides experimental DMI estimates (kg/day) "
    "developed from the INRAE Dataset 2 (DataBase_Cows.csv). This model is research-stage and has NOT been "
    "validated across commercial herds or different genetic breeds. Predicted DMI represents an estimated "
    "physiological consumption rate and is strictly **NOT a prescribed feed ration or feed-purchasing recommendation**. "
    "Always consult a qualified animal nutritionist or herd veterinarian for balanced ration formulation."
)

# Sensor and Vital Constants
NORMAL_BODY_TEMP_MIN = 38.0  # °C (Bovine normal minimum)
NORMAL_BODY_TEMP_MAX = 39.0  # °C (Bovine normal maximum; >=39.0°C considered elevated/fever)
HEAT_STRESS_AMBIENT_TEMP = 30.0  # °C (Threshold for heat stress context)
HIGH_HUMIDITY_THRESHOLD = 70.0  # % (Threshold for environmental heat/disease stress context)

ACTIVITY_LEVELS = ["Normal", "Low", "High"]
FEED_INTAKE_LEVELS = ["Normal", "Reduced", "Increased"]
SPECIES_LIST = ["Cattle", "Buffalo", "Sheep", "Goat"]

# Trained Model Weight Paths
DENSENET121_WEIGHTS = MODELS_DIR / "densenet121_cattle_best.pth"
EFFICIENTNET_WEIGHTS = MODELS_DIR / "efficientnet_cattle_best.pth"

# Exact class labels as mapped during training in the notebooks
# Order: Index 0 -> healthy, Index 1 -> foot-and-mouth, Index 2 -> lumpy
CLASS_KEYS = ["healthy", "foot-and-mouth", "lumpy"]

CLASS_DISPLAY_NAMES = {
    "healthy": "Healthy",
    "foot-and-mouth": "Foot-and-Mouth Disease (FMD)",
    "lumpy": "Lumpy Skin Disease (LSD)",
}

CLASS_DESCRIPTIONS = {
    "healthy": (
        "The cattle shows typical visual characteristics of good health without evident cutaneous lesions, "
        "vesicles, or distinctive symptoms of infectious mucosal/skin diseases."
    ),
    "foot-and-mouth": (
        "Foot-and-Mouth Disease (FMD) is a highly contagious viral disease affecting cloven-hoofed animals. "
        "Key clinical signs include fever, blistering/vesicles and erosions on the muzzle, tongue, lips, teats, "
        "and coronary band of the feet, leading to excessive salivation and lameness."
    ),
    "lumpy": (
        "Lumpy Skin Disease (LSD) is a poxviral disease characterized by distinct circumscribed, firm nodules "
        "(lumps) on the skin and mucous membranes, fever, enlarged lymph nodes, and potential subcutaneous edema."
    ),
}

# Image Preprocessing Constants (matching training notebooks)
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Veterinary Disclaimer
VETERINARY_DISCLAIMER = (
    "**⚠️ Veterinary Medical Disclaimer:** This tool provides automated preliminary computer-vision estimates "
    "based on deep learning models and is intended solely for research and screening support. "
    "Predictions are **AI estimates and NOT confirmed veterinary diagnoses**. "
    "Always consult a licensed veterinarian or certified livestock health authority for clinical diagnosis, "
    "quarantine measures, and treatment."
)

AVAILABLE_MODELS = {
    "DenseNet-121 (Recommended - 96.30% Acc)": {
        "id": "densenet121",
        "file": DENSENET121_WEIGHTS,
        "test_accuracy": "96.30%",
        "macro_f1": "0.9639",
        "description": "Dense Convolutional Network (DenseNet-121). Achieved highest validation and test accuracy across all benchmarks.",
    },
    "EfficientNet-B0 (Torchvision - 94.66% Acc)": {
        "id": "efficientnet_b0",
        "file": EFFICIENTNET_WEIGHTS,
        "test_accuracy": "94.66%",
        "macro_f1": "0.9525",
        "description": "Standard EfficientNet-B0 baseline with compound scaling. Fast inference and compact footprint.",
    },
}
