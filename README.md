# 🐄 Livestock Health AI: Cattle Disease Classification & Digital Twin

An integrated deep learning and IoT Digital Twin system for individual animal health screening, disease prediction, longitudinal telemetry tracking, and explainable multimodal risk assessment.

### Core Capabilities:
- **Computer-Vision Screening:** Fast, accurate identification of **Healthy**, **Foot-and-Mouth Disease (FMD)**, and **Lumpy Skin Disease (LSD)** using fine-tuned convolutional architectures.
- **Explainable AI (Grad-CAM):** Visual activation heatmaps highlighting the exact anatomical lesions and regions guiding the network's diagnosis.
- **Individual Animal Digital Twins:** Unique animal profiles tracking species, breed, historical screening trends, and real-time state.
- **Vital Signs & Environmental Telemetry:** Ingestion of body temperature, activity level, feed intake, barn ambient temperature, and humidity, with clear provenance badges (`[MANUAL ENTRY]` vs `[SIMULATED SENSOR]`).
- **Transparent Multimodal Risk Assessment:** Rule-based scoring engine (0–9 scale: Low, Moderate, High) with itemized contributing factors.
- **Local SQLite Persistence:** Complete offline persistence of animal profiles, screening records, sensor telemetry, and risk scores.
- **Clinical Explainability Reports:** Grounded follow-up guidance adhering strictly to veterinary screening protocols.

---

> ### ⚠️ Veterinary Medical Disclaimer
> **This software provides automated preliminary computer-vision screening estimates and prototype risk scores. Predictions and risk assessments are AI estimates and NOT confirmed veterinary diagnoses.**
> Always consult a licensed veterinarian or certified animal health authority for clinical diagnosis, confirmatory laboratory testing (PCR / ELISA), quarantine measures, and treatment plans.

---

## 🧬 Digital Twin Architecture & Risk Engine

Reusing and operationalizing the research algorithms from `dumm/notebookef7b497bba (1).ipynb`:

### 1. Data Structure
Each animal Digital Twin maintains:
- `animal_id` (Unique Identifier, e.g. `C001`)
- `species` (e.g. `Cattle`, `Buffalo`, `Sheep`, `Goat`)
- `current_state`: Latest CV prediction, confidence, and timestamp
- `history`: Chronological audit log of all vision observations
- `sensor_data`: Body temperature (°C), activity level, feed intake, provenance tag
- `environment_data`: Ambient temperature (°C), relative humidity (%)
- `risk_assessment`: Multimodal score, risk status, and contributing factors

### 2. Transparent Risk Scoring Breakdown

| Evaluation Factor | Points Awarded | Condition |
| :--- | :---: | :--- |
| **Current Disease Prediction** | **+4** | Latest observation is FMD or LSD |
| **Historical Disease History** | **+1** | Earlier observation in history indicated disease |
| **Elevated Body Temperature** | **+1** | Body temp >= 39.0 °C (bovine fever threshold) |
| **Low Activity Level** | **+1** | Activity marked as 'Low' |
| **Reduced Feed Intake** | **+1** | Feed intake marked as 'Reduced' |
| **High Ambient Temperature** | **+1** | Ambient temp >= 30.0 °C (heat stress context) |
| **High Relative Humidity** | **+1** | Humidity >= 70.0 % (disease transmission context) |

**Status Categorization:**
- **Low Risk:** 0 – 2 points
- **Moderate Risk:** 3 – 5 points
- **High Risk:** 6 – 9 points

---

## 🔍 Model Architecture Inspection & Evaluation

Evaluated on the untouched test dataset (487 test images):

| Architecture | Checkpoint Path | Test Accuracy | Macro F1 | Status & Details |
| :--- | :--- | :---: | :---: | :--- |
| **DenseNet-121** | `models/densenet121_cattle_best.pth` | **96.30%** | **0.9639** | **Selected Default.** Highest overall accuracy and balanced F1 score. Supports Grad-CAM via `features.denseblock4.denselayer16.conv2`. Fully offline & self-contained (28.4 MB). |
| **ResNet-50** | `/kaggle/working/resnet50_cattle_best.pth` | **95.89%** | **0.9637** | Trained in notebook `resnet50.ipynb`, but weights file was not saved/downloaded to local repository. |
| **EfficientNet-B0 (Torchvision)** | `models/efficientnet_cattle_best.pth` | **94.66%** | **0.9525** | **Fully Supported Alternative.** Fast and compact (16.3 MB). Supports Grad-CAM via `features[-1]`. |
| **DINOv2 ViT-S/14** | `models/dinov2_vits14_cattle_best.pth` | **94.05%** | **0.9444** | Self-supervised Vision Transformer head (88.3 MB). Requires remote GitHub access (`torch.hub`). |
| **EfficientNet-B0 (LMO Optimized)** | `models/efficientnet_b0_lmo_best.pth` | **93.02%** | **0.9354** | Outer hyperparameter optimization using Lexicographic Multi-Objective. Requires external `timm` library. |

---

## 🎯 Model Pipeline Details

- **Class Mapping (0-Indexed):**
  - `0`: `healthy` (Healthy)
  - `1`: `foot-and-mouth` (Foot-and-Mouth Disease)
  - `2`: `lumpy` (Lumpy Skin Disease)
- **Input Resolution:** `224 × 224` pixels
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Inference Hardware:** Automatic device selection (`CUDA` GPU if present, otherwise `CPU`)

---

## 🚀 How to Run the Application on Windows

### Prerequisites
Python 3.10+ (tested on Python 3.14 on Windows) with dependencies installed.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
You can start the app in any of the following ways:

#### Option A: Double-Click the Windows Launcher
Double-click `run_app.bat` in File Explorer.

#### Option B: PowerShell Launcher
```powershell
.\run_app.ps1
```

#### Option C: Direct Command
```bash
streamlit run app.py
```
*(If your `python` command points to the Windows Store stub, use the Python binary directly:)*
```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run app.py
```

The application will launch and open automatically in your browser at:
**`http://localhost:8501`**

---

## 🧪 Running the Automated Unit Test Suite

Run the full automated test suite (14 tests covering computer-vision models, Grad-CAM, SQLite persistence, telemetry logging, and multimodal risk assessment):

```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

All 14 tests should pass with status `OK`.

---

## 📁 Project Structure

```
Livestock_Health_AI/
├── app.py                     # Streamlit App (Diagnosis, Digital Twins, History, Benchmarks)
├── run_app.bat                # Windows Batch launcher
├── run_app.ps1                # Windows PowerShell launcher
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & benchmark analysis
├── data/                      # Local persistent storage
│   └── digital_twins.db       # SQLite Database (Animals, Observations, Sensors, Risk)
├── models/                    # Saved model weight checkpoints
│   ├── densenet121_cattle_best.pth       # Selected DenseNet-121 weights (96.30%)
│   ├── efficientnet_cattle_best.pth      # EfficientNet-B0 baseline weights (94.66%)
│   ├── efficientnet_b0_lmo_best.pth      # EfficientNet-B0 LMO weights
│   └── dinov2_vits14_cattle_best.pth     # DINOv2 ViT-S/14 weights
├── sample_images/             # Preloaded demo test images
│   ├── healthy_sample1.jpg
│   ├── healthy_sample2.jpg
│   ├── foot_and_mouth_sample1.jpg
│   ├── foot_and_mouth_sample2.jpg
│   ├── lumpy_skin_sample1.jpg
│   └── lumpy_skin_sample2.jpg
├── src/                       # Core python package
│   ├── __init__.py
│   ├── config.py              # Constants, thresholds, and paths
│   ├── database.py            # SQLite schema, migrations, and CRUD operations
│   ├── digital_twin.py        # Digital Twin domain model & multimodal risk engine
│   ├── model.py               # Model builders, loaders, and inference logic
│   └── gradcam.py             # Grad-CAM heatmap and overlay generation
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_pipeline.py       # End-to-end tests for CV models and Grad-CAM
│   └── test_digital_twin.py   # Tests for Digital Twin state, telemetry & risk engine
└── dumm/                      # Experimental notebooks and evaluation data
```
