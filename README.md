# SeeClearly AI — Diabetic Retinopathy Detection

> AI-powered early detection of Diabetic Retinopathy from retinal fundus images using deep learning with Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![React](https://img.shields.io/badge/React-18-blue?logo=react)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

SeeClearly AI classifies retinal fundus images into **5 stages** of Diabetic Retinopathy:

| Stage | Description | Severity |
|-------|-------------|----------|
| **No_DR** | No retinopathy detected | ✅ Normal |
| **Mild** | Microaneurysms present | ⚠️ Low |
| **Moderate** | Hemorrhages + exudates | 🟠 Medium |
| **Severe** | Significant hemorrhages, IRMA | 🔴 High |
| **PDR** | Neovascularization (new vessel growth) | 🚨 Critical |

### Key Features

- **EfficientNetB3** transfer learning with focal loss for class imbalance
- **Grad-CAM** heatmap explainability — see what the model focuses on
- **Production-grade Flask API** with automatic mock fallback
- **Premium dark-mode React UI** with glassmorphism design
- **Docker deployment** ready

---

## 📁 Project Structure

```
seeclearly_ai/
├── training/                    # 🧪 Model Training Pipeline
│   ├── config.py               # Training configuration constants
│   ├── data_preprocessing.py   # APTOS 2019 data loading + augmentation
│   ├── focal_loss.py           # Categorical focal loss implementation
│   ├── train.py                # Two-phase training script
│   ├── evaluate.py             # Confusion matrix, ROC-AUC, F1-score
│   └── requirements.txt        # Training dependencies
│
├── backend/                     # 🖥️ Flask API Server
│   ├── app.py                  # Flask endpoints (/health, /predict)
│   ├── dr_detection.py         # Prediction engine + mock fallback
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   └── requirements.txt        # Backend dependencies
│
├── src/                         # 🎨 React Frontend
│   ├── pages/Index.tsx         # Main analysis page
│   ├── components/
│   │   ├── ImageUploader.tsx   # Drag-and-drop image upload
│   │   ├── ResultDisplay.tsx   # Full result view
│   │   ├── SeverityGauge.tsx   # Animated severity indicator
│   │   ├── HeatmapViewer.tsx   # Original/Grad-CAM/Overlay toggle
│   │   ├── AnalysisAnimation.tsx # Processing animation
│   │   └── ConnectionStatus.tsx  # Backend status indicator
│   └── lib/
│       ├── api.ts              # API client with fallback
│       └── mockPredictor.ts    # Client-side mock for offline dev
│
├── models/                      # 📦 Trained Model Weights
│   └── .gitkeep                # Place dr_efficientnet_b3.keras here
│
├── utils/                       # 🔧 Shared Utilities
│   ├── preprocessing.py        # Image preprocessing functions
│   └── __init__.py
│
├── docs/                        # 📚 Documentation
│   ├── research_improvements.md
│   └── deployment_guide.md
│
├── Dockerfile                   # Docker multi-stage build
├── docker-compose.yml           # Single-command deployment
├── requirements.txt             # All Python dependencies
├── package.json                 # Frontend dependencies
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** v18+ ([download](https://nodejs.org/))
- **Python** v3.10+ ([download](https://www.python.org/))
- **pip** (comes with Python)

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/your-username/seeclearly-ai.git
cd seeclearly-ai

# Install frontend dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Locally

**Frontend only** (mock mode — no model needed):
```bash
npm run dev
# Opens at http://localhost:8080
```

**Frontend + Backend** (with Flask API):
```bash
# Terminal 1: Start Flask backend
python backend/app.py
# Server starts at http://localhost:5001

# Terminal 2: Start Vite frontend
npm run dev
# Opens at http://localhost:8080
```

> **Note:** Without a trained model file in `models/`, the backend runs in **mock mode** with simulated predictions. The UI works identically — you'll see a "Mock Mode" badge.

---

## 🧪 Training the Model

Training requires a **GPU** (Google Colab T4 or better recommended).

### 1. Download APTOS 2019 Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key (download from kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d aptos2019-blindness-detection/
```

**Expected folder structure:**
```
aptos2019-blindness-detection/
├── train.csv              # 3,662 rows (id_code, diagnosis)
├── train_images/          # 3,662 PNG fundus images
├── test.csv
└── test_images/
```

### 2. Train on Colab

1. Upload the `training/` folder to Google Colab
2. Upload the dataset to Colab or mount from Google Drive
3. Update `DATA_PATH` in `training/config.py` if needed
4. Run:

```python
# In a Colab cell:
%cd /content/training
!python train.py
```

**Training takes ~2-3 hours** on a T4 GPU.

### 3. Evaluate

```python
!python evaluate.py
```

Outputs saved to `training/outputs/`:
- `confusion_matrix.png`
- `classification_report.txt`
- `roc_curves.png`
- `training_history.png`

### 4. Download Model

Download `models/dr_efficientnet_b3.keras` from Colab and place it in the `models/` directory locally.

---

## 📊 Expected Performance

| Metric | Expected |
|--------|----------|
| **Overall Accuracy** | 82-85% |
| **Quadratic Weighted Kappa** | 0.78-0.85 |
| **Mean ROC-AUC** | 0.90+ |
| **No_DR Recall** | 90-95% |
| **Mild Recall** | 55-65% |
| **Moderate Recall** | 80-85% |
| **Severe Recall** | 60-70% |
| **PDR Recall** | 75-85% |

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:5001
```

For detailed deployment instructions (Render, HuggingFace Spaces), see [docs/deployment_guide.md](docs/deployment_guide.md).

---

## 🔬 Research Improvements

See [docs/research_improvements.md](docs/research_improvements.md) for detailed suggestions on:
- Vision Transformers (ViT, DeiT)
- Ensemble methods
- Advanced preprocessing (Ben Graham's technique)
- Multi-modal fusion
- Federated learning
- Uncertainty quantification

---

## ⚠️ Disclaimer

This is an **AI-assisted screening tool** for educational and research purposes only. It does **not** constitute medical advice. Always consult a qualified ophthalmologist for diagnosis and treatment of Diabetic Retinopathy.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
