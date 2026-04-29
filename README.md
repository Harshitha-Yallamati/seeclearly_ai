# RetinoCheck — Diabetic Retinopathy Detection

> **AI-powered early screening of Diabetic Retinopathy from retinal fundus images using Xception Deep Learning with Grad-CAM++ explainability and clinical triage.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![React](https://img.shields.io/badge/React-18-blue?logo=react)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-LIVE%20AI-brightgreen)

---

## 🎯 Project Overview

RetinoCheck is a **full-stack medical imaging application** designed for early screening of Diabetic Retinopathy (DR). It uses **Transfer Learning with Xception** (pre-trained on ImageNet) to classify retinal fundus images into 5 severity stages, provides **Grad-CAM++ heatmaps** for explainability, and outputs **clinical triage recommendations** for healthcare professionals.

### The Problem
Diabetic Retinopathy is a leading cause of preventable blindness worldwide. Early detection is critical, but manual screening requires specialized ophthalmologists and is time-consuming.

### Our Solution
A production-ready web platform that provides:
1. **5-Stage Classification** — No DR, Mild, Moderate, Severe, Proliferative DR
2. **Explainable AI (XAI)** — Grad-CAM++ attention heatmaps showing exactly where the model detected lesions
3. **Clinical Triage System** — Automated action recommendations (Routine / Monitor / Refer / Urgent Referral)
4. **Safety Systems** — OOD detection (rejects non-fundus images) + confidence thresholding (rejects uncertain predictions <60%)
5. **Premium UI** — Modern glassmorphism interface with real-time analysis animations

---

## 🧠 AI Architecture

### Model: Xception (Transfer Learning)

| Property | Value |
|---|---|
| **Base Model** | Xception (ImageNet pre-trained) |
| **Input Size** | 299 × 299 × 3 (RGB) |
| **Head** | Flatten → Dense(5, softmax) |
| **Loss** | Categorical Crossentropy |
| **Optimizer** | Adam |
| **Training Data** | Kaggle Diabetic Retinopathy Level Detection Dataset |
| **Output Classes** | 5 (No_DR, Mild, Moderate, Severe, PDR) |

### How Prediction Works

```
Upload Image → OOD Check → Resize to 299×299 → Rescale (÷ 255)
    → Xception Inference → Softmax Probabilities
    → Confidence Check (≥60%) → Grad-CAM++ Heatmap
    → Clinical Triage → Final Result
```

1. **OOD Guard**: Verifies the image is a retinal fundus scan (RGB channel analysis)
2. **Preprocessing**: Resize to 299×299, normalize pixels to [0, 1]
3. **Inference**: Xception CNN extracts features and classifies into 5 DR stages
4. **Confidence Gate**: Predictions below 60% confidence are rejected with a "Retake" warning
5. **Explainability**: Grad-CAM++ generates attention heatmaps from `block14_sepconv2_act` layer
6. **Triage Output**: Maps severity to clinical action (Routine → Urgent Referral)

---

## 🏥 Clinical Decision Layer

| DR Stage | Severity Index | Triage Action | Meaning |
|---|---|---|---|
| No DR | 0 | **Routine** | No signs of disease. Annual checkup recommended |
| Mild | 1 | **Monitor** | Early microaneurysms detected. Recheck in 6 months |
| Moderate | 2 | **Refer** | Widespread hemorrhages. Ophthalmology referral needed |
| Severe | 3 | **Urgent Referral** | Significant vessel damage. Immediate specialist review |
| PDR | 4 | **Urgent Referral** | Proliferative stage. Risk of vision loss |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Deep Learning** | TensorFlow/Keras, Xception, NumPy |
| **Explainability** | Grad-CAM++, OpenCV |
| **Backend** | Flask, Python 3.10+, Flask-CORS |
| **Frontend** | React 18, TypeScript, Vite |
| **UI** | Tailwind CSS, Shadcn/UI, Radix UI, Lucide Icons |
| **Deployment** | Docker, Docker Compose |

---

## 📁 Project Structure

```
retinocheck/
├── backend/                 # Flask API Server
│   ├── app.py               # REST endpoints (/predict, /health)
│   ├── dr_detection.py      # Model loading, preprocessing, inference, triage
│   └── gradcam.py           # Grad-CAM++ heatmap generation
│
├── training/                # Model Training Pipeline
│   ├── train.py             # Xception training script (Flatten + Dense head)
│   ├── config.py            # Hyperparameters, paths, augmentation config
│   ├── data_preprocessing.py # Data loading & generators (rescale only)
│   └── evaluate.py          # Model evaluation & metrics
│
├── src/                     # React Frontend
│   ├── components/          # UI components (ImageUploader, HeatmapViewer, SeverityGauge)
│   ├── lib/                 # API client & mock predictor
│   └── pages/               # Main page (Index.tsx)
│
├── models/                  # Trained Model Weights
│   └── Updated-Xception-diabetic-retinopathy.h5  (96 MB)
│
├── docs/                    # Documentation
├── public/                  # Static assets
├── utils/                   # Helper utilities
│
├── requirements.txt         # Python dependencies
├── package.json             # Node.js dependencies
├── Dockerfile               # Container build
└── docker-compose.yml       # Full-stack deployment
```

---

## 🔒 Safety Features

### 1. Out-of-Distribution (OOD) Detection
Rejects non-fundus images before they reach the CNN. Uses RGB channel dominance analysis — retinal images have characteristic R > G > B patterns.

### 2. Confidence Thresholding
If the model's maximum softmax probability is below **60%**, the system returns:
> "Low confidence. Please retake the image to ensure accurate screening."

### 3. Graceful Fallback
If no trained model is available, the system automatically enters **Demo Mode** with simulated predictions, keeping the UI fully functional for development.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- TensorFlow 2.x

### 1. Install Dependencies
```bash
# Frontend
npm install

# Backend
pip install -r requirements.txt
```

### 2. Place Model File
Ensure the trained `.h5` model file is inside the `models/` directory:
```
models/Updated-Xception-diabetic-retinopathy.h5
```

### 3. Run the Full Stack
```bash
# Option 1: One command (Windows)
npm run dev:full

# Option 2: Separate terminals
python backend/app.py          # Backend → http://localhost:5001
npx vite                       # Frontend → http://localhost:8080
```

### 4. Use the App
1. Open `http://localhost:8080` in your browser
2. Verify the header shows **"LIVE MODEL"** (green dot)
3. Upload a retinal fundus image
4. Click **"Analyze Image"**
5. View: severity classification, confidence %, probability bars, Grad-CAM heatmap, triage recommendation

---

## 🧪 Training Your Own Model

If you need to retrain the model:

1. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d arbethi/diabetic-retinopathy-level-detection
   ```

2. Extract it to the project root as `aptos2019-blindness-detection/`

3. Run the training script (GPU recommended):
   ```bash
   cd training
   python train.py
   ```

4. The trained model will be saved to `models/Updated-Xception-diabetic-retinopathy.h5`

See `training/COLAB_GUIDE.md` for Google Colab instructions.

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns model status (LIVE/FALLBACK_MOCK), TF availability, CAM layer info |
| `/predict` | POST | Accepts multipart image upload, returns full prediction with probabilities, heatmap, triage |

### Sample `/predict` Response
```json
{
  "label": "Moderate",
  "severity_index": 2,
  "triage_action": "Refer",
  "confidence": 0.867,
  "probabilities": { "No_DR": 0.014, "Mild": 0.042, "Moderate": 0.867, "Severe": 0.077, "PDR": 0.0 },
  "explanation": "Triage Action: Refer\n\nModerate non-proliferative diabetic retinopathy detected...",
  "heatmap": "<base64>",
  "overlay": "<base64>",
  "is_mock": false,
  "needs_doctor": true
}
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

This starts both the Flask backend and Vite frontend in a single containerized environment.

---

## ⚠️ Disclaimer

This is an **AI-assisted screening tool** for **educational and research purposes only**. It does **not** constitute medical advice or diagnosis. Always consult a qualified ophthalmologist for diagnosis and treatment of Diabetic Retinopathy.

---

## 👥 Team

**RetinoCheck** — SRM University AP

## 📝 License

SRMAP License — © 2026 RetinoCheck Project
