# SeeClearly AI — Diabetic Retinopathy Detection

> **AI-powered early detection of Diabetic Retinopathy (DR) from retinal fundus images using Deep Learning with Grad-CAM explainability.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![React](https://img.shields.io/badge/React-18-blue?logo=react)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

SeeClearly AI is a full-stack medical imaging application designed to assist in the early screening of Diabetic Retinopathy. By leveraging state-of-the-art Computer Vision and Explainable AI (XAI), the platform provides both a severity classification and a visual heatmap indicating the pathological regions identified by the model.

### The Problem
Diabetic Retinopathy is a leading cause of blindness. Early detection is critical, but manual screening of retinal fundus images is time-consuming and requires specialized expertise.

### Our Solution
A premium, production-ready web platform that provides:
1.  **Instant Classification**: Categorizes images into 5 stages (No DR to PDR).
2.  **Explainable AI**: Visualizes model attention using **Grad-CAM++**.
3.  **Clinical Context**: Provides automated medical explanations for findings.
4.  **Seamless Experience**: A modern, interactive UI built for healthcare professionals.

---

## 🚀 Development Progress & Achievements (PPT Ready)

*Use the bullet points below for your project presentation slides:*

### ✅ 1. Advanced AI Backend
- **Core Architecture**: Integrated **EfficientNetB3** via Transfer Learning for high-precision feature extraction.
- **Explainability (XAI)**: Implemented **Grad-CAM++** to generate high-resolution attention heatmaps, allowing clinicians to see exactly which retinal regions (microaneurysms, hemorrhages) triggered the AI's decision.
- **Robust Inference Engine**: Built a Flask-based API with a **Live Model Discovery** system and an automatic **Mock Fallback** for offline development.
- **Preprocessing Pipeline**: Automated retinal mask estimation to suppress image artifacts and edge noise.

### ✅ 2. Premium UI/UX Design
- **Modern Aesthetic**: Developed a high-end **Glassmorphism** interface using Tailwind CSS and Framer Motion.
- **Interactive Visualization**: Created a custom **Heatmap Viewer** with real-time toggles (Original vs. Heatmap vs. Overlay).
- **Dynamic Analysis UI**: Implemented simulated progress animations and "Smart Gauges" for severity visualization.
- **Accessibility & Feedback**: Added a live **Connection Status** indicator and detailed error handling for robust user sessions.

### ✅ 3. Production Readiness
- **Docker Integration**: Fully containerized the stack (Flask + Vite) for one-command deployment using `docker-compose`.
- **Comprehensive Training Pipeline**: Developed a modular training system including **Categorical Focal Loss** to handle class imbalance in medical datasets.
- **Performance Optimized**: Optimized frontend builds using Vite and responsive components for cross-device compatibility.

---

## 🛠️ Technical Stack

| Component | Technologies |
|-----------|--------------|
| **Deep Learning** | TensorFlow 2.12, Keras, EfficientNetB3, NumPy |
| **Explainability** | Grad-CAM, Grad-CAM++, OpenCV |
| **Backend API** | Flask, Python 3.10+, Flask-CORS |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS |
| **UI Components** | Radix UI, Shadcn/UI, Lucide React |
| **Animations** | Framer Motion, CSS Keyframes |
| **Deployment** | Docker, Docker Compose |

---

## 📁 Architecture Overview

```bash
seeclearly_ai/
├── training/           # 🧪 Research & Development (Model Training)
│   ├── train.py        # Two-phase training logic
│   └── focal_loss.py   # Advanced loss for imbalanced medical data
├── backend/            # 🖥️ Inference API (Python/Flask)
│   ├── app.py          # RESTful endpoints (/predict, /health)
│   ├── dr_detection.py # Prediction logic & Model discovery
│   └── gradcam.py      # Heatmap generation & Image processing
├── src/                # 🎨 UI Layer (React/TypeScript)
│   ├── components/     # Reusable UI (ImageUploader, HeatmapViewer)
│   └── lib/            # API & Mock predictor logic
├── models/             # 📦 Serialized weights (.keras / .h5)
└── Dockerfile          # 🐳 Multi-stage production build
```

---

## 🔬 Medical AI Insight: How it Works

1.  **Image Normalization**: The input image is resized to 224x224 and normalized for the EfficientNetB3 backbone.
2.  **Feature Extraction**: The model processes the image through deep convolutional blocks, identifying complex patterns like cotton-wool spots or neovascularization.
3.  **Explainability Step**: During the backward pass, gradients are pooled from the `block6a_expand_activation` layer to create a **Class Activation Map (CAM)**.
4.  **Clinical Mapping**: The CAM is upsampled and overlaid on the original image, highlighting areas of high clinical significance in red/yellow.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install
pip install -r requirements.txt
```

### 2. Run the Full Stack
```bash
# Start both Backend and Frontend in one command (Windows)
npm run dev:full
```

*The frontend will be available at http://localhost:8080 and the backend at http://localhost:5001.*

---

## ⚠️ Disclaimer
This is an **AI-assisted screening tool** for educational and research purposes only. It does **not** constitute medical advice. Always consult a qualified ophthalmologist for diagnosis and treatment of Diabetic Retinopathy.

---

## 📝 License
SRMAP License — © 2026 SeeClearly AI Project
