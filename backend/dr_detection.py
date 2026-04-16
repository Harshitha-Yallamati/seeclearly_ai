# -*- coding: utf-8 -*-
"""
SeeClearly AI — DR Detection Engine

Core prediction logic with model management and mock fallback.
"""

import os
import numpy as np
import cv2

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from gradcam import generate_heatmap_overlay, get_medical_explanation

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dr_efficientnet_b3.keras")
IMG_SIZE = (224, 224)
LABEL_MAP = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]

# Singleton model holder
_MODEL_INSTANCE = None
_MODEL_LOADED = False


def get_model():
    """Load model once and cache it (singleton pattern)."""
    global _MODEL_INSTANCE, _MODEL_LOADED
    if not _MODEL_LOADED:
        if TF_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                # Import focal loss for custom_objects if model was saved with it
                import sys
                sys.path.insert(0, os.path.join(BASE_DIR, "training"))
                try:
                    from focal_loss import categorical_focal_loss
                    _MODEL_INSTANCE = load_model(
                        MODEL_PATH,
                        custom_objects={"focal_loss_fixed": categorical_focal_loss()},
                    )
                except ImportError:
                    _MODEL_INSTANCE = load_model(MODEL_PATH)
                print(f"✅ Model loaded from: {MODEL_PATH}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                _MODEL_INSTANCE = None
        else:
            if not TF_AVAILABLE:
                print("⚠️ TensorFlow not installed — running in MOCK mode")
            else:
                print(f"⚠️ Model not found at {MODEL_PATH} — running in MOCK mode")
            _MODEL_INSTANCE = None
        _MODEL_LOADED = True
    return _MODEL_INSTANCE


def get_model_status():
    """Get current model loading status for health checks."""
    model = get_model()
    return {
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "mode": "LIVE (EfficientNetB3)" if model is not None else "MOCK (Simulated predictions)",
        "tf_available": TF_AVAILABLE,
    }


def preprocess_image(img_path_or_file):
    """
    Preprocess image for EfficientNetB3 prediction.

    Parameters
    ----------
    img_path_or_file : str or file-like
        Path to image file or Flask file stream

    Returns
    -------
    img_norm : np.ndarray
        Preprocessed image array of shape (224, 224, 3) normalized to [0,1]
    """
    if isinstance(img_path_or_file, str):
        img = cv2.imread(img_path_or_file)
    else:
        nparr = np.frombuffer(img_path_or_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img / 255.0
    return img_norm


def predict_dr(img_path_or_file):
    """
    Full prediction pipeline: preprocess → predict → Grad-CAM → explanation.

    Parameters
    ----------
    img_path_or_file : str or file-like
        Image path or Flask uploaded file stream

    Returns
    -------
    dict with keys:
        label, confidence, explanation, heatmap (base64),
        is_mock, needs_doctor, severity_index
    """
    model = get_model()

    # Reset file pointer if needed
    if hasattr(img_path_or_file, "seek"):
        img_path_or_file.seek(0)

    img = preprocess_image(img_path_or_file)
    if img is None:
        return {"error": "Invalid image. Could not decode the uploaded file."}

    img_expanded = np.expand_dims(img, axis=0)

    # 1. PREDICTION
    is_mock = False
    if model is not None:
        pred = model.predict(img_expanded, verbose=0)[0]
        label_idx = int(np.argmax(pred))
        confidence = float(pred[label_idx])
        all_probs = {LABEL_MAP[i]: round(float(pred[i]), 4) for i in range(5)}
    else:
        # Mock fallback — generate realistic-looking predictions
        is_mock = True
        label_idx = int(np.random.choice(5, p=[0.35, 0.15, 0.25, 0.10, 0.15]))
        confidence = float(np.random.uniform(0.65, 0.96))

        # Generate fake probability distribution
        probs = np.random.dirichlet(np.ones(5) * 0.5)
        probs[label_idx] = confidence
        probs = probs / probs.sum()  # Re-normalize
        all_probs = {LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(5)}

    label_str = LABEL_MAP[label_idx]

    # 2. GRAD-CAM HEATMAP
    conv_layer_name = "top_activation"  # EfficientNetB3's last conv layer
    heatmap_b64 = generate_heatmap_overlay(
        img_np=img,
        model=model,
        conv_layer=conv_layer_name,
    )

    # 3. MEDICAL EXPLANATION
    explanation = get_medical_explanation(label_str, confidence)

    return {
        "label": label_str,
        "severity_index": label_idx,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "explanation": explanation,
        "heatmap": heatmap_b64,
        "is_mock": is_mock,
        "needs_doctor": label_idx >= 2 or confidence < 0.75,
    }
