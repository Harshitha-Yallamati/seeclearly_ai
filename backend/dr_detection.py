# -*- coding: utf-8 -*-
"""
Core DR prediction logic with live-model loading and mock fallback.
"""

import os
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from gradcam import (
    DEFAULT_CAM_METHOD,
    generate_heatmap_assets,
    get_medical_explanation,
    resolve_cam_layer,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "dr_efficientnet_b3.keras")
IMG_SIZE = (224, 224)
LABEL_MAP = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]
MODEL_EXTENSIONS = {".keras", ".h5", ".hdf5"}
CAM_LAYER_NAME = os.getenv("DR_CAM_LAYER", "block6a_expand_activation").strip()

_MODEL_INSTANCE = None
_MODEL_LOADED = False
_MODEL_PATH = None
_MODEL_LOAD_ERROR = None


def _load_model_for_inference(model_path):
    """
    Load a Keras model in inference mode.

    compile=False avoids failures caused by missing optimizer / loss objects when
    the saved model is only being used for prediction.
    """
    try:
        return load_model(model_path, compile=False)
    except Exception as first_error:
        # Fall back to the focal-loss custom object if the model was saved with it.
        try:
            import sys

            training_dir = os.path.join(BASE_DIR, "training")
            if training_dir not in sys.path:
                sys.path.insert(0, training_dir)
            from focal_loss import categorical_focal_loss

            return load_model(
                model_path,
                compile=False,
                custom_objects={"focal_loss_fixed": categorical_focal_loss()},
            )
        except Exception as second_error:
            raise RuntimeError(
                f"{type(first_error).__name__}: {first_error}; "
                f"fallback failed with {type(second_error).__name__}: {second_error}"
            ) from second_error


def _discover_model_candidates():
    """
    Search the project for likely trained-model files.
    """
    seen = set()
    candidates = []

    def add(path_value):
        if not path_value:
            return
        path_obj = Path(path_value).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path(BASE_DIR) / path_obj
        resolved = path_obj.resolve()
        path_str = str(resolved)
        if path_str not in seen:
            seen.add(path_str)
            candidates.append(path_str)

    env_path = os.getenv("DR_MODEL_PATH", "").strip()
    add(DEFAULT_MODEL_PATH)
    if env_path:
        add(env_path)

    preferred_dirs = [
        Path(BASE_DIR) / "models",
        Path(BASE_DIR) / "training" / "outputs",
    ]

    for directory in preferred_dirs:
        if directory.is_dir():
            for preferred_name in [
                "dr_efficientnet_b3.keras",
                "dr_efficientnet_b3.h5",
                "best_model.keras",
                "best_model.h5",
                "model.keras",
                "model.h5",
            ]:
                add(directory / preferred_name)

            discovered = sorted(
                (
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for path in discovered:
                add(path)

    return [path for path in candidates if os.path.exists(path)]


def get_model():
    """
    Load the trained model once and cache it.
    """
    global _MODEL_INSTANCE, _MODEL_LOADED, _MODEL_PATH, _MODEL_LOAD_ERROR

    if _MODEL_LOADED:
        return _MODEL_INSTANCE

    _MODEL_LOAD_ERROR = None

    if not TF_AVAILABLE:
        _MODEL_LOAD_ERROR = "TensorFlow is not installed."
        _MODEL_LOADED = True
        print("Warning: TensorFlow is not installed; prediction will use fallback mode.")
        return None

    candidates = _discover_model_candidates()
    if not candidates:
        _MODEL_LOAD_ERROR = (
            "No trained model file was found. Place the exported model in "
            "'models/dr_efficientnet_b3.keras' or set DR_MODEL_PATH."
        )
        _MODEL_LOADED = True
        print(f"Warning: {_MODEL_LOAD_ERROR}")
        return None

    last_error = None
    for candidate in candidates:
        try:
            _MODEL_INSTANCE = _load_model_for_inference(candidate)
            _MODEL_PATH = candidate
            _MODEL_LOADED = True
            print(f"Model loaded from: {candidate}")
            return _MODEL_INSTANCE
        except Exception as exc:
            last_error = f"Failed to load {candidate}: {exc}"

    _MODEL_LOAD_ERROR = last_error or "Unknown model-loading error."
    _MODEL_INSTANCE = None
    _MODEL_LOADED = True
    print(f"Warning: {_MODEL_LOAD_ERROR}")
    return None


def get_model_status():
    """
    Return the current backend model status for health checks and debugging.
    """
    model = get_model()
    cam_layer = None
    if model is not None:
        try:
            cam_layer = resolve_cam_layer(model, CAM_LAYER_NAME)
        except Exception:
            cam_layer = CAM_LAYER_NAME or None

    return {
        "model_loaded": model is not None,
        "model_path": _MODEL_PATH or DEFAULT_MODEL_PATH,
        "mode": "LIVE" if model is not None else "FALLBACK_MOCK",
        "tf_available": TF_AVAILABLE,
        "load_error": _MODEL_LOAD_ERROR,
        "cam_layer": cam_layer,
        "cam_method": DEFAULT_CAM_METHOD,
    }


def _decode_image(img_path_or_file):
    if isinstance(img_path_or_file, str):
        img = cv2.imread(img_path_or_file, cv2.IMREAD_COLOR)
    else:
        file_bytes = np.frombuffer(img_path_or_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def preprocess_image(img_path_or_file):
    """
    Decode and resize the image into the model's training-time input format.
    """
    img = _decode_image(img_path_or_file)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def _normalize_probabilities(probabilities):
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    if probs.size != len(LABEL_MAP):
        raise ValueError(
            f"Expected {len(LABEL_MAP)} class probabilities, received {probs.size}."
        )

    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Model returned an invalid probability vector.")

    return probs / total


def _mock_probabilities():
    stage_index = int(np.random.choice(5, p=[0.35, 0.15, 0.25, 0.10, 0.15]))
    scores = np.random.dirichlet(np.ones(5) * 0.5) * 0.3
    scores[stage_index] += 0.7
    return _normalize_probabilities(scores)


def predict_dr(img_path_or_file):
    """
    Full prediction pipeline:
        image -> probabilities -> label/confidence -> Grad-CAM -> explanation
    """
    model = get_model()

    if hasattr(img_path_or_file, "seek"):
        img_path_or_file.seek(0)

    img = preprocess_image(img_path_or_file)
    if img is None:
        return {"error": "Invalid image. Could not decode the uploaded file."}

    img_expanded = np.expand_dims(img, axis=0)
    prediction_is_mock = False

    if model is not None:
        raw_pred = model.predict(img_expanded, verbose=0)[0]
        probs = _normalize_probabilities(raw_pred)
    else:
        prediction_is_mock = True
        probs = _mock_probabilities()

    label_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label_str = LABEL_MAP[label_idx]
    probability_map = {
        LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(len(LABEL_MAP))
    }

    heatmap_assets = generate_heatmap_assets(
        img_np=img,
        model=model,
        conv_layer=CAM_LAYER_NAME,
        pred_index=label_idx,
        method=DEFAULT_CAM_METHOD,
    )
    explanation = get_medical_explanation(label_str, confidence)

    return {
        "label": label_str,
        "severity_index": label_idx,
        "confidence": round(confidence, 4),
        "probabilities": probability_map,
        "all_probabilities": probability_map,
        "explanation": explanation,
        "heatmap": heatmap_assets["heatmap"],
        "overlay": heatmap_assets["overlay"],
        "gradcam_layer": heatmap_assets["conv_layer"],
        "gradcam_method": heatmap_assets["method"],
        "is_mock": prediction_is_mock,
        "heatmap_is_mock": heatmap_assets["is_mock"],
        "needs_doctor": label_idx >= 2 or confidence < 0.75,
    }
