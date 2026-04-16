# -*- coding: utf-8 -*-
"""
SeeClearly AI — Flask API Backend

Endpoints:
  GET  /health   → System health check with model status
  POST /predict  → Diabetic Retinopathy prediction with Grad-CAM heatmap

Run:
  python backend/app.py
  Server starts on http://localhost:5001
"""

import os
import sys
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add directories to sys.path for imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "utils"))

from dr_detection import predict_dr, get_model_status

app = Flask(__name__, static_folder=os.path.join(PROJECT_DIR, "dist"))
CORS(app)  # Enable CORS for frontend dev server


@app.route("/health", methods=["GET"])
def health():
    """Health check with model status information."""
    status = get_model_status()
    return jsonify({
        "status": "healthy",
        "model_loaded": status["model_loaded"],
        "model_path": status["model_path"],
        "mode": status["mode"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict DR stage from an uploaded retinal fundus image.

    Expects: multipart/form-data with 'image' file field
    Returns: JSON with prediction, confidence, heatmap (base64), explanation
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send a file with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    # Validate file type
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({
            "error": f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"
        }), 400

    try:
        result = predict_dr(file)

        if "error" in result:
            return jsonify(result), 422

        # Add preprocessing steps info for the UI
        result["preprocessing"] = [
            "Resized to 224×224",
            "Normalized pixel values (0–1)",
            "Applied EfficientNetB3 feature extraction",
            "Generated Grad-CAM heatmap overlay",
        ]

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# Serve static frontend in production
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve the React frontend build in production."""
    dist_dir = os.path.join(PROJECT_DIR, "dist")
    if path and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    return send_from_directory(dist_dir, "index.html")


if __name__ == "__main__":
    print("=" * 50)
    print("  SeeClearly AI — Backend API")
    print("=" * 50)

    status = get_model_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Model: {status['model_path']}")
    print(f"  Server: http://localhost:5001")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5001, debug=True)
