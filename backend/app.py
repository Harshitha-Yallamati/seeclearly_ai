# -*- coding: utf-8 -*-
"""
Flask API backend for RetinoCheck.

Endpoints:
  GET  /health
  POST /predict
"""

import os
import sys
import traceback

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "utils"))

from dr_detection import predict_dr, get_model_status


app = Flask(__name__, static_folder=os.path.join(PROJECT_DIR, "dist"))
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    status = get_model_status()
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": status["model_loaded"],
            "model_path": status["model_path"],
            "mode": status["mode"],
            "tf_available": status["tf_available"],
            "load_error": status["load_error"],
            "cam_layer": status["cam_layer"],
            "cam_method": status["cam_method"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict DR severity from an uploaded retinal image.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send the file as form field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        allowed_list = ", ".join(sorted(allowed_extensions))
        return jsonify(
            {"error": f"Unsupported file type: {ext}. Allowed types: {allowed_list}"}
        ), 400

    try:
        result = predict_dr(file)
        if "error" in result:
            return jsonify(result), 422

        method_label = str(result.get("gradcam_method") or "gradcam").upper()
        layer_label = result.get("gradcam_layer") or "auto-selected feature layer"
        result["preprocessing"] = [
            "Decoded image and resized to 299x299 pixels",
            "Scaled pixels to the training-time range (0-1)",
            "Ran Xception inference and normalized class probabilities",
            f"Generated {method_label} explainability maps from {layer_label}",
        ]
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(PROJECT_DIR, "dist")
    if path and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    return send_from_directory(dist_dir, "index.html")


if __name__ == "__main__":
    status = get_model_status()
    print("=" * 50)
    print("  RetinoCheck Backend API")
    print("=" * 50)
    print(f"  Mode: {status['mode']}")
    print(f"  Model: {status['model_path']}")
    if status["load_error"]:
        print(f"  Load issue: {status['load_error']}")
    print(f"  Explainability: {status['cam_method']} @ {status['cam_layer']}")
    print("  Server: http://localhost:5001")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5001, debug=True)
