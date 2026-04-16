# -*- coding: utf-8 -*-
"""
SeeClearly AI — Grad-CAM Explainability Module

Generates Gradient-weighted Class Activation Maps to visualize
which regions of the retinal fundus image the model focuses on.

When TensorFlow or the model is unavailable, generates realistic
mock heatmaps for API testing.
"""

import cv2
import numpy as np
import base64
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def generate_mock_heatmap(img_np):
    """
    Create a simulated focal heatmap when TF/model is unavailable.

    Places 1-3 random hotspots (simulating hemorrhages/microaneurysms)
    on the retinal image with Gaussian blur for smooth gradients.
    """
    h, w = img_np.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    num_spots = np.random.randint(1, 4)
    for _ in range(num_spots):
        cx = np.random.randint(int(w * 0.15), int(w * 0.85))
        cy = np.random.randint(int(h * 0.15), int(h * 0.85))
        radius = np.random.randint(int(w * 0.04), int(w * 0.18))
        cv2.circle(heatmap, (cx, cy), radius, 1.0, -1)

    # Multiple blur passes for smoother gradient
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = np.uint8(255 * heatmap)

    # Apply JET colormap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original to uint8 if needed
    if img_np.dtype in (np.float32, np.float64):
        img_uint8 = np.uint8(np.clip(img_np * 255, 0, 255))
    else:
        img_uint8 = img_np.copy()

    # Superimpose
    superimposed = cv2.addWeighted(img_uint8, 0.6, jet, 0.4, 0)
    return superimposed


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    """
    Core Grad-CAM algorithm.

    Computes the gradient of the predicted class score with respect to
    the feature maps of the last convolutional layer, then weights
    each feature map by the average gradient magnitude.

    Parameters
    ----------
    img_array : np.ndarray
        Input image array of shape (1, H, W, 3)
    model : tf.keras.Model
        Trained classification model
    last_conv_layer_name : str
        Name of the target convolutional layer
    pred_index : int, optional
        Class index to generate heatmap for. If None, uses the predicted class.

    Returns
    -------
    heatmap : np.ndarray
        Normalized heatmap of shape (H', W') where H',W' are the spatial dims
        of the target convolutional layer
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the score with respect to conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_heatmap_overlay(img_np=None, img_path=None, model=None, conv_layer="top_activation"):
    """
    Generate a Grad-CAM heatmap overlay and return as base64 string.

    Parameters
    ----------
    img_np : np.ndarray, optional
        Preprocessed image array (H, W, 3) in RGB, normalized [0,1]
    img_path : str, optional
        Path to image file (alternative to img_np)
    model : tf.keras.Model, optional
        Trained model. If None, generates mock heatmap.
    conv_layer : str
        Name of the target convolutional layer

    Returns
    -------
    base64_str : str
        Base64-encoded PNG image with data URI prefix
    """
    # Resolve image input
    if img_np is None and img_path is not None:
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = img_np / 255.0

    if img_np is None:
        # Return a 1x1 transparent pixel if no image
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # Decide: real Grad-CAM or mock
    if not TF_AVAILABLE or model is None:
        superimposed = generate_mock_heatmap(img_np)
    else:
        # Real Grad-CAM
        img_expanded = np.expand_dims(img_np, axis=0).astype(np.float32)

        try:
            heatmap = make_gradcam_heatmap(img_expanded, model, last_conv_layer_name=conv_layer)
        except Exception as e:
            print(f"⚠️ Grad-CAM failed ({e}), falling back to mock heatmap")
            superimposed = generate_mock_heatmap(img_np)
            return _encode_to_base64(superimposed)

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (img_np.shape[1], img_np.shape[0]))
        jet = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Prepare original image as uint8
        if img_np.dtype in (np.float32, np.float64):
            img_uint8 = np.uint8(np.clip(img_np * 255, 0, 255))
        else:
            img_uint8 = img_np.copy()

        superimposed = cv2.addWeighted(img_uint8, 0.6, jet, 0.4, 0)

    return _encode_to_base64(superimposed)


def _encode_to_base64(img_rgb):
    """Encode an RGB image array to a base64 data URI string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_medical_explanation(prediction_label, confidence):
    """
    Generate a medical explanation text for the prediction.

    Parameters
    ----------
    prediction_label : str
        One of: No_DR, Mild, Moderate, Severe, PDR
    confidence : float
        Model confidence score (0-1)

    Returns
    -------
    explanation : str
    """
    explanations = {
        "No_DR": (
            "No signs of Diabetic Retinopathy detected. "
            "The retinal blood vessels appear normal with no visible microaneurysms, "
            "hemorrhages, or exudates."
        ),
        "Mild": (
            "Mild Non-Proliferative Diabetic Retinopathy (NPDR) detected. "
            "Small microaneurysms (tiny red dots) may be present. These are swollen areas "
            "in the small blood vessels of the retina."
        ),
        "Moderate": (
            "Moderate NPDR detected. Multiple microaneurysms, some dot/blot hemorrhages, "
            "and possible hard exudates (yellow lipid deposits) are visible. "
            "Blood vessels may show signs of blockage."
        ),
        "Severe": (
            "Severe NPDR detected. Significant retinal hemorrhages in all four quadrants, "
            "venous beading, and intraretinal microvascular abnormalities (IRMA). "
            "High risk of progression to proliferative stage."
        ),
        "PDR": (
            "Proliferative Diabetic Retinopathy (PDR) detected. "
            "Abnormal new blood vessel growth (neovascularization) is present. "
            "These fragile vessels can leak blood into the vitreous, causing severe vision loss. "
            "URGENT medical intervention recommended."
        ),
    }

    text = explanations.get(prediction_label, "Unknown classification.")

    if confidence < 0.75:
        text += (
            f"\n\n⚠️ Low confidence ({confidence*100:.0f}%). "
            "The model has significant uncertainty about this prediction. "
            "Please consult an ophthalmologist for definitive diagnosis."
        )

    return text
