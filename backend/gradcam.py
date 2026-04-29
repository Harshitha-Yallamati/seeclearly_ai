# -*- coding: utf-8 -*-
"""
Grad-CAM utilities for DR classification explainability.

This module generates both:
1. A raw heatmap image with the JET colormap applied
2. An alpha-blended overlay on top of the retinal image

It also masks out the dark background and outer retinal rim to reduce the
common circular "ring" artifact that appears when low-resolution CAMs are
resized back to the input image size.
"""

import base64
import os

import cv2
import numpy as np

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


DEFAULT_CAM_METHOD = os.getenv("DR_CAM_METHOD", "gradcam++").strip().lower()
DEFAULT_CAM_LAYERS = [
    "block6a_expand_activation",
    "block5e_expand_activation",
    "block6b_expand_activation",
    "block6c_expand_activation",
    "top_conv",
    "top_activation",
]


def _transparent_pixel():
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


def _ensure_uint8_rgb(img_np):
    if img_np is None:
        return None

    if img_np.dtype in (np.float32, np.float64):
        scale = 255.0 if np.max(img_np) <= 1.0 else 1.0
        return np.uint8(np.clip(img_np * scale, 0, 255))

    return np.uint8(np.clip(img_np, 0, 255))


def _encode_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", img_bgr)
    if not ok:
        return _transparent_pixel()
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _iter_layers(model):
    for layer in getattr(model, "layers", []):
        yield layer
        if hasattr(layer, "layers"):
            yield from _iter_layers(layer)


def _find_layer(model, layer_name):
    if not layer_name:
        return None

    try:
        return model.get_layer(layer_name)
    except Exception:
        pass

    for layer in _iter_layers(model):
        if layer.name == layer_name:
            return layer
    return None


def resolve_cam_layer(model, requested_layer=None):
    """
    Resolve the best Grad-CAM layer for the current model.

    Prefers an earlier high-resolution Xception stage such as
    block14_sepconv2_act, then falls back to the last 4D layer.
    """
    candidate_names = []
    if requested_layer:
        candidate_names.append(requested_layer)
    candidate_names.extend(
        name for name in DEFAULT_CAM_LAYERS if name not in candidate_names
    )

    for name in candidate_names:
        layer = _find_layer(model, name)
        output_shape = getattr(getattr(layer, "output", None), "shape", None)
        if layer is not None and output_shape is not None and len(output_shape) == 4:
            return layer.name

    last_4d_layer = None
    for layer in _iter_layers(model):
        output_shape = getattr(getattr(layer, "output", None), "shape", None)
        if output_shape is not None and len(output_shape) == 4:
            last_4d_layer = layer.name

    if last_4d_layer is None:
        raise ValueError("No 4D convolutional feature layer found for Grad-CAM.")

    return last_4d_layer


def _retinal_mask(img_uint8):
    """
    Estimate the fundus disc and shrink it slightly to suppress bright edge rims.
    """
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 8, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    min_dim = min(img_uint8.shape[:2])
    erode_size = max(5, (min_dim // 18) | 1)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_size, erode_size)
    )
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, min_dim / 60.0))
    return np.clip(soft_mask.astype(np.float32) / 255.0, 0.0, 1.0)


def _normalize_heatmap(heatmap, retinal_mask):
    heatmap = np.maximum(heatmap, 0).astype(np.float32)
    heatmap *= retinal_mask

    valid_region = retinal_mask > 0.05
    if np.any(valid_region):
        masked_values = heatmap[valid_region]
        hi = float(masked_values.max())
        lo = float(np.percentile(masked_values, 45))
        if hi > lo:
            heatmap = np.clip((heatmap - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        elif hi > 0:
            heatmap = heatmap / (hi + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)
    else:
        peak = float(heatmap.max())
        if peak > 0:
            heatmap = heatmap / peak

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=1.0)
    heatmap *= retinal_mask
    return np.clip(heatmap, 0.0, 1.0)


def _colorize_heatmap(heatmap):
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _blend_overlay(img_uint8, colored_heatmap, heatmap, retinal_mask, alpha=0.45):
    alpha_map = np.power(np.clip(heatmap, 0.0, 1.0), 0.85) * alpha
    alpha_map *= retinal_mask
    alpha_map = alpha_map[..., np.newaxis]

    overlay = (
        img_uint8.astype(np.float32) * (1.0 - alpha_map)
        + colored_heatmap.astype(np.float32) * alpha_map
    )
    return np.uint8(np.clip(overlay, 0, 255))


def _build_grad_model(model, cam_layer_name):
    cam_layer = _find_layer(model, cam_layer_name)
    if cam_layer is None:
        raise ValueError(f"Grad-CAM layer '{cam_layer_name}' was not found.")

    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[cam_layer.output, model.output],
    )


def _compute_cam(conv_outputs, grads, method):
    conv_outputs = conv_outputs[0]
    grads = grads[0]

    if method == "gradcam++":
        grads = tf.cast(grads, tf.float32)
        conv_outputs = tf.cast(conv_outputs, tf.float32)

        second = tf.square(grads)
        third = tf.pow(grads, 3.0)
        global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1), keepdims=True)
        alpha_denom = 2.0 * second + third * global_sum
        alpha_denom = tf.where(
            alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom)
        )
        alphas = second / (alpha_denom + 1e-7)
        weights = tf.reduce_sum(tf.nn.relu(grads) * alphas, axis=(0, 1))
        cam = tf.reduce_sum(conv_outputs * weights, axis=-1)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    cam = tf.nn.relu(cam)
    heatmap = cam.numpy()
    peak = float(heatmap.max()) if heatmap.size else 0.0
    if peak > 0:
        heatmap = heatmap / peak
    return heatmap.astype(np.float32)


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name=None,
    pred_index=None,
    method=DEFAULT_CAM_METHOD,
):
    """
    Compute a normalized Grad-CAM or Grad-CAM++ heatmap.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available.")

    cam_layer_name = resolve_cam_layer(model, last_conv_layer_name)
    grad_model = _build_grad_model(model, cam_layer_name)

    inputs = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Failed to compute gradients for Grad-CAM.")

    heatmap = _compute_cam(conv_outputs, grads, method=method)
    return heatmap, cam_layer_name


def _mock_heatmap_arrays(img_uint8):
    h, w = img_uint8.shape[:2]
    retinal_mask = _retinal_mask(img_uint8)
    heatmap = np.zeros((h, w), dtype=np.float32)

    num_spots = np.random.randint(2, 5)
    for _ in range(num_spots):
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.2), int(h * 0.8))
        radius = np.random.randint(max(8, w // 28), max(16, w // 10))
        cv2.circle(heatmap, (cx, cy), radius, 1.0, -1)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=max(4, w / 40.0))
    heatmap = _normalize_heatmap(heatmap, retinal_mask)
    colored = _colorize_heatmap(heatmap)
    overlay = _blend_overlay(img_uint8, colored, heatmap, retinal_mask)
    return colored, overlay


def generate_heatmap_assets(
    img_np=None,
    img_path=None,
    model=None,
    conv_layer=None,
    pred_index=None,
    method=DEFAULT_CAM_METHOD,
):
    """
    Generate explainability assets for the UI and API.

    Returns a dict with:
        heatmap: raw JET heatmap image (data URI)
        overlay: heatmap blended on the original image (data URI)
        conv_layer: actual layer used
        method: gradcam or gradcam++
        is_mock: whether explainability fell back to a synthetic heatmap
    """
    if img_np is None and img_path is not None:
        bgr = cv2.imread(img_path)
        if bgr is not None:
            img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if img_np is None:
        return {
            "heatmap": _transparent_pixel(),
            "overlay": _transparent_pixel(),
            "conv_layer": conv_layer,
            "method": method,
            "is_mock": True,
        }

    img_uint8 = _ensure_uint8_rgb(img_np)

    if not TF_AVAILABLE or model is None:
        heatmap_rgb, overlay_rgb = _mock_heatmap_arrays(img_uint8)
        return {
            "heatmap": _encode_to_base64(heatmap_rgb),
            "overlay": _encode_to_base64(overlay_rgb),
            "conv_layer": None,
            "method": "mock",
            "is_mock": True,
        }

    try:
        img_expanded = np.expand_dims(img_np, axis=0).astype(np.float32)
        raw_heatmap, used_layer = make_gradcam_heatmap(
            img_expanded,
            model,
            last_conv_layer_name=conv_layer,
            pred_index=pred_index,
            method=method,
        )
        raw_heatmap = cv2.resize(
            raw_heatmap,
            (img_uint8.shape[1], img_uint8.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        retinal_mask = _retinal_mask(img_uint8)
        normalized_heatmap = _normalize_heatmap(raw_heatmap, retinal_mask)
        heatmap_rgb = _colorize_heatmap(normalized_heatmap)
        overlay_rgb = _blend_overlay(
            img_uint8, heatmap_rgb, normalized_heatmap, retinal_mask
        )
        return {
            "heatmap": _encode_to_base64(heatmap_rgb),
            "overlay": _encode_to_base64(overlay_rgb),
            "conv_layer": used_layer,
            "method": method,
            "is_mock": False,
        }
    except Exception as exc:
        print(f"Warning: Grad-CAM failed ({exc}); using fallback heatmap.")
        heatmap_rgb, overlay_rgb = _mock_heatmap_arrays(img_uint8)
        return {
            "heatmap": _encode_to_base64(heatmap_rgb),
            "overlay": _encode_to_base64(overlay_rgb),
            "conv_layer": conv_layer,
            "method": "mock",
            "is_mock": True,
        }


def generate_heatmap_overlay(img_np=None, img_path=None, model=None, conv_layer=None):
    """
    Backward-compatible helper that returns only the overlay data URI.
    """
    assets = generate_heatmap_assets(
        img_np=img_np,
        img_path=img_path,
        model=model,
        conv_layer=conv_layer,
    )
    return assets["overlay"]


def get_medical_explanation(prediction_label, confidence):
    """
    Generate a short clinical explanation for the predicted DR stage.
    """
    explanations = {
        "No_DR": (
            "No signs of diabetic retinopathy were detected. The retinal image does "
            "not show clear microaneurysms, hemorrhages, or hard exudates."
        ),
        "Mild": (
            "Mild non-proliferative diabetic retinopathy was detected. Small "
            "microaneurysms may be visible in the retina."
        ),
        "Moderate": (
            "Moderate non-proliferative diabetic retinopathy was detected. The image "
            "suggests more widespread microaneurysms, hemorrhages, or exudates."
        ),
        "Severe": (
            "Severe non-proliferative diabetic retinopathy was detected. The retinal "
            "findings may include extensive hemorrhages, venous changes, or IRMA."
        ),
        "PDR": (
            "Proliferative diabetic retinopathy was detected. This stage can include "
            "new abnormal blood vessel growth and requires prompt specialist review."
        ),
    }

    text = explanations.get(prediction_label, "Unknown classification.")
    if confidence < 0.75:
        text += (
            f"\n\nLow confidence ({confidence * 100:.0f}%). Please treat this as a "
            "screening result and confirm it with an ophthalmologist."
        )
    return text
