# -*- coding: utf-8 -*-
"""
RetinoCheck — Shared Image Preprocessing Utilities

Functions shared between training and backend for consistent image processing.
"""

import cv2
import numpy as np

IMG_SIZE = (224, 224)


def load_and_preprocess(image_path, target_size=IMG_SIZE):
    """
    Load an image from disk and preprocess for model input.

    Parameters
    ----------
    image_path : str
        Path to the image file
    target_size : tuple
        Target (width, height) for resizing

    Returns
    -------
    img : np.ndarray
        Preprocessed RGB image normalized to [0, 1], shape (H, W, 3)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_from_bytes(file_bytes, target_size=IMG_SIZE):
    """
    Preprocess an image from raw bytes (e.g., from a Flask file upload).

    Parameters
    ----------
    file_bytes : bytes
        Raw image file bytes
    target_size : tuple
        Target (width, height) for resizing

    Returns
    -------
    img : np.ndarray or None
        Preprocessed image or None if decoding fails
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def apply_ben_graham_preprocessing(img, sigmaX=10):
    """
    Ben Graham's preprocessing technique (Kaggle winning solution).

    Subtracts a Gaussian-blurred version of the image to enhance
    local contrast and remove uneven illumination — critical for
    retinal fundus images with varying lighting conditions.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (uint8)
    sigmaX : int
        Gaussian blur sigma

    Returns
    -------
    processed : np.ndarray
        Contrast-enhanced image (uint8)
    """
    # Additive weighted blend: original * 4 - gaussian_blur * 4 + 128
    processed = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128,
    )
    return processed


def crop_circle(img, tolerance=10):
    """
    Crop the circular retinal region from a fundus image.

    Many fundus images have black borders around the circular retina.
    This function detects and crops to the circular region.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or RGB)
    tolerance : int
        Pixel intensity threshold for detecting the retina boundary

    Returns
    -------
    cropped : np.ndarray
        Cropped image containing only the retinal region
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    mask = gray > tolerance

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = img[y0:y1, x0:x1]
    return cropped
