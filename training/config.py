# -*- coding: utf-8 -*-
"""
SeeClearly AI — Training Configuration
All constants and paths for the training pipeline.
"""

import os

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "aptos2019-blindness-detection")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "dr_efficientnet_b3.keras")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# IMAGE / DATA
# ==========================================
IMG_SIZE = (224, 224)  # EfficientNetB3 optimal input size
BATCH_SIZE = 32
NUM_CLASSES = 5
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# ==========================================
# TRAINING
# ==========================================
# Phase 1: Train custom head (base frozen)
PHASE1_EPOCHS = 20
PHASE1_LR = 1e-3

# Phase 2: Fine-tune top layers
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-5
UNFREEZE_LAYERS = 30  # Number of top layers to unfreeze

# Callbacks
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# Focal Loss
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ==========================================
# LABELS
# ==========================================
LABEL_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "PDR"
}
LABEL_LIST = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]

# ==========================================
# AUGMENTATION
# ==========================================
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "zoom_range": 0.15,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "fill_mode": "constant",
    "cval": 0,
}
