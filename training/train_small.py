# -*- coding: utf-8 -*-
"""
SeeClearly AI — Small Dataset Training Script (Quick Test)
Modified version of train.py to use a subset of data for faster execution.
"""

import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# Import from the original training package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    IMG_SIZE, NUM_CLASSES, MODEL_SAVE_PATH,
    PHASE1_LR, PHASE2_LR,
    UNFREEZE_LAYERS, EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_MIN, FOCAL_ALPHA, FOCAL_GAMMA,
    LABEL_LIST
)
from focal_loss import categorical_focal_loss
from data_preprocessing import (
    load_aptos_dataframe, create_data_generators, compute_class_weights
)

# SMALL DATASET SETTINGS
SMALL_SAMPLE_SIZE = 100  # Number of samples to use
SMALL_EPOCHS_P1 = 2
SMALL_EPOCHS_P2 = 1

def check_gpu():
    """Check and report GPU availability."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[OK] {len(gpus)} GPU(s) detected:")
        for gpu in gpus:
            print(f"   {gpu.name}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("[WARN] No GPU detected. Training will be VERY slow on CPU.")
    return len(gpus) > 0

def build_model(num_classes=NUM_CLASSES):
    """Build EfficientNetB3 with custom classification head."""
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = BatchNormalization(name="bn_head")(x)
    x = Dense(512, activation="relu", name="dense_512")(x)
    x = Dropout(0.4, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    output = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=output, name="SeeClearlyAI_Small_EfficientNetB3")
    return model

def train_small():
    """Execute training on a small subset of data."""
    print("=" * 60)
    print("  SeeClearly AI - SMALL DATASET Training (Quick Test)")
    print("=" * 60)

    check_gpu()

    # 1. Load and subset data
    print(f"\n[INFO] Loading and subsetting data (n={SMALL_SAMPLE_SIZE})...")
    df_full = load_aptos_dataframe()
    
    # Stratified sample to maintain class distribution if possible
    # But for a very small set, simple sampling is fine for testing
    df = df_full.sample(n=min(SMALL_SAMPLE_SIZE, len(df_full)), random_state=42)
    
    class_weights = compute_class_weights(df)
    train_gen, val_gen = create_data_generators(df)

    # 2. Build model
    model = build_model()

    # PHASE 1
    print("\n" + "=" * 60)
    print(f"  PHASE 1: {SMALL_EPOCHS_P1} Epochs")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=categorical_focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=SMALL_EPOCHS_P1,
        class_weight=class_weights,
        verbose=1,
    )

    # PHASE 2
    print("\n" + "=" * 60)
    print(f"  PHASE 2: {SMALL_EPOCHS_P2} Epochs")
    print("=" * 60)

    for layer in model.layers[-UNFREEZE_LAYERS:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=categorical_focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=SMALL_EPOCHS_P2,
        class_weight=class_weights,
        verbose=1,
    )

    # Save small model with a different name to avoid overwriting production model
    small_model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "dr_small_model.keras")
    model.save(small_model_path)
    print(f"\n[INFO] Small model saved to: {small_model_path}")

if __name__ == "__main__":
    train_small()
    print("\n[OK] Quick training complete!")
