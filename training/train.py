# -*- coding: utf-8 -*-
"""
SeeClearly AI — Complete Training Pipeline

Two-phase training strategy:
  Phase 1: Train custom classification head with frozen EfficientNetB3 base
  Phase 2: Fine-tune top layers of the base model with very low learning rate

Usage (Google Colab / Kaggle):
  1. Upload this directory to Colab
  2. Ensure APTOS 2019 dataset is at the configured DATA_PATH
  3. Run: python train.py
"""

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from config import (
    IMG_SIZE, NUM_CLASSES, MODEL_SAVE_PATH,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    UNFREEZE_LAYERS, EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_MIN, FOCAL_ALPHA, FOCAL_GAMMA,
    LABEL_LIST, OUTPUT_DIR
)
from focal_loss import categorical_focal_loss
from data_preprocessing import (
    load_aptos_dataframe, create_data_generators, compute_class_weights
)


def check_gpu():
    """Check and report GPU availability."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) detected:")
        for gpu in gpus:
            print(f"   {gpu.name}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("⚠️  No GPU detected. Training will be VERY slow on CPU.")
        print("   Recommended: Use Google Colab (Runtime > Change runtime type > T4 GPU)")
    return len(gpus) > 0


def build_model(num_classes=NUM_CLASSES):
    """
    Build EfficientNetB3 with custom classification head.

    Architecture:
        EfficientNetB3 (frozen, ImageNet weights)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(512, relu)
        → Dropout(0.4)
        → Dense(256, relu)
        → Dropout(0.3)
        → Dense(num_classes, softmax)

    Returns
    -------
    model : tf.keras.Model
    """
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    # Freeze all base model layers for Phase 1
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

    model = Model(inputs=base_model.input, outputs=output, name="SeeClearlyAI_EfficientNetB3")

    print(f"\nModel built: {model.name}")
    print(f"  Total layers: {len(model.layers)}")
    print(f"  Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"  Non-trainable params: {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")

    return model


def get_callbacks(phase="phase1"):
    """Get training callbacks for a given phase."""
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]
    return callbacks


def train():
    """
    Execute the full two-phase training pipeline.

    Returns
    -------
    model : tf.keras.Model
    history_phase1 : History
    history_phase2 : History
    val_gen : DirectoryIterator
    """
    print("=" * 60)
    print("  SeeClearly AI — Diabetic Retinopathy Training Pipeline")
    print("=" * 60)

    # 1. Check GPU
    has_gpu = check_gpu()

    # 2. Load data
    print("\n📂 Loading APTOS 2019 dataset...")
    df = load_aptos_dataframe()
    class_weights = compute_class_weights(df)
    train_gen, val_gen = create_data_generators(df)

    # 3. Build model
    print("\n🏗️  Building EfficientNetB3 model...")
    model = build_model()

    # =============================================
    # PHASE 1: Train custom head (base frozen)
    # =============================================
    print("\n" + "=" * 60)
    print("  PHASE 1: Training Custom Head (Base Frozen)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=categorical_focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    history_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    print(f"\n✅ Phase 1 complete.")
    print(f"   Best val_accuracy: {max(history_p1.history['val_accuracy']):.4f}")
    print(f"   Best val_auc: {max(history_p1.history['val_auc']):.4f}")

    # =============================================
    # PHASE 2: Fine-tune top layers
    # =============================================
    print("\n" + "=" * 60)
    print(f"  PHASE 2: Fine-Tuning Top {UNFREEZE_LAYERS} Layers")
    print("=" * 60)

    # Unfreeze top layers (skip BatchNormalization layers)
    for layer in model.layers[-UNFREEZE_LAYERS:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"  Trainable params after unfreeze: {trainable_count:,}")

    # Recompile with much lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=categorical_focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    history_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    print(f"\n✅ Phase 2 complete.")
    print(f"   Best val_accuracy: {max(history_p2.history['val_accuracy']):.4f}")
    print(f"   Best val_auc: {max(history_p2.history['val_auc']):.4f}")

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Final model saved to: {MODEL_SAVE_PATH}")

    return model, history_p1, history_p2, val_gen


if __name__ == "__main__":
    model, h1, h2, val_gen = train()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print(f"  Run 'python evaluate.py' to generate evaluation metrics.")
