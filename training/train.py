# -*- coding: utf-8 -*-
"""
RetinoCheck — Complete Training Pipeline (Xception)

Single-phase training strategy using Transfer Learning:
  - Xception base (frozen, ImageNet weights)
  - Flatten + Dense(5, softmax) classification head

Usage (Google Colab / Kaggle):
  1. Upload this directory to Colab
  2. Ensure dataset is at the configured DATA_PATH
  3. Run: python train.py
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from config import (
    IMG_SIZE, NUM_CLASSES, MODEL_SAVE_PATH,
    PHASE1_EPOCHS, PHASE1_LR,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    LABEL_LIST, OUTPUT_DIR, LOSS_FUNCTION
)
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
    Build Xception with a simple classification head.

    Architecture (matches user's Kaggle notebook exactly):
        Xception (frozen, ImageNet weights)
        → Flatten
        → Dense(num_classes, softmax)

    Returns
    -------
    model : tf.keras.Model
    """
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    # Freeze all base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name="flatten_head")(x)
    output = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=output, name="RetinoCheck_Xception")

    print(f"\nModel built: {model.name}")
    print(f"  Total layers: {len(model.layers)}")
    print(f"  Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"  Non-trainable params: {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")

    return model


def get_callbacks():
    """Get training callbacks."""
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
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]
    return callbacks


def train():
    """
    Execute Xception training pipeline.

    Returns
    -------
    model : tf.keras.Model
    history : History
    val_gen : DirectoryIterator
    """
    print("=" * 60)
    print("  RetinoCheck — Diabetic Retinopathy Training (Xception)")
    print("=" * 60)

    # 1. Check GPU
    has_gpu = check_gpu()

    # 2. Load data
    print("\n📂 Loading dataset...")
    df = load_aptos_dataframe()
    train_gen, val_gen = create_data_generators(df)

    # 3. Build model
    print("\n🏗️  Building Xception model...")
    model = build_model()

    # 4. Compile
    print("\n⚡ Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=LOSS_FUNCTION,
        metrics=["accuracy"],
    )

    # 5. Train
    print("\n" + "=" * 60)
    print("  Training (Base Frozen)")
    print("=" * 60)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )

    print(f"\n✅ Training complete.")
    print(f"   Best val_accuracy: {max(history.history['val_accuracy']):.4f}")

    # 6. Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 Final model saved to: {MODEL_SAVE_PATH}")

    return model, history, val_gen


if __name__ == "__main__":
    model, h, val_gen = train()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
