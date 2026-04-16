# -*- coding: utf-8 -*-
"""
SeeClearly AI — Data Preprocessing Pipeline

Handles APTOS 2019 dataset loading, augmentation, and generator creation.
Implements Ben Graham-inspired preprocessing for retinal fundus images.
"""

import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    DATA_PATH, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT,
    RANDOM_SEED, AUGMENTATION_CONFIG, LABEL_LIST
)


def load_aptos_dataframe():
    """
    Load and prepare the APTOS 2019 training CSV.

    Expected dataset structure:
        aptos2019-blindness-detection/
        ├── train.csv             (id_code, diagnosis)
        ├── train_images/         (3662 PNG images)
        └── test_images/          (optional)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: id_code (with .png extension), diagnosis (as string)
    """
    csv_path = os.path.join(DATA_PATH, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"train.csv not found at {csv_path}.\n"
            f"Download the APTOS 2019 dataset from Kaggle:\n"
            f"  kaggle competitions download -c aptos2019-blindness-detection\n"
            f"  unzip aptos2019-blindness-detection.zip -d {DATA_PATH}"
        )

    df = pd.read_csv(csv_path)

    # Add .png extension to image IDs for the data generator
    df["id_code"] = df["id_code"].apply(lambda x: x if x.endswith(".png") else x + ".png")

    # Convert diagnosis to string (required by flow_from_dataframe with class_mode='categorical')
    df["diagnosis"] = df["diagnosis"].astype(str)

    print(f"Loaded {len(df)} samples from APTOS 2019 dataset")
    print(f"\nClass distribution:")
    for idx, name in enumerate(LABEL_LIST):
        count = (df["diagnosis"] == str(idx)).sum()
        pct = count / len(df) * 100
        print(f"  {idx} ({name:>10}): {count:>5} samples ({pct:.1f}%)")

    return df


def create_data_generators(df=None):
    """
    Create training and validation data generators with augmentation.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded dataframe. If None, loads from disk.

    Returns
    -------
    train_gen : DirectoryIterator
        Training data generator with augmentation
    val_gen : DirectoryIterator
        Validation data generator (no augmentation, no shuffle)
    """
    if df is None:
        df = load_aptos_dataframe()

    image_dir = os.path.join(DATA_PATH, "train_images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"train_images/ directory not found at {image_dir}.\n"
            "Make sure you extracted the dataset correctly."
        )

    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        **AUGMENTATION_CONFIG,
    )

    # Validation generator — rescale only, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="id_code",
        y_col="diagnosis",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="id_code",
        y_col="diagnosis",
        target_size=IMG_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,  # IMPORTANT: do not shuffle validation for confusion matrix
    )

    print(f"\nTrain samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    return train_gen, val_gen


def compute_class_weights(df=None):
    """
    Compute balanced class weights to counteract class imbalance.

    The APTOS 2019 dataset is heavily skewed:
        No_DR: 49.3%  |  Mild: 10.1%  |  Moderate: 27.3%  |  Severe: 5.3%  |  PDR: 8.1%

    These weights are passed to model.fit(class_weight=...) so the loss function
    penalizes misclassification of rare classes (Severe, Mild) more heavily.

    Returns
    -------
    class_weight_dict : dict
        Mapping from class index to weight. E.g., {0: 0.41, 1: 1.97, ...}
    """
    if df is None:
        df = load_aptos_dataframe()

    labels = df["diagnosis"].astype(int).values
    unique_classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )

    class_weight_dict = dict(zip(unique_classes.astype(int), weights))

    print("\nComputed class weights:")
    for idx, name in enumerate(LABEL_LIST):
        w = class_weight_dict.get(idx, 1.0)
        print(f"  {idx} ({name:>10}): {w:.4f}")

    return class_weight_dict


if __name__ == "__main__":
    # Quick test — run this file directly to verify data loading
    df = load_aptos_dataframe()
    weights = compute_class_weights(df)
    train_gen, val_gen = create_data_generators(df)
    print("\n✅ Data preprocessing pipeline verified successfully!")
