# -*- coding: utf-8 -*-
"""
SeeClearly AI — Comprehensive Model Evaluation

Generates:
  1. Confusion Matrix (heatmap PNG)
  2. Classification Report (Precision, Recall, F1-score per class)
  3. ROC-AUC curves (one-vs-rest, per class)
  4. Training history plots (accuracy + loss)

All outputs saved to training/outputs/

Usage:
  python evaluate.py                           # Evaluate saved model
  python evaluate.py --model path/to/model.keras  # Evaluate specific model
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

import tensorflow as tf
from tensorflow.keras.models import load_model

from config import MODEL_SAVE_PATH, OUTPUT_DIR, LABEL_LIST, NUM_CLASSES
from focal_loss import categorical_focal_loss
from data_preprocessing import load_aptos_dataframe, create_data_generators


def load_trained_model(model_path=None):
    """Load the trained model with custom focal loss."""
    path = model_path or MODEL_SAVE_PATH
    if not os.path.exists(path):
        print(f"❌ Model not found at: {path}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    model = load_model(
        path,
        custom_objects={"focal_loss_fixed": categorical_focal_loss()},
    )
    print(f"✅ Loaded model from: {path}")
    return model


def get_predictions(model, val_gen):
    """Run inference on the full validation set."""
    print("🔄 Running inference on validation set...")
    val_gen.reset()

    Y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(Y_pred_proba, axis=1)
    y_true = val_gen.classes

    print(f"   Predictions: {len(y_pred)} samples")
    return y_true, y_pred, Y_pred_proba


def plot_confusion_matrix(y_true, y_pred):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_LIST,
        yticklabels=LABEL_LIST,
        ax=ax,
        linewidths=0.5,
        square=True,
    )
    ax.set_ylabel("Actual Stage", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Stage", fontsize=13, fontweight="bold")
    ax.set_title("Diabetic Retinopathy — Confusion Matrix", fontsize=15, fontweight="bold")

    # Add accuracy annotation
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(
        0.5, -0.12,
        f"Overall Accuracy: {accuracy:.2%}",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        style="italic",
    )

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {save_path}")
    return cm


def generate_classification_report(y_true, y_pred):
    """Generate and save per-class precision, recall, F1-score."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_LIST,
        digits=4,
    )

    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(report)

    # Save to file
    save_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(save_path, "w") as f:
        f.write("SeeClearly AI — Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    print(f"   💾 Saved: {save_path}")

    return report


def plot_roc_curves(y_true, Y_pred_proba):
    """
    Generate per-class ROC-AUC curves (One-vs-Rest strategy).
    """
    # Binarize true labels for OVR
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]

    auc_scores = {}
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], Y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[LABEL_LIST[i]] = roc_auc

        ax.plot(
            fpr, tpr,
            color=colors[i],
            linewidth=2.5,
            label=f"{LABEL_LIST[i]} (AUC = {roc_auc:.3f})",
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title("ROC Curves — One-vs-Rest per DR Stage", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {save_path}")

    # Print AUC summary
    print("\n" + "=" * 60)
    print("  ROC-AUC Scores (One-vs-Rest)")
    print("=" * 60)
    for name, score in auc_scores.items():
        print(f"  {name:>10}: {score:.4f}")
    mean_auc = np.mean(list(auc_scores.values()))
    print(f"  {'Mean':>10}: {mean_auc:.4f}")

    return auc_scores


def plot_training_history(history_path=None):
    """
    Plot training history if a history dict is provided.
    Can also accept serialized history from a JSON file.
    """
    # Look for saved history
    import json
    hist_file = os.path.join(OUTPUT_DIR, "training_history.json")

    if history_path and os.path.exists(history_path):
        hist_file = history_path

    if not os.path.exists(hist_file):
        print("   ⚠️  No training history file found. Skipping history plots.")
        return

    with open(hist_file, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Accuracy
    if "accuracy" in history:
        axes[0].plot(history["accuracy"], label="Train", linewidth=2)
        axes[0].plot(history["val_accuracy"], label="Validation", linewidth=2)
        axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Loss
    if "loss" in history:
        axes[1].plot(history["loss"], label="Train", linewidth=2, color="orange")
        axes[1].plot(history["val_loss"], label="Validation", linewidth=2, color="red")
        axes[1].set_title("Focal Loss", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # AUC
    if "auc" in history:
        axes[2].plot(history["auc"], label="Train", linewidth=2, color="green")
        axes[2].plot(history["val_auc"], label="Validation", linewidth=2, color="teal")
        axes[2].set_title("AUC Score", fontsize=14, fontweight="bold")
        axes[2].set_ylabel("AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Saved: {save_path}")


def evaluate(model_path=None):
    """Run the complete evaluation pipeline."""
    print("=" * 60)
    print("  SeeClearly AI — Model Evaluation")
    print("=" * 60)

    # 1. Load model
    model = load_trained_model(model_path)

    # 2. Load validation data
    df = load_aptos_dataframe()
    _, val_gen = create_data_generators(df)

    # 3. Get predictions
    y_true, y_pred, Y_pred_proba = get_predictions(model, val_gen)

    # 4. Confusion Matrix
    print("\n📊 Generating Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred)

    # 5. Classification Report
    print("\n📋 Generating Classification Report...")
    generate_classification_report(y_true, y_pred)

    # 6. ROC-AUC Curves
    print("\n📈 Generating ROC-AUC Curves...")
    plot_roc_curves(y_true, Y_pred_proba)

    # 7. Training History
    print("\n📉 Generating Training History plots...")
    plot_training_history()

    print("\n" + "=" * 60)
    print("  ✅ Evaluation Complete!")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DR model")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    args = parser.parse_args()

    evaluate(args.model)
