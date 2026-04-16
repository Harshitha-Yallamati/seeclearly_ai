# -*- coding: utf-8 -*-
"""
SeeClearly AI — Categorical Focal Loss

Focal loss down-weights well-classified examples and focuses training
on hard, misclassified samples. Critical for the heavily imbalanced
APTOS 2019 dataset where Severe DR is only 5.3% of all samples.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Categorical focal loss for multi-class classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values give more weight to hard examples.
        - gamma=0: equivalent to standard cross-entropy
        - gamma=2: recommended default (from original paper)
        - gamma=5: very aggressive focus on hard examples

    alpha : float
        Balancing factor. Prevents the model from being overwhelmed by
        easy negatives.

    Returns
    -------
    focal_loss_fixed : callable
        Loss function compatible with Keras model.compile()
    """

    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0) = NaN
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Standard cross-entropy component
        cross_entropy = -y_true * K.log(y_pred)

        # Focal modulation: (1 - p_t)^gamma
        # This term is close to 1 for misclassified examples (p_t near 0)
        # and close to 0 for well-classified examples (p_t near 1)
        focal_weight = alpha * K.pow(1 - y_pred, gamma)

        # Final focal loss
        loss = focal_weight * cross_entropy

        return K.mean(K.sum(loss, axis=-1))

    return focal_loss_fixed
