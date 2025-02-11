import numpy as np
import skimage.feature as sf
import cv2
from sklearn.metrics import accuracy_score

def compute_glcm(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """
    Generate the Gray-Level Co-occurrence Matrix (GLCM) for a given grayscale image.

    GLCM captures the texture of an image by analyzing how often pixel intensity values 
    occur together at a certain distance and angle.
    """
    glcm = sf.greycomatrix(image, distances=distances, angles=angles, levels=levels,
                           symmetric=symmetric, normed=normed)
    return glcm

def mean_model_prediction(image, threshold=0.01):
    """
    Use a simple threshold on the GLCM mean value to classify an image.

    How it works:
    - Compute the GLCM matrix.
    - Calculate the mean value of the matrix.
    - If the mean value is higher than the threshold, classify as 1 (Positive).
    - Otherwise, classify as 0 (Negative).
    """
    glcm = compute_glcm(image)
    mean_value = glcm.mean()  # Average intensity of texture features

    return 1 if mean_value > threshold else 0  # Simple decision rule

def train_mean_model(X_train, Y_train, X_val, Y_val, size=128, patch_size=(32, 32), num_classes=2, batch_size=8, epochs=50, threshold=0.01):
    """
    Train a simple GLCM-based classifier using a mean-value threshold approach.

    This function applies the mean-based model to a dataset and evaluates its performance.

    """
    # Apply the mean model to all training images
    y_train_pred = [mean_model_prediction(img, threshold) for img in X_train]
    # Apply the mean model to all validation images
    y_val_pred = [mean_model_prediction(img, threshold) for img in X_val]

    # Compute accuracy scores for training and validation sets
    train_acc = accuracy_score(Y_train, y_train_pred)
    val_acc = accuracy_score(Y_val, y_val_pred)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    return val_acc
