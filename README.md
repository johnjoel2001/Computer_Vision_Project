# Breast Cancer Detection using BreakHis

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Approaches](#approaches)
  - [Naive Approach](#naive-approach)
  - [Machine Learning Approach](#machine-learning-approach)
  - [Deep Learning Approach (Hybrid Model Architecture)](#deep-learning-approach-hybrid-model-architecture)
    - [1. High-Level Overview](#1-high-level-overview)
    - [2. Squeeze-and-Excitation Mechanism](#2-squeeze-and-excitation-mechanism)
    - [3. Depthwise-Separable Convolution](#3-depthwise-separable-convolution)
    - [4. Pretrained EfficientNetV2B0 Bottleneck](#4-pretrained-efficientnetv2b0-bottleneck)
    - [5. Neck Section](#5-neck-section)
    - [6. Vision Transform Patch-Based Branch](#6-vision-transform-patch-based-branch)
    - [7. Adaptive Branch Fusion](#7-adaptive-branch-fusion)
    - [8. Final Classification](#8-final-classification)
    - [9. Model Construction](#9-model-construction)
    - [10. Training Process](#10-training-process)
- [Installation and Usage](#installation-and-usage)

---

## Introduction
This project analyzes histopathology images of breast tumor tissue to determine whether a sample is benign or malignant. It uses the [BreakHis dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and demonstrates different classification techniques, including simple thresholding methods, traditional machine learning models, and more advanced deep learning architectures.

The workflow diagram below (created using Mermaid) illustrates the complete project workflow. Since the model files were too large, we stored them in an S3 bucket and load them in real time for real-time inference on our Streamlit app. In the app, a user can upload an image, select a model, and run the inference.

![Workflow](https://github.com/user-attachments/assets/0f83cb5e-9e0b-4b39-984c-071c91d717ff)


---

## Dataset
The **Breast Cancer Histopathological Image Classification ([BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/))** dataset contains **9,109** microscopic images of breast tumor tissue collected from **82** patients. These images are provided at magnification levels of **40X**, **100X**, **200X**, and **400X**, each with a resolution of **700×460 pixels** in **PNG** format. There are **2,480** benign and **5,429** malignant samples.

The scripts **make_dataset.py** and **process_dataset.py** handle dataset loading and processing. They create directories for augmented and processed data, gather all PNG images, and separate them into benign and malignant folders. To address class imbalance, the pipeline upsamples the minority class (benign) until the classes are balanced.

All images are then resized to **128×128** pixels and normalized to the [0,1] range. The dataset is split into training, validation, and test sets using 10-fold cross-validation, and labels are one-hot encoded for classification. Data augmentation (rotation, zoom, shifting, flipping, brightness adjustments) is selectively applied to enhance model generalization. Finally, the processed arrays are saved in NumPy format.

---

## Approaches

### Naive Approach
1. **Feature Extraction**: Use a texture descriptor **GLCM (Gray Level Co-occurrence Matrix)** to extract basic features.
2. **Simple Thresholding**: Apply a threshold to these extracted features to classify images as benign or malignant.

This approach offers a straightforward baseline.

### Machine Learning Approach
The scripts **process_data_non_dl.py** and **non-dl.py** build a pipeline for classifying histopathological images into benign and malignant categories. They load a dataset from a CSV file, filter images at 40x magnification, and extract labels from file paths. The data is split into training, validation, and test sets using stratified sampling. 

The pipeline uses Local Binary Pattern (LBP) features to capture texture information. These features are then fed into four different machine learning models: Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Logistic Regression. The best-performing model is chosen based on accuracy, and each trained model is saved as a `.pkl` file.

---

## Deep Learning Approach (Hybrid Model Architecture)

### 1. High-Level Overview
This hybrid architecture processes input images through three parallel branches:
1. **Pretrained EfficientNetV2B0**: Extracts high-level features from a frozen, pretrained backbone.
2. **Vision Transform (Patch-Based)**: Splits the input into patches and applies depthwise-separable convolutions.
3. **Neck Section**: Comprises depthwise-separable convolutions, pooling, and squeeze-and-excitation.

All outputs are fused by an **Adaptive Branch Fusion** mechanism that learns how to weight each branch’s contribution before concatenation. A final Dense layer provides class probabilities.

### 2. Squeeze-and-Excitation Mechanism
Squeeze-and-Excitation adaptively recalibrates each channel’s importance:
1. **Global Average Pooling** generates a single value per feature map.
2. **Dense layers** (squeeze → excite) produce a set of channel-wise scaling factors in [0,1].
3. **Channel-wise Multiplication** scales each channel accordingly.

### 3. Depthwise-Separable Convolution
Convolution is split into:
1. **Depthwise Convolution**: Each filter applies to one channel at a time.
2. **Pointwise Convolution**: Combines channel-wise outputs with 1×1 kernels.
This reduces computation while preserving crucial spatial information.

### 4. Pretrained EfficientNetV2B0 Bottleneck
1. **Load Pretrained Model** with ImageNet weights.
2. **Freeze Layers** to retain pretrained weights.
3. **Global Average Pooling** to create a feature vector.
4. **Batch Normalization + Dense (256 units)** to produce a compact representation for fusion.

### 5. Neck Section
1. **Depthwise-Separable Convolution** for efficient spatial feature extraction.
2. **Max Pooling** to reduce spatial dimensions.
3. **Squeeze-and-Excitation** to emphasize important channels.
4. **Flatten** to create a 1D feature vector.

### 6. Vision Transform Patch-Based Branch
1. **Reshape** the input into smaller patches.
2. **Normalization** to [0,1].
3. **Depthwise-Separable Convolution + Pooling** to extract features locally.
4. **Squeeze-and-Excitation** for channel-wise recalibration.
5. **Flatten** for the final output of this branch.

### 7. Adaptive Branch Fusion
1. **Gate Generation**: A small subnetwork outputs a scalar gate for each branch.
2. **Weighted Features**: Multiply each branch’s output by its gate.
3. **Concatenate** the gated feature vectors.

### 8. Final Classification
A Dense layer with softmax activation converts the fused feature vector into class probabilities.

### 9. Model Construction
1. **Input** shape: 128×128×3 (example).
2. **Branches** for:
   - Pretrained EfficientNetV2B0
   - Neck Section
   - Patch-Based Transform
3. **Adaptive Fusion** to combine all branch outputs.
4. **Output**: Softmax classification layer.

### 10. Training Process
1. **Distribution Strategy** for multi-GPU or multi-CPU.
2. **Compile** with an optimizer (Adam), a suitable loss function, and relevant metrics.
3. **Callbacks** for early stopping, monitoring validation performance.
4. **Fit** on training data and validate on a separate split.
5. **Model Saving** in TensorFlow’s SavedModel format.

---

## Explainability

To enhance the novelty of our approach, we incorporated interpretability techniques such as LIME (Local Interpretable Model-agnostic Explanations) and Integrated Gradients to gain a deeper understanding of which pixels contribute most significantly to the final decision-making process of our model.

LIME helps by approximating the model’s behavior locally, generating perturbations of the input and analyzing how predictions change in response to those modifications. This enables us to identify which regions of the image are truly influencing the model’s output. Similarly, Integrated Gradients attribute importance to input features by computing the gradients of the prediction with respect to the input, providing a more holistic view of how different pixels contribute to the decision.

Through these techniques, we discovered an intriguing phenomenon: even the best-performing models occasionally assigned importance to random background pixels, suggesting potential biases or spurious correlations within the model’s learning process. This highlighted the importance of interpretability tools in evaluating model reliability.

By leveraging these techniques, we ensure greater accountability in our approach. Rather than treating model decisions as black-box outputs, we can scrutinize how and why the model reaches its conclusions. This is particularly crucial in high-stakes applications, where understanding model reasoning can prevent erroneous or biased decision-making.

---

## Usage

Streamlit App Link: 

---

## Ethics Statement

We recognize the importance of safeguarding patient privacy and ensuring responsible use of medical data. All histopathology images in this project come from publicly available, anonymized sources, and they are used strictly for research and educational purposes. This work is not a substitute for clinical diagnosis or professional medical advice. Medical practitioners should rely on clinical expertise, laboratory testing, and standard diagnostic procedures when making any healthcare decisions. Our goal is to advance understanding of breast cancer detection methods in a manner that respects patient dignity, maintains scientific rigor, and adheres to ethical guidelines.




