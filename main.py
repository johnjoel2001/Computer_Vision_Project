"""
    This is the script to run the streamlit application
    """
import streamlit as st
import os
import numpy as np
import joblib
import lime
import lime.lime_image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
import boto3
import io

from skimage.io import imread
from skimage.transform import resize
from skimage.segmentation import mark_boundaries, slic

class S3ModelLoader:
    """
    Handles downloading models from AWS S3.
    """
    def __init__(self, aws_access_key, aws_secret_key, aws_region, bucket_name):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        self.bucket_name = bucket_name

    def download_model(self, model_key, local_path):
        """Downloads the model from S3 to a local path."""
        self.s3.download_file(self.bucket_name, model_key, local_path)


class HybridModel:
    """
    Defines the Deep Learning hybrid model architecture and loads pre-trained weights.
    """
    def __init__(self, input_shape=(128, 128, 3), num_classes=2, patch_size=(32, 32), model_path=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.model = self.build_hybrid_model()
        if model_path:
            self.model.load_weights(model_path)

    def squeeze_excitation_block(self, input_tensor, reduction_ratio=16):
        """Implements a Squeeze-and-Excitation (SE) block."""
        channel_dim = int(input_tensor.shape[-1])
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Dense(channel_dim // reduction_ratio, activation='relu')(se)
        se = layers.Dense(channel_dim, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, channel_dim))(se)
        return layers.Multiply()([input_tensor, se])

    def depthwise_separable_conv(self, x, filters, kernel_size, strides=(1, 1), padding='same'):
        """Implements a depthwise-separable convolution layer."""
        x = layers.SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def bottleneck0(self, inputs):
        """Creates a bottleneck feature extractor using EfficientNetV2B0."""
        backbone = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False)
        for layer in backbone.layers:
            layer.trainable = False
        x = backbone(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        return x

    def neck_section(self, inputs):
        """Defines the neck section of the model with SE blocks and pooling layers."""
        x = self.depthwise_separable_conv(inputs, filters=256, kernel_size=7, strides=(2, 2), padding='same')
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = self.squeeze_excitation_block(x)
        x = layers.Flatten()(x)
        return x

    def vision_transform(self, inputs, patch_size):
        """Defines the vision transformer branch of the model."""
        x = layers.Reshape((patch_size[0], patch_size[1], -1))(inputs)
        x = layers.Lambda(lambda image: K.cast(image, 'float32') / 255.0)(x)
        x = self.depthwise_separable_conv(x, 64, kernel_size=3)
        x = self.depthwise_separable_conv(x, 64, kernel_size=3)
        x = layers.MaxPooling2D((2, 2))(x)
        x = self.depthwise_separable_conv(x, 128, kernel_size=3)
        x = self.depthwise_separable_conv(x, 128, kernel_size=3)
        x = layers.MaxPooling2D((2, 2))(x)
        x = self.squeeze_excitation_block(x)
        x = layers.Flatten()(x)
        return x

    def adaptive_branch_fusion(self, *branches, hidden_dim=128):
        """Performs adaptive weighting of multiple model branches."""
        weighted_branches = []
        for branch in branches:
            gate = layers.Dense(hidden_dim, activation='relu')(branch)
            gate = layers.Dense(1, activation='sigmoid')(gate)
            weighted = layers.Multiply()([branch, gate])
            weighted_branches.append(weighted)
        return layers.Concatenate()(weighted_branches)

    def build_hybrid_model(self):
        """Builds the complete hybrid model architecture."""
        inputs = layers.Input(shape=self.input_shape)
        efficientnet_features = self.bottleneck0(inputs)
        patches_features = self.vision_transform(inputs, self.patch_size)
        neck_features = self.neck_section(inputs)
        fused = self.adaptive_branch_fusion(efficientnet_features, patches_features, neck_features, hidden_dim=64)
        outputs = layers.Dense(self.num_classes, activation='softmax')(fused)
        return models.Model(inputs=inputs, outputs=outputs)

    def classify_image(self, image):
        """Classifies an image using the trained deep learning model."""
        img = self.preprocess_image(image)
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        return "Malignant (Yes)" if predicted_class == 1 else "Benign (No)"

    def preprocess_image(self, image, img_size=(128, 128)):
        """Preprocesses an image before classification."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)


# Initialize S3 Model Loader
s3_loader = S3ModelLoader(
    aws_access_key="your_access_key",
    aws_secret_key="your_secret_key",
    aws_region="us-east-1",
    bucket_name="dlmodel540"
)
s3_loader.download_model("final_hybrid_model.h5", "final_hybrid_model.h5")

# Load Deep Learning Model
dl_model = HybridModel(model_path="final_hybrid_model.h5")

st.title("Breast Cancer Classification")
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose a model type:", ["Deep Learning Model"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = imread(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = dl_model.classify_image(image)
    st.success(f"Prediction: {prediction}")
