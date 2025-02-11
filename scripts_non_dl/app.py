import streamlit as st
import os
import numpy as np
import joblib
import lime
import lime.lime_image
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.segmentation import mark_boundaries, slic
from process_data_non_dl import extract_lbp_features  # Import LBP feature extractor



# ✅ Load trained model
MODEL_PATH = "models/RandomForest_breakhis.pkl"
model = joblib.load(MODEL_PATH)

# ✅ Function to preprocess image and extract LBP features
def process_image(image):
    image_resized = resize(image, (128, 128))  # Resize to required size
    features = extract_lbp_features(image_resized)
    return np.array(features).reshape(1, -1)  # Convert to 2D array

# ✅ Function to predict using model
def classify_image(image):
    features = process_image(image)
    prediction = model.predict(features)[0]
    return "Malignant (Yes)" if prediction == 1 else "Benign (No)"

# ✅ **Better LIME prediction function**
def model_predict(image_batch):
    processed_features = []
    for img in image_batch:
        img_resized = resize(img, (128, 128))
        features = extract_lbp_features(img_resized)
        processed_features.append(features)
    
    return model.predict_proba(np.array(processed_features))  # ✅ LIME now passes LBP features!

# ✅ **Improved LIME Explanation**
def generate_lime_explanation(image):
    explainer = lime.lime_image.LimeImageExplainer()

    # ✅ Use SLIC segmentation with more superpixels
    segmentation_fn = lambda x: slic(x, n_segments=300, compactness=15, sigma=1)  

    explanation = explainer.explain_instance(
        image, 
        model_predict, 
        top_labels=2, 
        hide_color=0, 
        num_samples=1000,
        segmentation_fn=segmentation_fn  # Use improved segmentation
    )

    # ✅ **Extract important positive regions**
    temp_pos, mask_pos = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
    )

    # ✅ **Extract both positive & negative regions**
    temp_neg, mask_neg = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
    )

    return temp_pos, mask_pos, temp_neg, mask_neg

# ✅ Streamlit App Layout
st.title("🩺 Breast Cancer Classification with LIME Explainability")

# Upload Image Section
uploaded_file = st.file_uploader("Upload a Histopathological Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = imread(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Predict Classification
    prediction = classify_image(image)
    st.subheader(f"Prediction: **{prediction}**")

    # ✅ Generate LIME explanation
    with st.spinner("Generating LIME Explainability..."):
        temp_pos, mask_pos, temp_neg, mask_neg = generate_lime_explanation(image)

    # ✅ Display LIME Explanation with **Improved Masks**
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ✅ Important Regions Only
    axes[0].imshow(mark_boundaries(temp_pos, mask_pos, color=(1, 1, 0), mode='thick'))  # Yellow boundaries
    axes[0].axis("off")
    axes[0].set_title("LIME: Important Regions", fontsize=12)

    # ✅ Positive & Negative Regions
    axes[1].imshow(mark_boundaries(temp_neg, mask_neg, color=(1, 1, 0), mode='thick'))  # Yellow boundaries
    axes[1].axis("off")
    axes[1].set_title("LIME: Positive & Negative Regions", fontsize=12)

    st.pyplot(fig)  # Show the plot in Streamlit


