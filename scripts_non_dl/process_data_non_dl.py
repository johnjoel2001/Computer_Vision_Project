import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import feature

# ✅ Configuration
CONFIGURATION = {
    "IMAGE_SIZE": 128,
}

# ✅ Define dataset path
DATASET_PATH = "data/raw/breakhis/BreaKHis_v1/"

# ✅ LBP Feature Extraction
def extract_lbp_features(image, num_points=24, radius=8):
    gray = rgb2gray(image)
    lbp = feature.local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ✅ Load Images & Extract Features
def load_data_lbp(dataframe):
    X, y = [], []
    for _, row in dataframe.iterrows():
        img_path = os.path.join(DATASET_PATH, row["path"])
        if os.path.exists(img_path):
            image = imread(img_path)
            image = resize(image, (CONFIGURATION["IMAGE_SIZE"], CONFIGURATION["IMAGE_SIZE"]))
            features = extract_lbp_features(image)
            X.append(features)
            y.append(row["class"])
    return np.array(X), np.array(y)

sample_img_path = r"C:\Users\USER\Documents\dl_cv\data\raw\breakhis\BreaKHis_v1\BreaKHis_v1\histology_slides\breast\malignant\SOB\lobular_carcinoma\SOB_M_LC_14-15570C\100X\SOB_M_LC-14-15570C-100-004.png"
from skimage.io import imread
from skimage.transform import resize


# ✅ Read & Process Image
image = imread(sample_img_path)
image_resized = resize(image, (128, 128))
features = extract_lbp_features(image_resized)

print(f"Extracted Features Shape: {features.shape}")  # Should be (26,)