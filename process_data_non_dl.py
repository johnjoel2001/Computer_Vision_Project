import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import feature

# Configuration
CONFIGURATION = {
    "IMAGE_SIZE": 128,
}


# LBP Feature Extraction
def extract_lbp_features(image, num_points=24, radius=8):
    gray = rgb2gray(image)
    lbp = feature.local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist