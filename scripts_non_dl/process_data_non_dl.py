import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import feature

class LBPFeatureExtractor:
    """
    A class for extracting Local Binary Pattern (LBP) features from images.
    """
    
    def __init__(self, image_size=128, num_points=24, radius=8):
        """
        Initializing  the feature extractor with configurable parameters.
        
        Parameters:
        image_size (int): The target image size for resizing.
        num_points (int): Number of circularly symmetric points for LBP.
        radius (int): Radius of LBP circular neighborhood.
        """
        self.image_size = image_size
        self.num_points = num_points
        self.radius = radius
    
    def extract_lbp_features(self, image):
        """
        Extracting LBP features from a given image.
        
        Parameters:
        image (ndarray): Input image array.
        
        Returns:
        ndarray: Normalized LBP histogram features.
        """
        gray = rgb2gray(image)  # Converting image to grayscale
        lbp = feature.local_binary_pattern(gray, self.num_points, self.radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))
        hist = hist.astype("float") / (hist.sum() + 1e-6)  # Normalizing histogram
        return hist
    
    def process_image(self, img_path):
        """
        Loaing  and processesinh a single image.
        
        Parameters:
        img_path (str): Path to the image file.
        
        Returns:
        ndarray: Extracted LBP features for the image.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = imread(img_path)  # Reading a image
        image_resized = resize(image, (self.image_size, self.image_size))  # Resizing a image
        return self.extract_lbp_features(image_resized)
    
    def load_data_from_dataframe(self, dataframe, dataset_path):
        """
        Loading a  images and extracts LBP features from a dataset given a dataframe containing image paths and labels.
        
        Parameters:
        dataframe (pd.DataFrame): Dataframe containing image paths and class labels.
        dataset_path (str): Root directory of the dataset.
        
        Returns:
        tuple: (features array, labels array)
        """
        X, y = [], []
        
        for _, row in dataframe.iterrows():
            img_path = os.path.join(dataset_path, row["path"])
            try:
                features = self.process_image(img_path)
                X.append(features)
                y.append(row["class"])
            except FileNotFoundError as e:
                print(f"Warning: {e}")  # TO skip missing files
                
        return np.array(X), np.array(y)


if __name__ == "__main__":
 
    DATASET_PATH = "data/raw/breakhis/BreaKHis_v1/"
    sample_img_path = r"C:\Users\USER\Documents\dl_cv\data\raw\breakhis\BreaKHis_v1\histology_slides\breast\malignant\SOB\lobular_carcinoma\SOB_M_LC_14-15570C\100X\SOB_M_LC-14-15570C-100-004.png"
    
    # Initializinge Feature Extractor
    extractor = LBPFeatureExtractor()
    
    # Processing a single image
    features = extractor.process_image(sample_img_path)
    print(f"Extracted Features Shape: {features.shape}")
