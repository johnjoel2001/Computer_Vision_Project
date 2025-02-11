import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_image
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
from process_dataset_non_dl import extract_lbp_features  # Import LBP feature extractor

def load_model(model_path):
    """
    Loading a trained machine learning model.
    :param model_path: Path to the trained model file.
    :return: Loaded model.
    """
    return joblib.load(model_path)

def load_dataset(csv_path):
    """
    Loadi ng and preprocessing the dataset from a CSV file.
    :param csv_path: Path to the dataset CSV file.
    :return: Preprocessed pandas DataFrame.
    """
    dataset = pd.read_csv(csv_path)
    dataset = dataset[dataset["mag"] == 40]  # Filter for 40X magnification
    dataset = dataset.rename(columns={"filename": "path"})
    dataset["label"] = dataset["path"].apply(lambda x: x.split("/")[3])
    dataset["class"] = dataset["label"].apply(lambda x: 0 if x == "benign" else 1)
    return dataset

def predict_with_lbp(image_batch, model):
    """
    Extracting LBP features and predicts probabilities using a trained model.
    :param image_batch: List of images.
    :param model: Trained machine learning model.
    :return: Predicted probabilities.
    """
    processed_features = [extract_lbp_features(resize(img, (128, 128))) for img in image_batch]
    return model.predict_proba(np.array(processed_features))

def explain_random_images_lime(test_df, model, dataset_path, num_samples=5):
    """
    Explaiing thes model predictions on randomly selected test images using LIME.
    :param test_df: DataFrame containing test images.
    :param model: Trained model.
    :param dataset_path: Path to dataset images.
    :param num_samples: Number of images to explain.
    """
    sample_test_images = test_df.sample(num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i, (_, row) in enumerate(sample_test_images.iterrows()):
        image_path = os.path.join(dataset_path, row["path"])
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue

        image = imread(image_path)
        image_resized = resize(image, (128, 128))
        features_array = np.array(extract_lbp_features(image_resized)).reshape(1, -1)

        prediction = model.predict(features_array)[0]
        predicted_label = "Malignant (Yes)" if prediction == 1 else "Benign (No)"
        true_label = "Malignant (Yes)" if row["class"] == 1 else "Benign (No)"

        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"True: {true_label}\nPredicted: {predicted_label}",
                             fontsize=10, color="red" if prediction != row["class"] else "green")

        explainer = lime.lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_resized, lambda x: predict_with_lbp(x, model), top_labels=2, hide_color=0, num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
        )
        axes[i, 1].imshow(mark_boundaries(temp, mask))
        axes[i, 1].axis("off")
        axes[i, 1].set_title("LIME: Important Regions")

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
        )
        axes[i, 2].imshow(mark_boundaries(temp, mask))
        axes[i, 2].axis("off")
        axes[i, 2].set_title("LIME: Positive & Negative Regions")

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the LIME explanation pipeline.
    """
    model_path = "models/RandomForest_breakhis.pkl"
    dataset_path = "data/raw/breakhis/BreaKHis_v1/"
    csv_path = "data/raw/breakhis/Folds.csv"

    model = load_model(model_path)
    dataset = load_dataset(csv_path)
    explain_random_images_lime(dataset, model, dataset_path, num_samples=5)

if __name__ == "__main__":
    main()
