import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from process_data_non_dl import extract_lbp_features, load_data_lbp

def load_and_preprocess_data(csv_path):
    """
    Loading and preprocessing the  dataset from CSV.
    - Readings the dataset from the CSV file..
    - Extractsing th label and class information.
    
    :param csv_path: Path to the dataset CSV file.
    :return: Preprocessed pandas DataFrame.
    """
    dataset = pd.read_csv(csv_path)
    dataset = dataset[dataset["mag"] == 40]
    dataset = dataset.rename(columns={"filename": "path"})
    dataset["label"] = dataset["path"].apply(lambda x: x.split("/")[3])
    dataset["class"] = dataset["label"].apply(lambda x: 0 if x == "benign" else 1)
    return dataset

def split_data(dataset):
    """
    Splitting dataset into training, validation, and test sets using stratified shuffle split.
    
    :param dataset: Preprocessed pandas DataFrame.
    :return: DataFrames for training, validation, and testing.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    for train_index, val_test_index in sss.split(dataset["path"], dataset["class"]):
        train_df, val_test_df = dataset.iloc[train_index], dataset.iloc[val_test_index]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.67, random_state=1)
    for val_index, test_index in sss.split(val_test_df["path"], val_test_df["class"]):
        val_df, test_df = val_test_df.iloc[val_index], val_test_df.iloc[test_index]
    
    return train_df, val_df, test_df

def train_models(X_train, y_train, X_test, y_test):
    """
    Training multiple machine learning models and evaluate their performance.
    
    - Models used: RandomForest, SVM, KNN, Logistic Regression.
    - Saves trained models to disk.
    - Computeing accuracy scores for each model.
    
    :param X_train: Training feature set.
    :param y_train: Training labels.
    :param X_test: Testing feature set.
    :param y_test: Testing labels.
    :return: Dictionary containing accuracy scores for each model.
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(kernel="linear", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(),
    }
    
    results = {}
    os.makedirs("models", exist_ok=True)  # Ensure the models directory exists
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        joblib.dump(model, f"models/{model_name}_breakhis.pkl")  # Save trained model
    
    return results

def main():
    """
    Main function to execute the pipeline:
    - Loading and preprocesing thes data.
    - Splitting dataset into train, validation, and test sets.
    - Training models and evaluate their performance.
    """
    csv_path = "Folds.csv"
    dataset = load_and_preprocess_data(csv_path)
    train_df, val_df, test_df = split_data(dataset)
    
    print("Train Size:", train_df.shape, "Val Size:", val_df.shape, "Test Size:", test_df.shape)
    
    X_train, y_train = load_data_lbp(train_df)
    X_test, y_test = load_data_lbp(test_df)
    
    results = train_models(X_train, y_train, X_test, y_test)
    
    print("\nModel Performance:")
    for model, acc in results.items():
        print(f"{model}: {acc:.2f}")

if __name__ == "__main__":
    main()

