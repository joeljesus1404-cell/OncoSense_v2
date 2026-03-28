"""
OncoSense - Data Preprocessing Module
Handles data loading, cleaning, normalization, PCA reduction, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
    "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean",
    "Fractal Dimension Mean", "Radius SE", "Texture SE", "Perimeter SE", "Area SE",
    "Smoothness SE", "Compactness SE", "Concavity SE", "Concave Points SE",
    "Symmetry SE", "Fractal Dimension SE", "Radius Worst", "Texture Worst",
    "Perimeter Worst", "Area Worst", "Smoothness Worst", "Compactness Worst",
    "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"
]


def load_data():
    """Load UCI Wisconsin Breast Cancer Dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # In sklearn, 0 = malignant, 1 = benign. We keep this convention.
    return df, data


def get_dataset_info(df):
    """Return dataset statistics."""
    info = {
        "total_samples": len(df),
        "benign_count": int((df['target'] == 1).sum()),
        "malignant_count": int((df['target'] == 0).sum()),
        "num_features": len(df.columns) - 1,
        "feature_names": FEATURE_NAMES,
        "missing_values": int(df.isnull().sum().sum()),
    }
    return info


def preprocess_data(df, n_components=4, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    1. Null handling
    2. MinMaxScaler normalization
    3. PCA dimensionality reduction
    4. Train/test split (80/20)
    """
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Step 1: Handle nulls (dataset is clean, but safety check)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    # Step 2: MinMaxScaler normalization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: PCA dimensionality reduction
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "pca": pca,
        "explained_variance": explained_variance,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "y": y,
        "n_components": n_components,
    }


def preprocess_patient_input(features_dict, scaler, pca):
    """
    Preprocess a single patient's biopsy data for prediction.
    features_dict: dict of 30 feature values
    """
    feature_values = np.array([features_dict[name] for name in load_breast_cancer().feature_names])
    feature_values = feature_values.reshape(1, -1)

    # Normalize
    X_scaled = scaler.transform(feature_values)

    # PCA transform
    X_pca = pca.transform(X_scaled)

    return X_pca
