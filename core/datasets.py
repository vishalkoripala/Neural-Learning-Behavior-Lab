"""
Dataset Generation and Preprocessing Utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from typing import Tuple, Optional, List, Dict, Any

AVAILABLE_DATASETS = [
    "Linearly Separable",
    "XOR Problem",
    "Two Moons",
    "Concentric Circles",
    "Two Spirals",
    "Anisotropic Blobs"
]

def generate_dataset(
    dataset_name: str,
    n_samples: int = 120,
    noise: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D synthetic datasets for neural learning behavior experimentation.
    Returns:
        X: np.ndarray of shape (N, 2), normalized/standardized
        y: np.ndarray of shape (N,), values in {-1, 1}
    """
    rng = np.random.RandomState(random_state)
    
    if dataset_name == "Linearly Separable":
        n_per_class = n_samples // 2
        cov = [[0.6, 0.1], [0.1, 0.6]]
        c1 = rng.multivariate_normal([1.8, 1.8], cov, n_per_class)
        c2 = rng.multivariate_normal([-1.8, -1.8], cov, n_samples - n_per_class)
        X = np.vstack([c1, c2])
        y = np.hstack([np.ones(n_per_class), -np.ones(n_samples - n_per_class)])
        if noise > 0:
            X += rng.normal(0, noise * 1.5, X.shape)

    elif dataset_name == "XOR Problem":
        # Multi-cluster XOR with spread
        n_per_cluster = max(1, n_samples // 4)
        spread = 0.2 + noise * 0.4
        c1 = rng.normal(loc=[1.0, 1.0], scale=spread, size=(n_per_cluster, 2))
        c2 = rng.normal(loc=[-1.0, -1.0], scale=spread, size=(n_per_cluster, 2))
        c3 = rng.normal(loc=[-1.0, 1.0], scale=spread, size=(n_per_cluster, 2))
        c4 = rng.normal(loc=[1.0, -1.0], scale=spread, size=(n_samples - 3 * n_per_cluster, 2))
        X = np.vstack([c1, c2, c3, c4])
        y = np.hstack([
            np.ones(n_per_cluster),
            np.ones(n_per_cluster),
            -np.ones(n_per_cluster),
            -np.ones(n_samples - 3 * n_per_cluster)
        ])

    elif dataset_name == "Two Moons":
        n_per_moon = n_samples // 2
        # Upper moon
        lin1 = np.linspace(0, np.pi, n_per_moon)
        x1 = np.cos(lin1)
        y1 = np.sin(lin1)
        moon1 = np.column_stack([x1, y1])
        
        # Lower moon
        lin2 = np.linspace(0, np.pi, n_samples - n_per_moon)
        x2 = 1.0 - np.cos(lin2)
        y2 = 1.0 - np.sin(lin2) - 0.5
        moon2 = np.column_stack([x2, y2])
        
        X = np.vstack([moon1, moon2])
        y = np.hstack([np.ones(n_per_moon), -np.ones(n_samples - n_per_moon)])
        if noise > 0:
            X += rng.normal(0, noise * 0.8, X.shape)

    elif dataset_name == "Concentric Circles":
        n_per_circle = n_samples // 2
        # Inner circle
        theta1 = rng.uniform(0, 2 * np.pi, n_per_circle)
        r1 = rng.uniform(0.1, 0.45, n_per_circle) + (rng.normal(0, noise * 0.2, n_per_circle) if noise > 0 else 0)
        c1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
        
        # Outer circle
        theta2 = rng.uniform(0, 2 * np.pi, n_samples - n_per_circle)
        r2 = rng.uniform(0.75, 1.15, n_samples - n_per_circle) + (rng.normal(0, noise * 0.2, n_samples - n_per_circle) if noise > 0 else 0)
        c2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        
        X = np.vstack([c1, c2])
        y = np.hstack([np.ones(n_per_circle), -np.ones(n_samples - n_per_circle)])

    elif dataset_name == "Two Spirals":
        n_per_spiral = n_samples // 2
        # Spiral 1
        t1 = np.linspace(0.5, 3.5 * np.pi, n_per_spiral)
        r1 = t1 / (3.5 * np.pi)
        x1 = r1 * np.cos(t1)
        y1 = r1 * np.sin(t1)
        s1 = np.column_stack([x1, y1])
        
        # Spiral 2
        t2 = np.linspace(0.5, 3.5 * np.pi, n_samples - n_per_spiral)
        r2 = t2 / (3.5 * np.pi)
        x2 = -r2 * np.cos(t2)
        y2 = -r2 * np.sin(t2)
        s2 = np.column_stack([x2, y2])
        
        X = np.vstack([s1, s2])
        y = np.hstack([np.ones(n_per_spiral), -np.ones(n_samples - n_per_spiral)])
        if noise > 0:
            X += rng.normal(0, noise * 0.15, X.shape)

    elif dataset_name == "Anisotropic Blobs":
        n_per_class = n_samples // 2
        # Elongated diagonal blobs
        transform = [[0.6, -0.6], [-0.4, 0.8]]
        c1 = np.dot(rng.randn(n_per_class, 2), transform) + [1.5, 0.5]
        c2 = np.dot(rng.randn(n_samples - n_per_class, 2), transform) + [-1.5, -0.5]
        X = np.vstack([c1, c2])
        y = np.hstack([np.ones(n_per_class), -np.ones(n_samples - n_per_class)])
        if noise > 0:
            X += rng.normal(0, noise, X.shape)
            
    else:
        # Fallback linearly separable
        return generate_dataset("Linearly Separable", n_samples, noise, random_state)

    # Standardize features for stable neural learning dynamics
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def process_uploaded_dataframe(
    df: pd.DataFrame,
    target_column: str,
    feature_selection: str = "First 2 Features",
    selected_features: Optional[List[str]] = None,
    target_classes: Optional[List[Any]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
    """
    Process user uploaded CSV into standardized 2D X and {-1, 1} binary y.
    Returns:
        (X, y, feature_names, error_message)
    """
    try:
        # Separate target
        y_raw = df[target_column].copy()
        X_df = df.drop(columns=[target_column]).copy()

        # Handle numeric features
        numeric_cols = list(X_df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 2 and feature_selection != "PCA (All Features)":
            return np.empty((0, 2)), np.empty((0,)), [], "CSV must contain at least 2 numeric feature columns."

        # Handle Target Encoding
        if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
            le = LabelEncoder()
            y_enc = le.fit_transform(y_raw.astype(str))
            unique_classes = le.classes_
        else:
            unique_classes = np.unique(y_raw[~pd.isna(y_raw)])
            y_enc = y_raw.values

        if len(unique_classes) < 2:
            return np.empty((0, 2)), np.empty((0,)), [], "Target column must have at least 2 distinct classes."

        # Binary Filter if more than 2 classes
        if len(unique_classes) > 2:
            if target_classes and len(target_classes) == 2:
                mask = np.isin(y_raw, target_classes)
                X_df = X_df[mask]
                y_raw = y_raw[mask]
                unique_classes = target_classes
            else:
                # Default take first 2 classes
                c1, c2 = unique_classes[0], unique_classes[1]
                mask = (y_raw == c1) | (y_raw == c2)
                X_df = X_df[mask]
                y_raw = y_raw[mask]
                unique_classes = [c1, c2]

        y = np.where(y_raw == unique_classes[0], -1, 1).astype(float)

        # Feature Dimensionality Reduction / Selection
        if feature_selection == "PCA (All Features)":
            # Impute missing values with median
            X_num = X_df[numeric_cols].fillna(X_df[numeric_cols].median()).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_num)
            pca = PCA(n_components=2)
            X = pca.fit_transform(X_scaled)
            feature_names = [f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"]
        elif selected_features and len(selected_features) == 2:
            X_num = X_df[selected_features].fillna(X_df[selected_features].median()).values
            scaler = StandardScaler()
            X = scaler.fit_transform(X_num)
            feature_names = selected_features
        else:
            first_two = numeric_cols[:2]
            X_num = X_df[first_two].fillna(X_df[first_two].median()).values
            scaler = StandardScaler()
            X = scaler.fit_transform(X_num)
            feature_names = first_two

        return X, y, feature_names, None

    except Exception as e:
        return np.empty((0, 2)), np.empty((0,)), [], f"Error processing dataset: {str(e)}"
