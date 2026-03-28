"""
OncoSense - Classical Baseline Engine
Implements Classical SVM baselines (Linear and RBF kernels) for comparison with Quantum SVM.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)


def train_classical_svm(X_train, y_train, X_test, y_test, kernel='linear', C=1.0):
    """
    Train a classical SVM classifier.

    Parameters:
        kernel: 'linear' or 'rbf'
        C: Regularization parameter
    """
    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    return {
        "model": model,
        "kernel": kernel,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }


def predict_single_classical(X_input, model):
    """Predict on a single patient input using classical model."""
    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)

    return {
        "prediction": int(prediction[0]),
        "label": "Benign" if prediction[0] == 1 else "Malignant",
        "confidence": float(np.max(probability)),
        "prob_benign": float(probability[0][1]),
        "prob_malignant": float(probability[0][0]),
    }


def train_all_classical(X_train, y_train, X_test, y_test):
    """Train both Linear and RBF SVM baselines."""
    results = {}

    # Linear SVM
    results['linear'] = train_classical_svm(
        X_train, y_train, X_test, y_test, kernel='linear'
    )

    # RBF SVM
    results['rbf'] = train_classical_svm(
        X_train, y_train, X_test, y_test, kernel='rbf'
    )

    return results


def compute_metrics(y_true, y_pred, y_prob):
    """Compute full evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": roc_curve(y_true, y_prob)[0].tolist(),
            "tpr": roc_curve(y_true, y_prob)[1].tolist(),
            "thresholds": roc_curve(y_true, y_prob)[2].tolist(),
        }
    }
