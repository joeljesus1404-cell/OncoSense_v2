"""
OncoSense - Quantum Processing Engine
Implements Quantum Kernel SVM using Qiskit's ZZFeatureMap and Fidelity Quantum Kernel.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def build_quantum_kernel(n_features=4, reps=2, entanglement='full'):
    """
    Build the Quantum Kernel using ZZFeatureMap.

    ZZFeatureMap applies:
    - Hadamard gates on all qubits
    - Parameterized Rz rotations (data encoding)
    - ZZ entanglement gates (captures second-order correlations)

    Parameters:
        n_features: Number of qubits (= PCA components)
        reps: Number of repetitions of the feature map circuit
        entanglement: Entanglement strategy ('full', 'linear', 'circular')
    """
    # Create the ZZFeatureMap
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=reps,
        entanglement=entanglement
    )

    # Create the Fidelity Quantum Kernel
    # K(x, x') = |<ψ(x)|ψ(x')>|²
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    return quantum_kernel, feature_map


def train_quantum_svm(X_train, y_train, X_test, y_test, n_features=4, reps=2):
    """
    Train the Quantum Kernel SVM classifier.

    Pipeline:
    1. Build ZZFeatureMap quantum circuit
    2. Compute quantum kernel matrices (train-train and test-train)
    3. Train SVC with precomputed kernel
    4. Evaluate on test set
    """
    # Build quantum kernel
    quantum_kernel, feature_map = build_quantum_kernel(
        n_features=n_features, reps=reps
    )

    # Compute kernel matrices
    # Training kernel matrix: K(x_train, x_train)
    kernel_matrix_train = quantum_kernel.evaluate(X_train)

    # Test kernel matrix: K(x_test, x_train)
    kernel_matrix_test = quantum_kernel.evaluate(X_test, X_train)

    # Train SVC with precomputed quantum kernel
    qsvm = SVC(kernel='precomputed', C=1.0, probability=True)
    qsvm.fit(kernel_matrix_train, y_train)

    # Predict
    y_pred = qsvm.predict(kernel_matrix_test)
    y_prob = qsvm.predict_proba(kernel_matrix_test)[:, 1]

    # Evaluate
    metrics = compute_metrics(y_test, y_pred, y_prob)

    return {
        "model": qsvm,
        "quantum_kernel": quantum_kernel,
        "feature_map": feature_map,
        "kernel_matrix_train": kernel_matrix_train,
        "kernel_matrix_test": kernel_matrix_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }


def predict_single_quantum(X_input, X_train, quantum_kernel, qsvm_model):
    """
    Predict on a single patient input using the trained quantum model.
    """
    kernel_matrix = quantum_kernel.evaluate(X_input, X_train)
    prediction = qsvm_model.predict(kernel_matrix)
    probability = qsvm_model.predict_proba(kernel_matrix)

    return {
        "prediction": int(prediction[0]),
        "label": "Benign" if prediction[0] == 1 else "Malignant",
        "confidence": float(np.max(probability)),
        "prob_benign": float(probability[0][1]),
        "prob_malignant": float(probability[0][0]),
    }


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


def get_circuit_info(feature_map):
    """Get quantum circuit details for display."""
    return {
        "num_qubits": feature_map.num_qubits,
        "depth": feature_map.depth(),
        "num_parameters": feature_map.num_parameters,
        "gate_counts": dict(feature_map.count_ops()),
    }
