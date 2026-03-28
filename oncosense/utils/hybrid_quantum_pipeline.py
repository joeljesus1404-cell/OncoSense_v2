"""
OncoSense - Hybrid Quantum Pipeline
Combines CNN feature extraction with Quantum Kernel SVM classification.

Full Pipeline:
    Image → ResNet18 → 512-dim features → PCA → 4 features → ZZFeatureMap → Quantum Kernel → SVM → Prediction
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class HybridQuantumPipeline:
    """
    Hybrid CNN-Quantum pipeline for image-based cancer classification.
    
    Training:
        1. CNN extracts 512-dim features from labeled images
        2. MinMaxScaler normalizes features
        3. PCA reduces to n_components (default 4) for quantum circuit compatibility
        4. ZZFeatureMap encodes features into quantum states
        5. Fidelity Quantum Kernel computes kernel matrix
        6. SVC trains on precomputed quantum kernel
    
    Inference:
        1. CNN extracts features from new image
        2. Same scaler + PCA transform
        3. Quantum kernel evaluates against training data
        4. SVC predicts malignant/benign with confidence
    """

    def __init__(self, n_components=4, n_qubits=4, reps=2, quantum_train_size=100):
        self.n_components = n_components
        self.n_qubits = n_qubits
        self.reps = reps
        self.quantum_train_size = quantum_train_size

        # Pipeline components (set during training)
        self.scaler = None
        self.pca = None
        self.quantum_kernel = None
        self.feature_map = None
        self.svm_model = None
        self.X_train_quantum = None  # Training data needed for kernel evaluation at inference

        # Classical baselines
        self.svm_linear = None
        self.svm_rbf = None

        # Metrics
        self.quantum_metrics = None
        self.linear_metrics = None
        self.rbf_metrics = None
        self.pca_variance = None

        self.is_trained = False

    def train(self, features, labels, test_size=0.2, random_state=42):
        """
        Train the full hybrid pipeline on CNN-extracted features.
        
        Parameters:
            features: numpy array (n_samples, 512) — CNN features
            labels: numpy array (n_samples,) — 0=malignant, 1=benign
            test_size: fraction for test split
            random_state: reproducibility seed
        """
        print(f"\n{'='*60}")
        print("TRAINING HYBRID QUANTUM PIPELINE")
        print(f"{'='*60}")
        print(f"Samples: {len(features)} | Features: {features.shape[1]}")
        print(f"Benign: {(labels==1).sum()} | Malignant: {(labels==0).sum()}")

        # Step 1: Normalize
        print("\n[1/6] MinMaxScaler normalization...")
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(features)

        # Step 2: PCA
        print(f"[2/6] PCA reduction: {features.shape[1]} -> {self.n_components} features...")
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        self.pca_variance = self.pca.explained_variance_ratio_
        print(f"       Explained variance: {[f'{v:.3f}' for v in self.pca_variance]}")
        print(f"       Total: {sum(self.pca_variance):.3f}")

        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        print(f"[3/6] Split: Train={len(X_train)}, Test={len(X_test)}")

        # Step 4: Classical baselines (full data)
        print("[4/6] Training Classical SVM baselines...")
        self.svm_linear = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        self.svm_linear.fit(X_train, y_train)
        y_pred_lin = self.svm_linear.predict(X_test)
        y_prob_lin = self.svm_linear.predict_proba(X_test)[:, 1]
        self.linear_metrics = self._compute_metrics(y_test, y_pred_lin, y_prob_lin)
        print(f"       Linear SVM:  acc={self.linear_metrics['accuracy']:.4f}  auc={self.linear_metrics['auc_roc']:.4f}")

        self.svm_rbf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        self.svm_rbf.fit(X_train, y_train)
        y_pred_rbf = self.svm_rbf.predict(X_test)
        y_prob_rbf = self.svm_rbf.predict_proba(X_test)[:, 1]
        self.rbf_metrics = self._compute_metrics(y_test, y_pred_rbf, y_prob_rbf)
        print(f"       RBF SVM:     acc={self.rbf_metrics['accuracy']:.4f}  auc={self.rbf_metrics['auc_roc']:.4f}")

        # Step 5: Quantum Kernel SVM (subset for computational feasibility)
        print(f"[5/6] Training Quantum Kernel SVM ({self.quantum_train_size} samples)...")
        q_size = min(self.quantum_train_size, len(X_train))
        q_test_size = min(30, len(X_test))

        if q_size < len(X_train):
            X_q_train, _, y_q_train, _ = train_test_split(
                X_train, y_train, train_size=q_size, random_state=42, stratify=y_train
            )
        else:
            X_q_train, y_q_train = X_train, y_train

        if q_test_size < len(X_test):
            X_q_test, _, y_q_test, _ = train_test_split(
                X_test, y_test, train_size=q_test_size, random_state=42, stratify=y_test
            )
        else:
            X_q_test, y_q_test = X_test, y_test

        # Build quantum kernel
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, reps=self.reps, entanglement='full'
        )
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # Compute kernel matrices
        print("       Computing quantum kernel matrix...")
        K_train = self.quantum_kernel.evaluate(X_q_train)
        K_test = self.quantum_kernel.evaluate(X_q_test, X_q_train)

        # Train quantum SVM
        self.svm_model = SVC(kernel='precomputed', C=1.0, probability=True)
        self.svm_model.fit(K_train, y_q_train)
        self.X_train_quantum = X_q_train  # Save for inference

        y_pred_q = self.svm_model.predict(K_test)
        y_prob_q = self.svm_model.predict_proba(K_test)[:, 1]
        self.quantum_metrics = self._compute_metrics(y_q_test, y_pred_q, y_prob_q)
        print(f"       Quantum SVM: acc={self.quantum_metrics['accuracy']:.4f}  auc={self.quantum_metrics['auc_roc']:.4f}")

        # Step 6: Done
        self.is_trained = True
        print(f"\n[6/6] Pipeline trained successfully!")
        print(f"{'='*60}\n")

        return {
            'quantum': self.quantum_metrics,
            'linear': self.linear_metrics,
            'rbf': self.rbf_metrics,
        }

    def predict_image(self, cnn_features):
        """
        Predict on a single image's CNN features.
        
        Parameters:
            cnn_features: numpy array (512,) from FeatureExtractor
            
        Returns:
            dict with prediction, label, confidence, probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        # Preprocess
        features = cnn_features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)

        # Quantum prediction
        K = self.quantum_kernel.evaluate(features_pca, self.X_train_quantum)
        q_pred = self.svm_model.predict(K)
        q_prob = self.svm_model.predict_proba(K)

        # Classical predictions
        lin_pred = self.svm_linear.predict(features_pca)
        lin_prob = self.svm_linear.predict_proba(features_pca)

        rbf_pred = self.svm_rbf.predict(features_pca)
        rbf_prob = self.svm_rbf.predict_proba(features_pca)

        return {
            'quantum': {
                'prediction': int(q_pred[0]),
                'label': 'Benign' if q_pred[0] == 1 else 'Malignant',
                'confidence': float(np.max(q_prob)),
                'prob_benign': float(q_prob[0][1]) if q_prob.shape[1] > 1 else float(q_prob[0][0]),
                'prob_malignant': float(q_prob[0][0]),
            },
            'linear': {
                'prediction': int(lin_pred[0]),
                'label': 'Benign' if lin_pred[0] == 1 else 'Malignant',
                'confidence': float(np.max(lin_prob)),
            },
            'rbf': {
                'prediction': int(rbf_pred[0]),
                'label': 'Benign' if rbf_pred[0] == 1 else 'Malignant',
                'confidence': float(np.max(rbf_prob)),
            }
        }

    def save(self, save_dir):
        """Save all trained pipeline components."""
        os.makedirs(save_dir, exist_ok=True)

        state = {
            'scaler': self.scaler,
            'pca': self.pca,
            'svm_model': self.svm_model,
            'svm_linear': self.svm_linear,
            'svm_rbf': self.svm_rbf,
            'X_train_quantum': self.X_train_quantum,
            'quantum_metrics': self.quantum_metrics,
            'linear_metrics': self.linear_metrics,
            'rbf_metrics': self.rbf_metrics,
            'pca_variance': self.pca_variance,
            'n_components': self.n_components,
            'n_qubits': self.n_qubits,
            'reps': self.reps,
            'is_trained': self.is_trained,
        }

        path = os.path.join(save_dir, 'hybrid_pipeline.pkl')
        with open(path, 'wb') as f:
            pickle.dump(state, f)

        # Save quantum kernel separately (it contains the feature map)
        kernel_state = {
            'feature_map_params': {
                'feature_dimension': self.n_qubits,
                'reps': self.reps,
                'entanglement': 'full',
            }
        }
        kernel_path = os.path.join(save_dir, 'quantum_kernel_config.pkl')
        with open(kernel_path, 'wb') as f:
            pickle.dump(kernel_state, f)

        print(f"Pipeline saved to {save_dir}/")
        return path

    def load(self, save_dir):
        """Load a trained pipeline from disk."""
        path = os.path.join(save_dir, 'hybrid_pipeline.pkl')
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.scaler = state['scaler']
        self.pca = state['pca']
        self.svm_model = state['svm_model']
        self.svm_linear = state['svm_linear']
        self.svm_rbf = state['svm_rbf']
        self.X_train_quantum = state['X_train_quantum']
        self.quantum_metrics = state['quantum_metrics']
        self.linear_metrics = state['linear_metrics']
        self.rbf_metrics = state['rbf_metrics']
        self.pca_variance = state['pca_variance']
        self.n_components = state['n_components']
        self.n_qubits = state['n_qubits']
        self.reps = state['reps']
        self.is_trained = state['is_trained']

        # Rebuild quantum kernel
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, reps=self.reps, entanglement='full'
        )
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)

        print(f"Pipeline loaded from {save_dir}/")
        return self

    def _compute_metrics(self, y_true, y_pred, y_prob):
        """Compute evaluation metrics."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'roc_curve': {
                'fpr': roc_curve(y_true, y_prob)[0].tolist(),
                'tpr': roc_curve(y_true, y_prob)[1].tolist(),
            }
        }
