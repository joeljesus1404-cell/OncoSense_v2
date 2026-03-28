"""
OncoSense - Hybrid Pipeline Training Script
============================================

Run this script ONCE after downloading the BreakHis dataset to train
the CNN → Quantum pipeline and save model artifacts for deployment.

USAGE:
    python train_image_model.py --data_dir ./data/breakhis --output_dir ./models

DATASET SETUP:
    Download BreakHis from Kaggle:
    https://www.kaggle.com/datasets/ambarish/breakhis

    Organize the images into this folder structure:
    
    data/breakhis/
        benign/
            SOB_B_A-14-22549AB-40-001.png
            SOB_B_A-14-22549AB-40-002.png
            ...
        malignant/
            SOB_M_DC-14-2980-40-001.png
            SOB_M_DC-14-2980-40-002.png
            ...

    TIP: BreakHis has images at 4 magnifications (40X, 100X, 200X, 400X).
    For best results, pick ONE magnification (200X recommended) or mix all.
    A quick script to flatten the BreakHis folder structure:

        import shutil, os, glob
        for cls in ['benign', 'malignant']:
            os.makedirs(f'data/breakhis/{cls}', exist_ok=True)
            # Adjust the source path based on your download structure
            for img in glob.glob(f'BreaKHis_v1/**/*{cls}*/**/*.png', recursive=True):
                shutil.copy2(img, f'data/breakhis/{cls}/')
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_feature_extractor import FeatureExtractor
from utils.hybrid_quantum_pipeline import HybridQuantumPipeline


def main():
    parser = argparse.ArgumentParser(description='OncoSense - Train Hybrid CNN-Quantum Pipeline')
    parser.add_argument('--data_dir', type=str, default='./data/breakhis',
                        help='Path to image folder (with benign/ and malignant/ subfolders)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Where to save trained model artifacts')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                        help='CNN backbone for feature extraction')
    parser.add_argument('--n_components', type=int, default=4,
                        help='PCA components (must match n_qubits)')
    parser.add_argument('--quantum_train_size', type=int, default=100,
                        help='Number of training samples for quantum kernel')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images per class (None = use all)')
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║         OncoSense - Hybrid Pipeline Trainer          ║
║    CNN Feature Extraction → Quantum Kernel SVM       ║
╚══════════════════════════════════════════════════════╝

Config:
  Data:          {args.data_dir}
  CNN Backbone:  {args.model}
  PCA:           {args.n_components} components
  Quantum Train: {args.quantum_train_size} samples
  Output:        {args.output_dir}
""")

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print(f"\nPlease download BreakHis from:")
        print(f"  https://www.kaggle.com/datasets/ambarish/breakhis")
        print(f"\nAnd organize as:")
        print(f"  {args.data_dir}/benign/   (benign images)")
        print(f"  {args.data_dir}/malignant/ (malignant images)")
        sys.exit(1)

    benign_dir = os.path.join(args.data_dir, 'benign')
    malignant_dir = os.path.join(args.data_dir, 'malignant')

    if not os.path.isdir(benign_dir) or not os.path.isdir(malignant_dir):
        print(f"ERROR: Expected subfolders 'benign/' and 'malignant/' inside {args.data_dir}")
        sys.exit(1)

    # Step 1: Extract CNN features
    print("=" * 60)
    print("STEP 1: CNN Feature Extraction")
    print("=" * 60)

    start_time = time.time()
    extractor = FeatureExtractor(model_name=args.model)
    print(f"Loaded {args.model} (feature dim: {extractor.feature_dim})")

    features, labels, paths = extractor.extract_from_folder(args.data_dir)

    if args.max_images is not None:
        # Subsample per class
        import numpy as np
        indices = []
        for cls in [0, 1]:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) > args.max_images:
                cls_idx = np.random.choice(cls_idx, args.max_images, replace=False)
            indices.extend(cls_idx.tolist())
        indices = sorted(indices)
        features = features[indices]
        labels = labels[indices]
        print(f"Subsampled to {len(features)} images ({args.max_images} per class max)")

    extract_time = time.time() - start_time
    print(f"\nFeature extraction complete in {extract_time:.1f}s")
    print(f"Features shape: {features.shape}")

    # Step 2: Train Hybrid Pipeline
    print("\n" + "=" * 60)
    print("STEP 2: Training Hybrid Quantum Pipeline")
    print("=" * 60)

    pipeline = HybridQuantumPipeline(
        n_components=args.n_components,
        n_qubits=args.n_components,
        reps=2,
        quantum_train_size=args.quantum_train_size
    )

    train_start = time.time()
    metrics = pipeline.train(features, labels)
    train_time = time.time() - train_start

    # Step 3: Save
    print("=" * 60)
    print("STEP 3: Saving Model Artifacts")
    print("=" * 60)

    save_path = pipeline.save(args.output_dir)

    # Summary
    total_time = time.time() - start_time
    print(f"""
╔══════════════════════════════════════════════════════╗
║                 TRAINING COMPLETE                     ║
╠══════════════════════════════════════════════════════╣
║  Quantum SVM:  acc={metrics['quantum']['accuracy']:.4f}  auc={metrics['quantum']['auc_roc']:.4f}         ║
║  Linear SVM:   acc={metrics['linear']['accuracy']:.4f}  auc={metrics['linear']['auc_roc']:.4f}         ║
║  RBF SVM:      acc={metrics['rbf']['accuracy']:.4f}  auc={metrics['rbf']['auc_roc']:.4f}         ║
╠══════════════════════════════════════════════════════╣
║  Feature extraction: {extract_time:>6.1f}s                          ║
║  Pipeline training:  {train_time:>6.1f}s                          ║
║  Total time:         {total_time:>6.1f}s                          ║
╠══════════════════════════════════════════════════════╣
║  Saved to: {args.output_dir + '/':.<43s}║
╚══════════════════════════════════════════════════════╝

Next steps:
  1. Copy the '{args.output_dir}/' folder into your deployment
  2. The Streamlit app will auto-detect and load the trained model
  3. Deploy to Railway: git add . && git commit && git push
""")


if __name__ == '__main__':
    main()
