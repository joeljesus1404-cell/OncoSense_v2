"""
OncoSense - CNN Feature Extractor (Hybrid Pipeline)
Uses pretrained ResNet18 to extract deep features from histopathology images.
These features are then PCA-reduced and fed into the Quantum Kernel SVM.

Architecture: Image → ResNet18 (frozen) → 512-dim feature vector → PCA → 4 features → Quantum SVM
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os


class FeatureExtractor:
    """
    Extracts deep features from histopathology images using a pretrained ResNet18.
    
    The model is used purely as a feature extractor — the final classification layer
    is removed, outputting a 512-dimensional feature vector per image. These features
    capture high-level visual patterns (cell morphology, tissue structure, staining 
    patterns) that are then classified by the Quantum Kernel SVM.
    """

    def __init__(self, model_name='resnet18'):
        self.device = torch.device('cpu')  # CPU for deployment compatibility

        # Load pretrained ResNet18
        if model_name == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Remove classification head — keep only feature extraction layers
        if 'resnet' in model_name:
            self.model = nn.Sequential(*list(base_model.children())[:-1])
        elif 'efficientnet' in model_name:
            self.model = nn.Sequential(
                base_model.features,
                base_model.avgpool,
            )

        self.model.eval()  # Freeze in evaluation mode
        self.model.to(self.device)

        # Standard ImageNet preprocessing
        # Histopathology images are resized and normalized to match ImageNet distribution
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Histopathology-specific augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_single(self, image_path_or_pil):
        """
        Extract feature vector from a single image.
        
        Parameters:
            image_path_or_pil: str path to image or PIL Image object
            
        Returns:
            numpy array of shape (512,) — deep feature vector
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(input_tensor)

        return features.squeeze().cpu().numpy()

    def extract_batch(self, image_paths, show_progress=True):
        """
        Extract features from a batch of images.
        
        Parameters:
            image_paths: list of image file paths
            show_progress: print progress updates
            
        Returns:
            numpy array of shape (n_images, 512)
        """
        features_list = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            try:
                feat = self.extract_single(path)
                features_list.append(feat)
                if show_progress and (i + 1) % 100 == 0:
                    print(f"  Extracted features: {i + 1}/{total}")
            except Exception as e:
                print(f"  Warning: Skipping {path} — {e}")
                continue

        if show_progress:
            print(f"  Feature extraction complete: {len(features_list)}/{total} images")

        return np.array(features_list)

    def extract_from_folder(self, folder_path, label_map=None):
        """
        Extract features from an organized image folder.
        
        Expected structure:
            folder_path/
                benign/
                    img1.png, img2.png, ...
                malignant/
                    img1.png, img2.png, ...
        
        Parameters:
            folder_path: root folder containing class subfolders
            label_map: dict mapping folder names to labels (default: benign=1, malignant=0)
            
        Returns:
            features: numpy array (n_images, 512)
            labels: numpy array (n_images,)
            paths: list of file paths
        """
        if label_map is None:
            label_map = {'benign': 1, 'malignant': 0}

        all_features = []
        all_labels = []
        all_paths = []

        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

        for class_name, label in label_map.items():
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_dir):
                print(f"  Warning: Directory not found — {class_dir}")
                continue

            image_files = [
                os.path.join(class_dir, f) for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            print(f"  Processing {class_name}: {len(image_files)} images")
            features = self.extract_batch(image_files, show_progress=True)
            labels = np.full(len(features), label)

            all_features.append(features)
            all_labels.append(labels)
            all_paths.extend(image_files[:len(features)])

        if not all_features:
            raise ValueError(f"No images found in {folder_path}. Expected subfolders: {list(label_map.keys())}")

        return (
            np.vstack(all_features),
            np.concatenate(all_labels),
            all_paths
        )
