#!/usr/bin/env python3
"""
Generate test predictions for fungi classification competition submission.

This script loads a trained DINOv2 model and generates predictions for the test set
in the required CSV format for submission.

Example output format:
DINOv2_VITG14_seesaw_submission
fungi_test000000.jpg,99
fungi_test000001.jpg,2
fungi_test000002.jpg,107
...
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
import argparse


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class FungiTestDataset(Dataset):
    """Dataset class for test fungi images"""
    def __init__(self, df, image_path, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]
        
        # Load image
        with Image.open(os.path.join(self.image_path, file_path)) as img:
            image = img.convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, file_path


class DINOv2FungiClassifier(nn.Module):
    """DINOv2-based classifier for fungi classification."""
    
    def __init__(self, num_classes, model_name='dinov2_vits14', dropout_rate=0.2, classifier_type='simple'):
        super(DINOv2FungiClassifier, self).__init__()
        
        # Load DINOv2 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)  # type: ignore
        
        # Freeze backbone parameters for feature extraction
        for param in self.backbone.parameters():  # type: ignore
            param.requires_grad = False
        
        # Get embedding dimension
        if 'vits14' in model_name:
            embed_dim = 384
        elif 'vitb14' in model_name:
            embed_dim = 768
        elif 'vitl14' in model_name:
            embed_dim = 1024
        elif 'vitg14' in model_name:
            embed_dim = 1536
        else:
            embed_dim = 384
        
        # Classifier head - supports both simple and complex architectures
        if classifier_type == 'simple':
            # Simple linear classifier (used in Seesaw/Focal models)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            # 3-layer classifier (used in baseline models)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # type: ignore
        output = self.classifier(features)
        return output


def get_test_transforms(image_size=224):
    """Get test transforms (same as validation transforms)"""
    return Compose([
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def find_test_images(metadata_file):
    """Find all test images from the metadata file."""
    df = pd.read_csv(metadata_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')].copy()
    
    # Sort by filename to ensure consistent ordering
    test_df = test_df.sort_values('filename_index').reset_index(drop=True)
    
    print(f"Found {len(test_df)} test images")
    print(f"Test image range: {test_df['filename_index'].iloc[0]} to {test_df['filename_index'].iloc[-1]}")
    
    return test_df


def load_class_mapping(metadata_file):
    """Load the class mapping from training data."""
    df = pd.read_csv(metadata_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    
    # Get unique classes and create mapping
    unique_classes = sorted([int(x) for x in train_df['taxonID_index'].dropna().unique()])
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    print(f"Found {len(unique_classes)} classes")
    print(f"Class range: {min(unique_classes)} to {max(unique_classes)}")
    
    return idx_to_class, len(unique_classes)


def generate_predictions(model_path, metadata_file, image_path, output_path, 
                        model_name='dinov2_vitg14', image_size=224, 
                        classifier_type='simple', method_description=None):
    """
    Generate test predictions using a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint (.pth file)
        metadata_file: Path to the metadata CSV file
        image_path: Path to the directory containing images
        output_path: Path where to save the prediction CSV
        model_name: DINOv2 model variant used
        image_size: Image size used during training
        classifier_type: 'simple' or 'complex' classifier architecture
        method_description: Description for the first line of CSV (auto-generated if None)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find test images and load class mapping
    test_df = find_test_images(metadata_file)
    idx_to_class, num_classes = load_class_mapping(metadata_file)
    
    # Create test dataset and dataloader
    test_transforms = get_test_transforms(image_size)
    test_dataset = FungiTestDataset(test_df, image_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = DINOv2FungiClassifier(
        num_classes=num_classes, 
        model_name=model_name, 
        dropout_rate=0.2,
        classifier_type=classifier_type
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model: {model_name}")
    print(f"Classifier: {classifier_type}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Number of classes: {num_classes}")
    
    # Generate predictions
    predictions = []
    filenames = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for images, file_paths in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            
            # Get predicted class indices
            predicted_indices = outputs.argmax(1).cpu().numpy()
            
            # Convert indices back to original class labels
            predicted_classes = [idx_to_class[idx] for idx in predicted_indices]
            
            predictions.extend(predicted_classes)
            filenames.extend(file_paths)
    
    # Create method description if not provided
    if method_description is None:
        method_description = f"DINOv2_{model_name.split('_')[1].upper()}_{classifier_type}_submission"
    
    # Save predictions in competition format
    print(f"Saving predictions to: {output_path}")
    with open(output_path, 'w') as f:
        # First line: method description
        f.write(f"{method_description}\n")
        
        # Predictions: filename,class_label
        for filename, pred_class in zip(filenames, predictions):
            f.write(f"{filename},{pred_class}\n")
    
    print(f"✅ Predictions saved successfully!")
    print(f"📊 Generated {len(predictions)} predictions")
    print(f"🎯 Method: {method_description}")
    
    # Show some sample predictions
    print(f"\n📋 Sample predictions:")
    for i in range(min(5, len(filenames))):
        print(f"   {filenames[i]} → {predictions[i]}")
    if len(filenames) > 5:
        print(f"   ... and {len(filenames) - 5} more")


def main():
    parser = argparse.ArgumentParser(description='Generate test predictions for fungi classification')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path where to save the prediction CSV file')
    
    # Optional arguments
    parser.add_argument('--config_path', type=str, 
                       default='/home/pabllo/dl_projects/MultimodalDataChallenge2025/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_name', type=str, default='dinov2_vitg14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2 model variant')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size used during training')
    parser.add_argument('--classifier_type', type=str, default='simple',
                       choices=['simple', 'complex'],
                       help='Classifier architecture type')
    parser.add_argument('--method_description', type=str, default=None,
                       help='Description for the submission (first line of CSV)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    metadata_file = config['paths']['metadata_file']
    image_path = config['paths']['image_path']
    
    # Generate predictions
    generate_predictions(
        model_path=args.model_path,
        metadata_file=metadata_file,
        image_path=image_path,
        output_path=args.output_path,
        model_name=args.model_name,
        image_size=args.image_size,
        classifier_type=args.classifier_type,
        method_description=args.method_description
    )


if __name__ == "__main__":
    # Example usage without command line arguments
    if len(sys.argv) == 1:
        print("🍄 Fungi Test Prediction Generator")
        print("=" * 35)
        print()
        print("Usage examples:")
        print()
        print("1. Generate predictions from Seesaw model:")
        print("   python generate_test_predictions.py \\")
        print("     --model_path /path/to/seesaw/best_accuracy.pth \\")
        print("     --output_path seesaw_predictions.csv \\")
        print("     --classifier_type simple \\")
        print("     --method_description 'DINOv2_VITG14_seesaw_submission'")
        print()
        print("2. Generate predictions from baseline model:")
        print("   python generate_test_predictions.py \\")
        print("     --model_path /path/to/baseline/best_accuracy.pth \\")
        print("     --output_path baseline_predictions.csv \\")
        print("     --classifier_type complex \\")
        print("     --method_description 'DINOv2_VITG14_baseline_submission'")
        print()
        print("3. Quick example with auto-detection:")
        print("   python generate_test_predictions.py \\")
        print("     --model_path /media/small_diskie/fungidata/checkpoints/DINOv2_VITG14_baseline/best_accuracy.pth \\")
        print("     --output_path predictions.csv")
        print()
        print("Available model names: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
        print("Available classifier types: simple (for Seesaw/Focal), complex (for baseline)")
        
    else:
        main()
