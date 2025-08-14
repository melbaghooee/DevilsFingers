#!/usr/bin/env python3
"""
Extract DINOv2 CLS Tokens for Entire Dataset

This script pre-extracts CLS tokens from a DINOv2 model for all images in the dataset
and saves each feature as an individual .npy file using the image filename (without .jpg).

Usage:
    python src/extract_cls_tokens.py --model_size giant --save_folder /path/to/npy/files
"""

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import yaml
from tqdm import tqdm
import argparse
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class FungiDataset(Dataset):
    """Dataset class for fungi images"""
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
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, file_path, idx

def get_transforms(image_size):
    """Get transforms for feature extraction"""
    return Compose([
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_dinov2_model(model_size='giant'):
    """Load DINOv2 backbone model"""
    model_map = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14', 
        'large': 'dinov2_vitl14',
        'giant': 'dinov2_vitg14'
    }
    
    embed_dims = {
        'small': 384,
        'base': 768,
        'large': 1024,
        'giant': 1536
    }
    
    model_name = model_map[model_size]
    model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model, embed_dims[model_size]

def extract_features_and_save(model, dataloader, device, save_folder, model_size):
    """
    Extract CLS tokens and save each feature as individual .npy file
    """
    model.eval()
    
    # Create save folder
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"Extracting {model_size} CLS tokens and saving as .npy files...")
    
    processed_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, filenames, indices) in enumerate(tqdm(dataloader, desc="Extracting & Saving")):
            images = images.to(device)
            
            # Extract CLS tokens
            features = model(images)  # Shape: [batch_size, embed_dim]
            
            (b,1,)
            # Save each feature individually
            for feature, filename in zip(features, filenames):
                # Remove .jpg extension and add .npy
                base_name = os.path.splitext(filename)[0]
                npy_filename = f"{base_name}.npy"
                npy_path = os.path.join(save_folder, npy_filename)
                
                # Convert to numpy and save
                feature_np = feature.cpu().numpy()
                np.save(npy_path, feature_np)
                
                processed_count += 1
            
            # Progress update every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {processed_count} images...")
    
    print(f"✅ Extraction complete! Saved {processed_count} .npy files")
    print(f"📁 Files saved in: {save_folder}")
    
    return processed_count

def create_feature_summary(save_folder, model_size, processed_count):
    """Create a summary file with information about the extracted features"""
    
    embed_dims = {
        'small': 384,
        'base': 768,
        'large': 1024,
        'giant': 1536
    }
    
    summary = {
        'model_size': model_size,
        'embed_dim': embed_dims[model_size],
        'extraction_date': pd.Timestamp.now().isoformat(),
        'save_folder': save_folder,
        'total_files': processed_count,
        'file_format': '.npy',
        'feature_shape': f'({embed_dims[model_size]},)'
    }
    
    # Save summary
    summary_file = os.path.join(save_folder, f'extraction_summary_{model_size}.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {summary_file}")
    return summary_file

def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 CLS Tokens and Save as Individual .npy Files')
    parser.add_argument('--model_size', type=str, default='giant', 
                        choices=['small', 'base', 'large', 'giant'],
                        help='DINOv2 model size')
    parser.add_argument('--save_folder', type=str, required=True,
                        help='Folder to save individual .npy files')
    parser.add_argument('--image_size', type=int, default=224, 
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--config_path', type=str, 
                        default='/home/pabllo/dl_projects/MultimodalDataChallenge2025/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    print("🍄 Extracting DINOv2 CLS Tokens as Individual .npy Files")
    print("=" * 65)
    print(f"Model size: DINOv2 {args.model_size}")
    print(f"Save folder: {args.save_folder}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print("")
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load DINOv2 model
    print(f"Loading DINOv2 {args.model_size} model...")
    model, embed_dim = load_dinov2_model(args.model_size)
    model.to(device)
    print(f"✅ Model loaded! Embedding dimension: {embed_dim}")
    
    # Paths from config
    image_path = config['paths']['image_path']
    metadata_file = config['paths']['metadata_file']
    
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_file)
    print(f"Total samples in metadata: {len(df)}")
    
    # Use all images (no split distinction)
    print(f"Processing all {len(df)} images...")
    
    # Transforms
    transforms = get_transforms(args.image_size)
    
    # Create dataset and dataloader
    dataset = FungiDataset(df, image_path, transform=transforms)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Keep original order
        num_workers=4,
        pin_memory=True
    )
    
    # Extract features and save as .npy files
    processed_count = extract_features_and_save(
        model, dataloader, device, args.save_folder, args.model_size
    )
    
    # Create summary
    summary_file = create_feature_summary(args.save_folder, args.model_size, processed_count)
    
    print(f"\n🎉 Feature extraction completed!")
    print(f"📁 Save folder: {args.save_folder}")
    print(f"� Total files: {processed_count}")
    print(f"📋 Summary: {summary_file}")
    print(f"\n💡 Usage example:")
    print(f"   import numpy as np")
    print(f"   features = np.load('{args.save_folder}/fungi_train000001.npy')")
    print(f"   print(features.shape)  # Should be ({embed_dim},)")

if __name__ == "__main__":
    main()
