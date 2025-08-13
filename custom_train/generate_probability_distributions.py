#!/usr/bin/env python3
"""
Generate Full Probability Distributions for Validation Samples

This script loads a trained DINOv2 model and generates the complete probability 
distribution across all classes for each validation sample, saving the results to a CSV file.

Usage:
    python src/generate_probability_distributions.py --model_path <path> --output_path <path>
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
import json
import argparse
from sklearn.model_selection import train_test_split

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
        # Use the mapped label instead of the original taxonID_index
        label = self.df['label'].values[idx] if 'label' in self.df.columns else self.df['taxonID_index'].values[idx]
        
        if pd.isnull(label):
            label = -1  # Handle missing labels
        else:
            label = int(label)
        
        # Load image
        with Image.open(os.path.join(self.image_path, file_path)) as img:
            image = img.convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label, idx

class DINOv2Model(nn.Module):
    """DINOv2 model for fungi classification"""
    def __init__(self, num_classes, model_size='base'):
        super(DINOv2Model, self).__init__()
        
        # Load DINOv2 backbone
        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14'
        }
        
        model_name = model_map[model_size]
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        
        # Get embedding dimension
        embed_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        
        embed_dim = embed_dims[model_size]
        
        # Classification head - match the original training architecture
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  # Match the dropout rate from training
            nn.Linear(embed_dim, num_classes)
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Get features from DINOv2
        with torch.no_grad():
            features = self.backbone(x)
        
        # Apply classification head
        output = self.classifier(features)
        return output

def get_validation_transforms(image_size):
    """Get validation transforms"""
    return Compose([
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def generate_probability_distributions(model, dataloader, device, class_names, val_split):
    """
    Generate full probability distributions for each validation sample
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Computing device
        class_names: List of class names (original taxonID_index values)
        val_split: Validation split DataFrame
    
    Returns:
        DataFrame with probability distributions
    """
    model.eval()
    all_probabilities = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, (data, target, indices) in enumerate(tqdm(dataloader, desc="Generating Probability Distributions")):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            
            # Process each sample in the batch
            for i, (probs, true_label, sample_idx) in enumerate(zip(probabilities, target, indices)):
                probs_np = probs.cpu().numpy()
                true_label_np = true_label.cpu().item()
                sample_idx_np = sample_idx.cpu().item()
                
                # Get sample metadata
                sample_row = val_split.iloc[sample_idx_np]
                
                # Store probabilities
                all_probabilities.append(probs_np)
                
                # Store metadata
                metadata = {
                    'sample_index': sample_idx_np,
                    'filename': sample_row['filename_index'],
                    'true_label': class_names[true_label_np],
                    'true_label_index': true_label_np,
                    'predicted_label': class_names[np.argmax(probs_np)],
                    'predicted_label_index': int(np.argmax(probs_np)),
                    'max_probability': float(np.max(probs_np)),
                    'true_label_probability': float(probs_np[true_label_np]),
                    'is_correct': bool(np.argmax(probs_np) == true_label_np),
                    'entropy': float(-np.sum(probs_np * np.log(probs_np + 1e-10)))
                }
                
                # Add additional metadata if available
                metadata_cols = ['scientificName', 'family', 'genus', 'species', 'locality', 'country']
                for col in metadata_cols:
                    if col in sample_row.index and pd.notna(sample_row[col]):
                        metadata[col] = str(sample_row[col])
                    else:
                        metadata[col] = ''
                
                all_metadata.append(metadata)
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    probabilities_array = np.array(all_probabilities)
    
    # Create probability columns
    prob_columns = {}
    for i, class_name in enumerate(class_names):
        prob_columns[f'class_{class_name}_prob'] = probabilities_array[:, i]
    
    prob_df = pd.DataFrame(prob_columns)
    
    # Combine metadata and probabilities
    result_df = pd.concat([metadata_df, prob_df], axis=1)
    
    return result_df

def save_results(prob_df, output_path, model_name):
    """Save probability distributions to CSV and generate summary statistics"""
    
    # Save to CSV
    csv_path = output_path
    prob_df.to_csv(csv_path, index=False)
    
    print(f"✅ Probability distributions saved to: {csv_path}")
    
    # Generate summary statistics
    summary_stats = {
        'total_samples': len(prob_df),
        'num_classes': len([col for col in prob_df.columns if col.endswith('_prob')]),
        'correct_predictions': int(prob_df['is_correct'].sum()),
        'accuracy': float(prob_df['is_correct'].mean()),
        'average_entropy': float(prob_df['entropy'].mean()),
        'average_max_probability': float(prob_df['max_probability'].mean()),
        'average_true_label_probability': float(prob_df['true_label_probability'].mean()),
        'model_name': model_name,
        'confidence_thresholds': {
            'very_confident': float((prob_df['max_probability'] > 0.9).sum() / len(prob_df)),
            'confident': float((prob_df['max_probability'] > 0.8).sum() / len(prob_df)),
            'moderate': float((prob_df['max_probability'] > 0.7).sum() / len(prob_df)),
            'uncertain': float((prob_df['max_probability'] <= 0.5).sum() / len(prob_df))
        }
    }
    
    # Calculate per-class statistics
    prob_cols = [col for col in prob_df.columns if col.endswith('_prob')]
    class_stats = {}
    for col in prob_cols:
        class_name = col.replace('class_', '').replace('_prob', '')
        class_mask = prob_df['true_label'] == int(class_name)
        if class_mask.sum() > 0:
            class_stats[class_name] = {
                'sample_count': int(class_mask.sum()),
                'accuracy': float(prob_df[class_mask]['is_correct'].mean()),
                'avg_true_prob': float(prob_df[class_mask]['true_label_probability'].mean()),
                'avg_max_prob': float(prob_df[class_mask]['max_probability'].mean())
            }
    
    summary_stats['per_class_stats'] = class_stats
    
    # Save summary
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"📊 Summary statistics saved to: {summary_path}")
    print(f"├── Total samples: {summary_stats['total_samples']}")
    print(f"├── Number of classes: {summary_stats['num_classes']}")
    print(f"├── Accuracy: {summary_stats['accuracy']:.4f}")
    print(f"├── Average Entropy: {summary_stats['average_entropy']:.4f}")
    print(f"├── Average Max Probability: {summary_stats['average_max_probability']:.4f}")
    print(f"├── Very Confident (>0.9): {summary_stats['confidence_thresholds']['very_confident']:.3f}")
    print(f"├── Confident (>0.8): {summary_stats['confidence_thresholds']['confident']:.3f}")
    print(f"└── Uncertain (≤0.5): {summary_stats['confidence_thresholds']['uncertain']:.3f}")
    
    return csv_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='Generate Full Probability Distributions for Validation Samples')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save probability distributions CSV')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large', 'giant'],
                        help='DINOv2 model size')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--config_path', type=str, 
                        default='/home/pabllo/dl_projects/MultimodalDataChallenge2025/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    print("🍄 Generating Probability Distributions for Validation Samples")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Model size: DINOv2 {args.model_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print("")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model checkpoint not found at {args.model_path}")
        return
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths from config
    image_path = config['paths']['image_path']
    metadata_file = config['paths']['metadata_file']
    
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_file)
    
    # Filter only training data (since we'll split it into train/val)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    
    # Get unique classes from taxonID_index and create label mapping
    unique_classes = sorted([int(x) for x in train_df['taxonID_index'].dropna().unique()])
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    # Map taxonID_index to sequential indices
    train_df = train_df.copy()
    train_df['label'] = train_df['taxonID_index'].map(class_to_idx)
    
    # Remove any rows with missing labels
    train_df = train_df.dropna(subset=['taxonID_index'])
    
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Total samples: {len(train_df)}")
    
    # Split data (80% train, 20% validation) - same split as used in training
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['taxonID_index']
    )
    
    print(f"Validation samples: {len(val_split)}")
    
    # Reset index for validation split to ensure proper indexing
    val_split = val_split.reset_index(drop=True)
    
    # Validation transforms
    val_transforms = get_validation_transforms(args.image_size)
    
    # Validation dataset and dataloader
    val_dataset = FungiDataset(val_split, image_path, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize and load model
    print(f"Initializing DINOv2 {args.model_size} model...")
    model = DINOv2Model(len(unique_classes), args.model_size)
    
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    print("✅ Model loaded successfully!")
    
    # Generate probability distributions
    print("\nGenerating probability distributions...")
    prob_df = generate_probability_distributions(model, val_loader, device, unique_classes, val_split)
    
    # Save results
    print("\nSaving results...")
    csv_path, summary_path = save_results(prob_df, args.output_path, f"dinov2_{args.model_size}")
    
    print("\n🎉 Probability distribution generation completed!")
    print(f"📄 CSV file: {csv_path}")
    print(f"📊 Summary: {summary_path}")
    print(f"\nCSV Structure:")
    print(f"├── Metadata columns: sample_index, filename, true_label, predicted_label, etc.")
    print(f"├── Probability columns: class_0_prob, class_1_prob, ..., class_{len(unique_classes)-1}_prob")
    print(f"└── Total columns: {len(prob_df.columns)}")

if __name__ == "__main__":
    main()
