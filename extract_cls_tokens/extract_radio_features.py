#!/usr/bin/env python3
"""
Extract RADIO v2.5 features for all images in the dataset.

This script loads the RADIO v2.5 model and extracts normalized features
for each image in the dataset. Features are saved as individual .npy files
with filenames matching the original images.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import time

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd
import numpy as np
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def validate_paths(config: Dict[str, Any]) -> None:
    """Validate that required paths exist."""
    paths_to_check = [
        ('image_path', config['paths']['image_path']),
        ('metadata_file', config['paths']['metadata_file'])
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            logging.error(f"{name} does not exist: {path}")
            sys.exit(1)
        logging.info(f"✓ {name}: {path}")


def load_radio_model(version: str = 'radio_v2.5-g', device: str = 'cuda') -> torch.nn.Module:
    """
    Load RADIO v2.5 model.
    
    Args:
        version: RADIO model version ('radio_v2.5-b', 'radio_v2.5-h', 'radio_v2.5-g')
        device: Device to load model on
        
    Returns:
        Loaded RADIO model
    """
    try:
        logging.info(f"Loading RADIO model: {version}")
        model = torch.hub.load('NVlabs/RADIO', 'radio_model',
                              version=version, progress=True, skip_validation=True)
        model = model.to(device).eval()
        logging.info(f"✓ RADIO model loaded successfully on {device}")
        return model
    except Exception as e:
        logging.error(f"Failed to load RADIO model: {e}")
        logging.error("Please ensure you have the correct timm version (pip install timm>=0.9.7)")
        sys.exit(1)


def preprocess_image(image_path: str, model: torch.nn.Module, device: str, 
                    target_resolution: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    Load and preprocess image for RADIO model.
    
    Args:
        image_path: Path to image file
        model: RADIO model (for getting supported resolution)
        device: Device for tensors
        target_resolution: Optional fixed resolution. If None, uses RADIO's adaptive resolution
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load image and convert to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Convert to tensor [0,1]
        x = pil_to_tensor(img).float().div_(255.0).unsqueeze(0).to(device)
        
        if target_resolution is not None:
            # Use fixed resolution (square)
            target_size = (target_resolution, target_resolution)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:
            # Use RADIO's adaptive resolution
            nearest_h, nearest_w = model.get_nearest_supported_resolution(*x.shape[-2:])
            if (nearest_h, nearest_w) != x.shape[-2:]:
                x = F.interpolate(x, size=(nearest_h, nearest_w), mode='bilinear', align_corners=False)
            
        return x
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {e}")
        return None


def extract_features(model: torch.nn.Module, image_tensor: torch.Tensor, device: str) -> Optional[np.ndarray]:
    """
    Extract normalized features from RADIO model.
    
    Args:
        model: RADIO model
        image_tensor: Preprocessed image tensor
        device: Device for computation
        
    Returns:
        Normalized feature vector as numpy array
    """
    try:
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            # Forward pass to get summary features
            summary, spatial = model(image_tensor, feature_fmt='NCHW')
            
            # L2-normalize the summary features for retrieval
            summary_norm = torch.nn.functional.normalize(summary, dim=-1)
            
            # Convert to numpy and return
            return summary_norm.cpu().numpy().squeeze()
    except Exception as e:
        logging.error(f"Failed to extract features: {e}")
        return None


def extract_all_features(config: Dict[str, Any], output_dir: str, 
                        version: str = 'radio_v2.5-g', 
                        batch_size: int = 1,
                        target_resolution: Optional[int] = None) -> None:
    """
    Extract RADIO features for all images and save as .npy files.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save .npy feature files
        version: RADIO model version
        batch_size: Batch size for processing (currently only supports 1)
        target_resolution: Optional fixed resolution (e.g., 448). If None, uses RADIO's adaptive resolution
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Target resolution: {target_resolution if target_resolution else 'Adaptive (RADIO determined)'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Load RADIO model
    model = load_radio_model(version, device)
    
    # Load metadata
    metadata_path = config['paths']['metadata_file']
    image_dir = config['paths']['image_path']
    
    try:
        df = pd.read_csv(metadata_path)
        logging.info(f"Loaded metadata with {len(df)} samples")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        sys.exit(1)
    
    # Process each image
    successful = 0
    failed = 0
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting RADIO features"):
        image_filename = row['filename_index']
        image_path = os.path.join(image_dir, image_filename)
        
        # Generate output filename (replace extension with .npy)
        base_name = os.path.splitext(image_filename)[0]
        output_filename = f"{base_name}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            successful += 1
            continue
        
        # Check if image exists
        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}")
            failed += 1
            continue
        
        # Preprocess image
        image_tensor = preprocess_image(image_path, model, device, target_resolution)
        if image_tensor is None:
            failed += 1
            continue
        
        # Extract features
        features = extract_features(model, image_tensor, device)
        if features is None:
            failed += 1
            continue
        
        # Save features
        try:
            np.save(output_path, features)
            successful += 1
        except Exception as e:
            logging.error(f"Failed to save features for {image_filename}: {e}")
            failed += 1
    
    # Summary
    elapsed_time = time.time() - start_time
    total_images = len(df)
    
    logging.info("=" * 60)
    logging.info("RADIO FEATURE EXTRACTION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total images: {total_images}")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Success rate: {successful/total_images*100:.1f}%")
    logging.info(f"Total time: {elapsed_time:.1f} seconds")
    logging.info(f"Average time per image: {elapsed_time/total_images:.2f} seconds")
    logging.info(f"Output directory: {output_dir}")
    
    # Feature info
    if successful > 0:
        # Load one feature file to check dimensions
        sample_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        if sample_files:
            sample_path = os.path.join(output_dir, sample_files[0])
            sample_features = np.load(sample_path)
            logging.info(f"Feature dimensions: {sample_features.shape}")
            logging.info(f"Feature dtype: {sample_features.dtype}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract RADIO v2.5 features for fungi dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to save extracted features (.npy files)'
    )
    parser.add_argument(
        '--version', 
        type=str, 
        choices=['radio_v2.5-b', 'radio_v2.5-h', 'radio_v2.5-g'],
        default='radio_v2.5-g',
        help='RADIO model version'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing (currently only supports 1)'
    )
    parser.add_argument(
        '--target_resolution',
        type=int,
        default=None,
        help='Fixed resolution for all images (e.g., 448). If not specified, uses RADIO adaptive resolution'
    )
    parser.add_argument(
        '--log_level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logging.info("Starting RADIO feature extraction")
    logging.info(f"Arguments: {vars(args)}")
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_paths(config)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        logging.warning("CUDA not available, using CPU (will be slow)")
    
    # Extract features
    extract_all_features(
        config=config,
        output_dir=args.output_dir,
        version=args.version,
        batch_size=args.batch_size,
        target_resolution=args.target_resolution
    )
    
    logging.info("RADIO feature extraction completed")


if __name__ == "__main__":
    main()
