import os
from pathlib import Path
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter
import h5py
from torchvision import transforms
import argparse
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent

def ensure_folder(folder):
    """
    Ensure a folder exists; if not, create it.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_csv_logger(file_path):
    """Initialize the CSV file with header."""
    header = ["epoch", "time", "val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    """Log epoch summary to the CSV file."""
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy])

def get_dinov2_feature_dim(model_name):
    """Get the feature dimension for a given DINOv2 model."""
    if 'vits14' in model_name:
        return 384
    elif 'vitb14' in model_name:
        return 768
    elif 'vitl14' in model_name:
        return 1024
    elif 'vitg14' in model_name:
        return 1536
    else:
        raise ValueError(f"Unknown DINOv2 model: {model_name}")

def extract_dinov2_features(df, image_path, features_file, dinov2_model_name='dinov2_vitg14'):
    """
    Extract DINOv2 features for all images and save them to HDF5 file.
    """
    if os.path.exists(features_file):
        print(f"Features file {features_file} already exists. Skipping feature extraction.")
        return
    
    print(f"Extracting DINOv2 features to {features_file}...")
    
    # Load DINOv2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
    dinov2_model.eval()
    dinov2_model.to(device)
    
    # Freeze all parameters
    for param in dinov2_model.parameters():
        param.requires_grad = False
    
    # Get feature dimension
    feature_dim = get_dinov2_feature_dim(dinov2_model_name)
    
    # Preprocessing for DINOv2
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    filenames = df['filename_index'].values
    labels = df['taxonID_index'].values
    
    # Create HDF5 file
    with h5py.File(features_file, 'w') as h5f:
        features_ds = h5f.create_dataset('features', shape=(len(filenames), feature_dim), dtype='float32')
        labels_ds = h5f.create_dataset('labels', shape=(len(filenames),), dtype='int32')
        filenames_ds = h5f.create_dataset('filenames', shape=(len(filenames),), dtype=h5py.string_dtype())
        
        with torch.no_grad():
            for i, (fname, label) in enumerate(tqdm.tqdm(zip(filenames, labels), total=len(filenames), desc="Extracting features")):
                # Load and preprocess image
                img_path = os.path.join(image_path, fname)
                image = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Extract features
                features = dinov2_model(img_tensor).cpu().numpy().squeeze()
                
                # Store in HDF5
                features_ds[i] = features
                labels_ds[i] = int(label) if not pd.isnull(label) else -1
                filenames_ds[i] = fname
    
    print(f"Features saved to {features_file}")
    
    # Clean up GPU memory
    del dinov2_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop(width, height, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df['taxonID_index'].values[idx]  # Get label
        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            # Convert to RGB mode (handles grayscale images as well)
            image = img.convert('RGB')
        image = np.array(image)

        # Apply transformations if available
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, file_path

class FungiFeatureDataset(Dataset):
    """
    Dataset that loads pre-computed DINOv2 features from HDF5 file.
    """
    def __init__(self, features_file):
        self.features_file = features_file
        with h5py.File(features_file, 'r') as h5f:
            self.length = h5f['features'].shape[0]
            self.filenames = [fname.decode() for fname in h5f['filenames'][:]]
            self.labels = h5f['labels'][:]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.features_file, 'r') as h5f:
            features = torch.tensor(h5f['features'][idx], dtype=torch.float32)
            label = int(self.labels[idx])
            filename = self.filenames[idx]
        return features, label, filename

class DinoV2Linear(nn.Module):
    """
    DINOv2-based model with a trainable linear classifier.
    Uses pre-trained DINOv2 model to extract features and passes them to a linear layer.
    """
    def __init__(self, num_classes=183, dinov2_model_name='dinov2_vits14'):
        super(DinoV2Linear, self).__init__()
        
        # Load pre-trained DINOv2 model
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', dinov2_model_name)
        
        # Freeze DINOv2 parameters
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # Get the feature dimension from DINOv2 model
        # For DINOv2 models, the feature dimensions are:
        if 'vits14' in dinov2_model_name:
            feature_dim = 384
        elif 'vitb14' in dinov2_model_name:
            feature_dim = 768
        elif 'vitl14' in dinov2_model_name:
            feature_dim = 1024
        elif 'vitg14' in dinov2_model_name:
            feature_dim = 1536
        else:
            feature_dim = 384  # Default fallback
        
        # Define the linear classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        # Extract features using DINOv2 (frozen)
        with torch.no_grad():
            features = self.dinov2(x)
        
        # Pass features through the trainable linear classifier
        output = self.classifier(features)
        return output

class LinearClassifier(nn.Module):
    """
    Simple linear classifier for pre-computed features.
    """
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features):
        return self.classifier(features)

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for pre-computed features.
    Uses multi-head self-attention to learn feature relationships.
    """
    def __init__(self, feature_dim, num_classes, num_heads=1, num_layers=1, hidden_dim=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        # Input projection to match transformer hidden dimension
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding (not needed for single feature vector, but kept for extensibility)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features):
        # Project input features to hidden dimension
        x = self.input_projection(features)  # [batch_size, hidden_dim]
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1) + self.pos_encoding  # [batch_size, 1, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, 1, hidden_dim]
        
        # Remove sequence dimension and classify
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        output = self.classifier(x)
        
        return output

def load_features_and_labels(features_file):
    """Load features and labels from HDF5 file into numpy arrays."""
    with h5py.File(features_file, 'r') as h5f:
        features = h5f['features'][:]
        labels = h5f['labels'][:]
        filenames = [fname.decode() for fname in h5f['filenames'][:]]
    return features, labels, filenames

def train_xgboost_classifier(train_features_file, val_features_file, checkpoint_dir):
    """Train XGBoost classifier on pre-computed features."""
    print("=== Training XGBoost Classifier ===")
    
    # Load training data
    X_train, y_train, _ = load_features_and_labels(train_features_file)
    X_val, y_val, _ = load_features_and_labels(val_features_file)
    
    # Remove samples with invalid labels
    valid_train_mask = y_train >= 0
    valid_val_mask = y_val >= 0
    
    X_train = X_train[valid_train_mask]
    y_train = y_train[valid_train_mask]
    X_val = X_val[valid_val_mask]
    y_val = y_val[valid_val_mask]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=777,
        early_stopping_rounds=50,
        eval_metric='mlogloss',
        n_jobs=4
    )
    
    # Train the model
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Evaluate on validation set
    val_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save the model
    model_path = os.path.join(checkpoint_dir, "best_xgboost_model.pkl")
    joblib.dump(xgb_model, model_path)
    print(f"XGBoost model saved to {model_path}")
    
    return xgb_model

def evaluate_xgboost_on_test_set(test_features_file, checkpoint_dir, session_name):
    """Evaluate XGBoost classifier on test set."""
    print("=== Evaluating XGBoost on Test Set ===")
    
    # Load the trained model
    model_path = os.path.join(checkpoint_dir, "best_xgboost_model.pkl")
    xgb_model = joblib.load(model_path)
    
    # Load test data
    X_test, _, filenames = load_features_and_labels(test_features_file)
    
    # Make predictions
    test_pred = xgb_model.predict(X_test)
    
    # Save results
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(zip(filenames, test_pred))  # Write filenames and predictions
    
    print(f"XGBoost results saved to {output_csv_path}")

def train_transformer_classifier(train_features_file, val_features_file, checkpoint_dir, feature_dim, num_classes):
    """Train Transformer classifier on pre-computed features."""
    print("=== Training Transformer Classifier ===")
    
    # Initialize DataLoaders with pre-computed features
    train_dataset = FungiFeatureDataset(train_features_file)
    valid_dataset = FungiFeatureDataset(val_features_file)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Transformer model
    model = TransformerClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_heads=8,
        num_layers=3,
        hidden_dim=512,
        dropout=0.1
    )
    model.to(device)
    
    # Define Optimization, Scheduler, and Criterion
    optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    # Training setup
    patience = 15
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0
    max_epochs = 100
    
    # Initialize CSV logger
    csv_file_path = os.path.join(checkpoint_dir, 'transformer_train.csv')
    initialize_csv_logger(csv_file_path)
    
    # Training Loop
    epoch_pbar = tqdm.tqdm(range(max_epochs), desc="Transformer Training Progress")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Training Loop
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Training", leave=False)
        for features, labels, _ in train_pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate train accuracy
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)
            
            # Update progress bar with current metrics
            current_train_acc = total_correct_train / total_train_samples
            current_train_loss = train_loss / (train_pbar.n + 1)
            train_pbar.set_postfix({
                'Loss': f'{current_train_loss:.4f}', 
                'Acc': f'{current_train_acc:.4f}'
            })
        
        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        with torch.no_grad():
            val_pbar = tqdm.tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Validation", leave=False)
            for features, labels, _ in val_pbar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                
                # Calculate validation accuracy
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)
                
                # Update progress bar with current metrics
                current_val_acc = total_correct_val / total_val_samples
                current_val_loss = val_loss / (val_pbar.n + 1)
                val_pbar.set_postfix({
                    'Loss': f'{current_val_loss:.4f}', 
                    'Acc': f'{current_val_acc:.4f}'
                })

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Update main epoch progress bar
        epoch_pbar.set_postfix({
            'Train_Loss': f'{avg_train_loss:.4f}',
            'Train_Acc': f'{train_accuracy:.4f}', 
            'Val_Loss': f'{avg_val_loss:.4f}',
            'Val_Acc': f'{val_accuracy:.4f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # Log epoch metrics to the CSV file
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_transformer_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best transformer accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_transformer_loss.pth"))
            print(f"Epoch {epoch + 1}: Best transformer loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Early stopping condition
        if patience_counter >= patience:
            epoch_pbar.set_description("Training stopped early")
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break
    
    # Close the epoch progress bar
    epoch_pbar.close()
    print("=== Transformer Training Complete ===")

def evaluate_transformer_on_test_set(test_features_file, checkpoint_dir, session_name, feature_dim, num_classes):
    """Evaluate Transformer classifier on test set."""
    print("=== Evaluating Transformer on Test Set ===")
    
    # Load test data
    test_dataset = FungiFeatureDataset(test_features_file)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the Transformer model
    model = TransformerClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_heads=8,
        num_layers=3,
        hidden_dim=512,
        dropout=0.1
    )
    
    # Load the best model
    best_model_path = os.path.join(checkpoint_dir, "best_transformer_accuracy.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()
    
    # Make predictions
    results = []
    with torch.no_grad():
        for features, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating Transformer"):
            features = features.to(device)
            outputs = model(features).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))
    
    # Save results
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    
    print(f"Transformer results saved to {output_csv_path}")

def train_fungi_network(data_file, image_path, checkpoint_dir, dinov2_model_name='dinov2_vitg14', classifier_type='linear'):
    """
    Train the DINOv2 + classifier and save the best models based on validation accuracy and loss.
    Supports both linear and XGBoost classifiers.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    # Feature extraction step - extract and save DINOv2 features
    features_dir = os.path.join(ROOT_DIR, 'data', 'features', dinov2_model_name)
    ensure_folder(features_dir)
    
    train_features_file = os.path.join(features_dir, 'train_features.h5')
    val_features_file = os.path.join(features_dir, 'val_features.h5')
    
    print("=== Feature Extraction Phase ===")
    extract_dinov2_features(train_df, image_path, train_features_file, dinov2_model_name)
    extract_dinov2_features(val_df, image_path, val_features_file, dinov2_model_name)
    print("=== Feature Extraction Complete ===")

    # Branch based on classifier type
    if classifier_type == 'xgboost':
        # Train XGBoost classifier
        train_xgboost_classifier(train_features_file, val_features_file, checkpoint_dir)
        return
    elif classifier_type == 'transformer':
        # Train Transformer classifier
        num_classes = len(train_df['taxonID_index'].unique())
        feature_dim = get_dinov2_feature_dim(dinov2_model_name)
        train_transformer_classifier(train_features_file, val_features_file, checkpoint_dir, feature_dim, num_classes)
        return
    
    # Continue with linear classifier training
    # Initialize DataLoaders with pre-computed features
    train_dataset = FungiFeatureDataset(train_features_file)
    valid_dataset = FungiFeatureDataset(val_features_file)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create Linear classifier model (features are pre-computed)
    num_classes = len(train_df['taxonID_index'].unique())
    feature_dim = get_dinov2_feature_dim(dinov2_model_name)
    model = LinearClassifier(feature_dim, num_classes)
    model.to(device)

    # Define Optimization, Scheduler, and Criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Training Loop
    max_epochs = 100
    epoch_pbar = tqdm.tqdm(range(max_epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Training Loop
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Training", leave=False)
        for features, labels, _ in train_pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate train accuracy
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)
            
            # Update progress bar with current metrics
            current_train_acc = total_correct_train / total_train_samples
            current_train_loss = train_loss / (train_pbar.n + 1)  # +1 for current batch
            train_pbar.set_postfix({
                'Loss': f'{current_train_loss:.4f}', 
                'Acc': f'{current_train_acc:.4f}'
            })
        
        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        # Validation Loop
        with torch.no_grad():
            val_pbar = tqdm.tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{max_epochs} - Validation", leave=False)
            for features, labels, _ in val_pbar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                
                # Calculate validation accuracy
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)
                
                # Update progress bar with current metrics
                current_val_acc = total_correct_val / total_val_samples
                current_val_loss = val_loss / (val_pbar.n + 1)  # +1 for current batch
                val_pbar.set_postfix({
                    'Loss': f'{current_val_loss:.4f}', 
                    'Acc': f'{current_val_acc:.4f}'
                })

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Update main epoch progress bar
        epoch_pbar.set_postfix({
            'Train_Loss': f'{avg_train_loss:.4f}',
            'Train_Acc': f'{train_accuracy:.4f}', 
            'Val_Loss': f'{avg_val_loss:.4f}',
            'Val_Acc': f'{val_accuracy:.4f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # Log epoch metrics to the CSV file
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Early stopping condition
        if patience_counter >= patience:
            epoch_pbar.set_description("Training stopped early")
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break
    
    # Close the epoch progress bar
    epoch_pbar.close()

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name, dinov2_model_name='dinov2_vitg14', classifier_type='linear'):
    """
    Evaluate classifier on the test set and save predictions to a CSV file.
    Supports both linear and XGBoost classifiers.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    
    # Extract features for test set
    features_dir = os.path.join(ROOT_DIR, 'data', 'features', dinov2_model_name)
    ensure_folder(features_dir)
    test_features_file = os.path.join(features_dir, 'test_features.h5')
    
    print("=== Test Feature Extraction ===")
    extract_dinov2_features(test_df, image_path, test_features_file, dinov2_model_name)
    print("=== Test Feature Extraction Complete ===")
    
    # Branch based on classifier type
    if classifier_type == 'xgboost':
        # Evaluate XGBoost classifier
        evaluate_xgboost_on_test_set(test_features_file, checkpoint_dir, session_name)
        return
    elif classifier_type == 'transformer':
        # Evaluate Transformer classifier
        feature_dim = get_dinov2_feature_dim(dinov2_model_name)
        evaluate_transformer_on_test_set(test_features_file, checkpoint_dir, session_name, feature_dim, num_classes=183)
        return
    
    # Continue with linear classifier evaluation
        return
    
    # Continue with linear classifier evaluation
    test_dataset = FungiFeatureDataset(test_features_file)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the Linear classifier model
    feature_dim = get_dinov2_feature_dim(dinov2_model_name)
    model = LinearClassifier(feature_dim, num_classes=183)
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for features, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)
            outputs = model(features).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DINOv2 + classifier for fungi classification")
    parser.add_argument('--dinov2_model', type=str, default='dinov2_vitg14',
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DINOv2 model size to use (default: dinov2_vitg14)')
    parser.add_argument('--classifier', type=str, default='linear',
                        choices=['linear', 'xgboost', 'transformer'],
                        help='Classifier type to use (default: linear)')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to metadata CSV file (default: data/metadata/metadata.csv)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to fungi images directory (default: data/FungiImages)')
    parser.add_argument('--session', type=str, default=None,
                        help='Session name for experiment (default: auto-generated based on model)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_torch(777)
    
    # Set default paths if not provided
    if args.data_file is None:
        data_file = ROOT_DIR / 'data' / 'metadata' / 'metadata.csv'
    else:
        data_file = Path(args.data_file)
        
    if args.image_path is None:
        image_path = ROOT_DIR / 'data' / 'FungiImages'
    else:
        image_path = Path(args.image_path)

    # Set session name based on model if not provided
    if args.session is None:
        model_size = args.dinov2_model.replace('dinov2_', '').upper()
        classifier_name = args.classifier.capitalize()
        session = f"DinoV2{model_size}{classifier_name}"
    else:
        session = args.session

    # Folder for results of this experiment based on session name:
    checkpoint_dir = ROOT_DIR / 'results' / session
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training with DINOv2 + {args.classifier.capitalize()} classifier...")
    print(f"DINOv2 Model: {args.dinov2_model}")
    print(f"Classifier: {args.classifier}")
    print(f"Session: {session}")
    print(f"Data file: {data_file}")
    print(f"Image path: {image_path}")
    print(f"Results will be saved to: {checkpoint_dir}")
    
    train_fungi_network(data_file, image_path, checkpoint_dir, args.dinov2_model, args.classifier)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session, args.dinov2_model, args.classifier)
