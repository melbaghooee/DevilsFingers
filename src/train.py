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

def train_fungi_network(data_file, image_path, checkpoint_dir):
    """
    Train the DINOv2 + Linear network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
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

    # Initialize DataLoaders
    train_dataset = FungiDataset(train_df, image_path, transform=get_transforms(data='train'))
    valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms(data='valid'))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DINOv2 + Linear model
    num_classes = len(train_df['taxonID_index'].unique())
    model = DinoV2Linear(num_classes=num_classes, dinov2_model_name='dinov2_vitg14')
    model.to(device)

    # Define Optimization, Scheduler, and Criterion
    # Only optimize the Linear parameters (DINOv2 is frozen)
    optimizer = Adam(model.classifier.parameters(), lr=0.001)
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
        for images, labels, _ in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
            for images, labels, _ in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name):
    """
    Evaluate DINOv2 + Linear network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the DINOv2 + Linear model
    model = DinoV2Linear(num_classes=183, dinov2_model_name='dinov2_vitg14')  # Number of classes
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    # Set seed for reproducibility
    seed_torch(777)
    
    # Path to fungi images
    image_path = ROOT_DIR / 'data' / 'FungiImages'
    # Path to metadata file
    data_file = ROOT_DIR / 'data' / 'metadata' / 'metadata.csv'

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = "DinoV2VitG14Linear"

    # Folder for results of this experiment based on session name:
    checkpoint_dir = ROOT_DIR / 'results' / session
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training with DINOv2 + Linear model...")
    print(f"Session: {session}")
    print(f"Results will be saved to: {checkpoint_dir}")
    
    train_fungi_network(data_file, image_path, checkpoint_dir)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)
