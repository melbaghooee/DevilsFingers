import os
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
import yaml
from seesaw_loss import SeesawLoss

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_folder(folder):
    """Ensure a folder exists; if not, create it."""
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    """Set seed for reproducibility."""
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

def analyze_class_distribution(train_df):
    """Analyze and print class distribution statistics."""
    class_counts = Counter(train_df['taxonID_index'].values)
    total_samples = len(train_df)
    
    print(f"\nClass Distribution Analysis:")
    print(f"Total classes: {len(class_counts)}")
    print(f"Total samples: {total_samples}")
    print(f"Min samples per class: {min(class_counts.values())}")
    print(f"Max samples per class: {max(class_counts.values())}")
    print(f"Mean samples per class: {np.mean(list(class_counts.values())):.2f}")
    print(f"Median samples per class: {np.median(list(class_counts.values())):.2f}")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    return class_counts

def get_transforms(data):
    """Return augmentation transforms for the specified mode."""
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
        label = self.df['taxonID_index'].values[idx]
        if pd.isnull(label):
            label = -1
        else:
            label = int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = img.convert('RGB')
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, file_path

class DINOv2FungiClassifier(nn.Module):
    """DINOv2-based classifier with simple linear head for Seesaw Loss."""
    
    def __init__(self, num_classes, model_name='dinov2_vits14', dropout_rate=0.2):
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
            
        # Simple linear classifier head for Seesaw Loss
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # type: ignore
        output = self.classifier(features)
        return output

def train_fungi_network_seesaw(data_file, image_path, checkpoint_dir, dinov2_model='dinov2_vitg14'):
    """
    Train the network with Seesaw Loss for handling class imbalance.
    
    Args:
        data_file: Path to the metadata CSV file
        image_path: Path to the directory containing images
        checkpoint_dir: Directory to save model checkpoints
        dinov2_model: DINOv2 model variant
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['taxonID_index'])
    
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    # Analyze class distribution for Seesaw Loss
    class_counts = analyze_class_distribution(train_df)

    # Initialize DataLoaders
    train_dataset = FungiDataset(train_df, image_path, transform=get_transforms(data='train'))
    valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms(data='valid'))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=16)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_df['taxonID_index'].unique())
    
    model = DINOv2FungiClassifier(num_classes=num_classes, model_name=dinov2_model, dropout_rate=0.2)
    model.to(device)
    
    print(f"Using {dinov2_model} with Seesaw Loss and simple linear classifier")
    print(f"Model moved to device: {device}")
    print(f"Number of classes: {num_classes}")

    # Seesaw Loss for handling class imbalance
    criterion = SeesawLoss(num_classes=num_classes, p=0.8, q=2.0).to(device)
    
    # Optimizer and scheduler
    optimizer = Adam(model.classifier.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2, verbose=True, eps=1e-6)

    # Early stopping setup
    patience = 15  # Increased patience for Seesaw Loss
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Training Loop
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        epoch_start_time = time.time()
        
        # Training Loop
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/100 [Train-Seesaw]", leave=False)
        for images, labels, _ in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_correct = (outputs.argmax(1) == labels).sum().item()
            total_correct_train += batch_correct
            total_train_samples += labels.size(0)
            
            current_train_acc = total_correct_train / total_train_samples
            current_train_loss = train_loss / (train_pbar.n + 1)
            train_pbar.set_postfix({
                'loss': f'{current_train_loss:.4f}',
                'acc': f'{current_train_acc:.4f}'
            })
        
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        val_pbar = tqdm.tqdm(valid_loader, desc=f"Epoch {epoch + 1}/100 [Valid-Seesaw]", leave=False)
        with torch.no_grad():
            for images, labels, _ in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Use standard CrossEntropy for validation to get comparable metrics
                val_criterion = nn.CrossEntropyLoss()
                batch_val_loss = val_criterion(outputs, labels).item()
                val_loss += batch_val_loss
                
                batch_correct = (outputs.argmax(1) == labels).sum().item()
                total_correct_val += batch_correct
                total_val_samples += labels.size(0)
                
                current_val_acc = total_correct_val / total_val_samples
                current_val_loss = val_loss / (val_pbar.n + 1)
                val_pbar.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{current_val_acc:.4f}'
                })

        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        scheduler.step(avg_val_loss)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch + 1} Summary [Seesaw]: "
              f"Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, "
              f"Epoch Time = {epoch_time:.2f} seconds")
        
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        # Save best models
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name, dinov2_model='dinov2_vitg14'):
    """Evaluate network on the test set and save predictions."""
    ensure_folder(checkpoint_dir)

    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv2FungiClassifier(num_classes=183, model_name=dinov2_model, dropout_rate=0.2)
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            outputs = model(images).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])
        writer.writerows(results)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    # Load configuration
    config_path = "/home/pabllo/dl_projects/MultimodalDataChallenge2025/config.yaml"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    image_path = config['paths']['image_path']
    data_file = config['paths']['metadata_file']
    results_dir = config['paths']['results_dir']
    
    dinov2_model = "dinov2_vitg14"
    session = f"DINOv2_{dinov2_model.split('_')[1].upper()}_seesaw_224_224"
    checkpoint_dir = os.path.join(results_dir, session)

    seed_torch(777)

    print(f"Using Seesaw Loss for imbalanced fungi classification")
    print(f"Session: {session}")

    train_fungi_network_seesaw(data_file, image_path, checkpoint_dir, dinov2_model)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session, dinov2_model)
