import os
from pathlib import Path
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
from PIL import Image
import time
import csv
import h5py
from torchvision import transforms
import argparse
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from seesaw_loss import SeesawLoss

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
    

def load_and_preprocess_metadata(data_file):
    """
    Load and preprocess metadata using sklearn preprocessing utilities.
    """
    
    df = pd.read_csv(data_file)
    
    # Define column types
    categorical_columns = ["Habitat", "Substrate"]
    numeric_columns = ["Latitude", "Longitude"]
    date_column = "eventDate"
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Handle date column
    df_processed[date_column] = pd.to_datetime(df_processed[date_column], errors="coerce")
    df_processed["eventYear"] = df_processed[date_column].dt.year
    df_processed["eventMonth"] = df_processed[date_column].dt.month
    df_processed["eventDay"] = df_processed[date_column].dt.day
    
    # Add derived date features to numeric columns
    date_features = ["eventYear", "eventMonth", "eventDay"]
    all_numeric_columns = numeric_columns + date_features
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            # Categorical features: impute missing values and one-hot encode
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
            # Numeric features: impute missing values and standardize
            ('num', StandardScaler(), all_numeric_columns)
        ],
        remainder='passthrough'  # Keep other columns as-is
    )
    
    # Fit and transform the features
    feature_columns = categorical_columns + all_numeric_columns
    X = df_processed[feature_columns]
    
    # Handle missing values before preprocessing
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown', missing_values=pd.NA)
    numeric_imputer = SimpleImputer(strategy='mean')
    
    # Apply imputation
    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
    X[all_numeric_columns] = numeric_imputer.fit_transform(X[all_numeric_columns])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Create feature names for the processed data
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
    num_feature_names = all_numeric_columns
    feature_names = list(cat_feature_names) + num_feature_names
    
    # Create processed DataFrame
    df_features = pd.DataFrame(X_processed, columns=feature_names, index=df.index)

    # Exclude Unknowns
    cols = [c for c in df_features.columns if 'Unknown' not in c]
    df_features = df_features[cols]
    
    # Add back essential columns
    df_final = pd.concat([
        df[['filename_index', 'taxonID_index']],  # Keep original identifiers
        df_features
    ], axis=1)

    return df_final

def extract_metadata_features(df, metadata_file):
    """
    Pre-process metadata features from the DataFrame and save to a HDF5 file.
    """
    # check if metadata_file already exist
    if os.path.exists(metadata_file):
        print(f"Metadata features file {metadata_file} already exists. Skipping extraction.")
        return

    # save processed DataFrame to HDF5
    with h5py.File(metadata_file, "w") as h5f:
        h5f.create_dataset('filenames', data=df['filename_index'].values, dtype=h5py.string_dtype())

        for col in df.columns:
            if col in ['filename_index', 'taxonID_index']:
                continue
            
            h5f.create_dataset(col.replace('/', '_'), data=df[col].values.astype(np.float32), dtype='float32')

    print(f"Metadata features saved to {metadata_file}")
    
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
        # transforms.Resize((224, 224)),
        transforms.Resize((448, 448)),
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

class FungiFeatureDataset(Dataset):
    """
    Dataset that loads pre-computed DINOv2 features and optionally metadata features from HDF5 files.
    If metadata_file is provided, concatenates them into a single feature vector.
    If metadata_file is None, returns only DINOv2 features.
    """
    def __init__(self, features_file, metadata_file=None):
        self.features_file = features_file
        self.metadata_file = metadata_file
        self.use_metadata = metadata_file is not None
        
        # Load basic info from features file
        with h5py.File(features_file, 'r') as h5f:
            self.length = h5f['features'].shape[0]
            self.dinov2_filenames = [fname.decode() for fname in h5f['filenames'][:]]
            self.labels = h5f['labels'][:]
            self.dinov2_feature_dim = h5f['features'].shape[1]
        
        if self.use_metadata:
            # Load metadata info and verify compatibility
            df = {}
            with h5py.File(metadata_file, 'r') as h5f:
                for k in h5f.keys():
                    if h5f[k].dtype == h5py.string_dtype():
                        df[k] = [f.decode() for f in h5f[k][:]]
                    else:
                        df[k] = h5f[k][:]

            df = pd.DataFrame(df)

            # Verify that filenames match between both files
            metadata_filenames = df['filenames'].values
            assert (self.dinov2_filenames == metadata_filenames).all(), "Filenames must match between DINOv2 and metadata files"
            
            df = df.drop(columns=['filenames'])

            self.df_metadata = df

            # Count metadata features (excluding filenames)
            self.metadata_feature_names = [key for key in df.columns if key != 'filenames']
            self.metadata_feature_dim = len(self.metadata_feature_names)

            self.total_feature_dim = self.dinov2_feature_dim + self.metadata_feature_dim
            print(f"Combined feature dimensions: DINOv2({self.dinov2_feature_dim}) + Metadata({self.metadata_feature_dim}) = {self.total_feature_dim}")
            print(f"Metadata features: {self.metadata_feature_names}")
        else:
            # Only using DINOv2 features
            self.total_feature_dim = self.dinov2_feature_dim
            print(f"Using DINOv2 features only: dimension = {self.dinov2_feature_dim}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load DINOv2 features
        with h5py.File(self.features_file, 'r') as h5f:
            dinov2_features = torch.tensor(h5f['features'][idx], dtype=torch.float32)
            label = int(self.labels[idx])
            filename = self.dinov2_filenames[idx]
        
        if self.use_metadata:
            # Load metadata features
            metadata_features = self.df_metadata.iloc[idx].values
            metadata_features = torch.tensor(metadata_features, dtype=torch.float32)
            
            # Concatenate features
            combined_features = torch.cat([dinov2_features, metadata_features], dim=0)
            return combined_features, label, filename
        else:
            # Return only DINOv2 features
            return dinov2_features, label, filename

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
    def __init__(self, feature_dim, num_classes, num_heads=8, num_layers=3, hidden_dim=512, dropout=0.1):
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

def load_combined_features_and_labels(features_file, metadata_file):
    """Load combined DINOv2 and metadata features from HDF5 files into numpy arrays."""
    # Load DINOv2 features
    with h5py.File(features_file, 'r') as h5f:
        dinov2_features = h5f['features'][:]
        labels = h5f['labels'][:]
        filenames = [fname.decode() for fname in h5f['filenames'][:]]
    
    # Load metadata features
    with h5py.File(metadata_file, 'r') as h5f:
        metadata_feature_names = [key for key in h5f.keys() if key != 'filenames']
        metadata_features = []
        for feature_name in metadata_feature_names:
            metadata_features.append(h5f[feature_name][:])
        metadata_features = np.column_stack(metadata_features)
    
    # Combine features
    combined_features = np.concatenate([dinov2_features, metadata_features], axis=1)
    
    print(f"Combined features shape: DINOv2({dinov2_features.shape[1]}) + Metadata({metadata_features.shape[1]}) = {combined_features.shape[1]}")
    
    return combined_features, labels, filenames

def train_xgboost_classifier(train_features_file, val_features_file, train_metadata_file, val_metadata_file, checkpoint_dir, use_metadata=True):
    """Train XGBoost classifier on pre-computed features."""
    if use_metadata:
        print("=== Training XGBoost Classifier with Combined Features ===")
        # Load combined training data
        X_train, y_train, _ = load_combined_features_and_labels(train_features_file, train_metadata_file)
        X_val, y_val, _ = load_combined_features_and_labels(val_features_file, val_metadata_file)
    else:
        print("=== Training XGBoost Classifier with DinoV2 Features Only ===")
        # Load DinoV2 training data
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
    if use_metadata:
        print(f"Combined feature dimension: {X_train.shape[1]}")
    else:
        print(f"DinoV2 feature dimension: {X_train.shape[1]}")
    
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

def evaluate_xgboost_on_test_set(test_features_file, test_metadata_file, checkpoint_dir, session_name, use_metadata=True):
    """Evaluate XGBoost classifier on test set."""
    if use_metadata:
        print("=== Evaluating XGBoost on Test Set with Combined Features ===")
        # Load combined test data
        X_test, _, filenames = load_combined_features_and_labels(test_features_file, test_metadata_file)
    else:
        print("=== Evaluating XGBoost on Test Set with DinoV2 Features Only ===")
        # Load DinoV2 test data only
        X_test, _, filenames = load_features_and_labels(test_features_file)
    
    # Load the trained model
    model_path = os.path.join(checkpoint_dir, "best_xgboost_model.pkl")
    xgb_model = joblib.load(model_path)
    
    # Make predictions
    test_pred = xgb_model.predict(X_test)
    
    # Save results
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(zip(filenames, test_pred))  # Write filenames and predictions
    
    print(f"XGBoost results saved to {output_csv_path}")

def train_transformer_classifier(train_features_file, val_features_file, train_metadata_file, val_metadata_file, checkpoint_dir, num_classes, use_metadata=True):
    """Train Transformer classifier on pre-computed features."""
    if use_metadata:
        print("=== Training Transformer Classifier with Combined Features ===")
        # Initialize DataLoaders with combined features
        train_dataset = FungiFeatureDataset(train_features_file, train_metadata_file)
        valid_dataset = FungiFeatureDataset(val_features_file, val_metadata_file)
        feature_dim = train_dataset.total_feature_dim
    else:
        print("=== Training Transformer Classifier with DinoV2 Features Only ===")
        # Initialize DataLoaders with DinoV2 features only
        train_dataset = FungiFeatureDataset(train_features_file, metadata_file=None)
        valid_dataset = FungiFeatureDataset(val_features_file, metadata_file=None)
        feature_dim = train_dataset.total_feature_dim
        print(f"DinoV2 feature dimension: {feature_dim}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Transformer model with appropriate feature dimension
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
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True, eps=1e-8)
    criterion = SeesawLoss(num_classes=num_classes, p=0.8, q=2.0).to(device) # nn.CrossEntropyLoss()
    
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

def evaluate_transformer_on_test_set(test_features_file, test_metadata_file, checkpoint_dir, session_name, num_classes, use_metadata=True):
    """Evaluate Transformer classifier on test set."""
    if use_metadata:
        print("=== Evaluating Transformer on Test Set with Combined Features ===")
        # Load combined test data
        test_dataset = FungiFeatureDataset(test_features_file, test_metadata_file)
        feature_dim = test_dataset.total_feature_dim
    else:
        print("=== Evaluating Transformer on Test Set with DinoV2 Features Only ===")
        # Load DinoV2 test data only
        test_dataset = FungiFeatureDataset(test_features_file, metadata_file=None)
        feature_dim = test_dataset.total_feature_dim
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the Transformer model with appropriate feature dimension
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

def train_linear_classifier(train_features_file, val_features_file, train_metadata_file, val_metadata_file, checkpoint_dir, num_classes, use_metadata=True):
    """Train Linear classifier on pre-computed features."""
    if use_metadata:
        print("=== Training Linear Classifier with Combined Features ===")
        # Initialize DataLoaders with combined features (DINOv2 + metadata)
        train_dataset = FungiFeatureDataset(train_features_file, train_metadata_file)
        valid_dataset = FungiFeatureDataset(val_features_file, val_metadata_file)
        feature_dim = train_dataset.total_feature_dim  # Use combined feature dimension
    else:
        print("=== Training Linear Classifier with DinoV2 Features Only ===")
        # Initialize DataLoaders with DinoV2 features only
        train_dataset = FungiFeatureDataset(train_features_file, metadata_file=None)
        valid_dataset = FungiFeatureDataset(val_features_file, metadata_file=None)
        feature_dim = train_dataset.total_feature_dim
        print(f"DinoV2 feature dimension: {feature_dim}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Linear classifier model with appropriate feature dimension
    model = LinearClassifier(feature_dim, num_classes)
    model.to(device)

    # Define Optimization, Scheduler, and Criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)
    criterion = SeesawLoss(num_classes=num_classes, p=0.8, q=2.0).to(device) # nn.CrossEntropyLoss()

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Initialize CSV logger
    csv_file_path = os.path.join(checkpoint_dir, 'linear_train.csv')
    initialize_csv_logger(csv_file_path)

    # Training Loop
    max_epochs = 100
    epoch_pbar = tqdm.tqdm(range(max_epochs), desc="Linear Training Progress")
    
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
            print(f"Epoch {epoch + 1}: Best linear accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best linear loss updated to {best_loss:.4f}")
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
    print("=== Linear Training Complete ===")

def evaluate_linear_on_test_set(test_features_file, test_metadata_file, checkpoint_dir, session_name, num_classes, use_metadata=True):
    """Evaluate Linear classifier on test set."""
    if use_metadata:
        print("=== Evaluating Linear Classifier on Test Set with Combined Features ===")
        # Use combined features (DINOv2 + metadata) for evaluation
        test_dataset = FungiFeatureDataset(test_features_file, test_metadata_file)
        feature_dim = test_dataset.total_feature_dim  # Use combined feature dimension
    else:
        print("=== Evaluating Linear Classifier on Test Set with DinoV2 Features Only ===")
        # Use DinoV2 features only for evaluation
        test_dataset = FungiFeatureDataset(test_features_file, metadata_file=None)
        feature_dim = test_dataset.total_feature_dim
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load the Linear classifier model with appropriate feature dimension
    model = LinearClassifier(feature_dim, num_classes)
    
    # Load the best model
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for features, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating Linear"):
            features = features.to(device)
            outputs = model(features).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    
    print(f"Linear results saved to {output_csv_path}")

def train_fungi_network(data_file, image_path, checkpoint_dir, dinov2_model_name='dinov2_vitg14', classifier_type='linear', use_metadata=False):
    """
    Train the DINOv2 + classifier and save the best models based on validation accuracy and loss.
    Supports linear, XGBoost, and Transformer classifiers with optional metadata features.
    
    Args:
        use_metadata: If True, use both DinoV2 and metadata features. If False, use only DinoV2 features.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Load metadata
    df = load_and_preprocess_metadata(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    # Feature extraction step - extract and save DINOv2 features
    features_dir = os.path.join(ROOT_DIR, 'data', 'features', dinov2_model_name)
    ensure_folder(features_dir)
    
    train_features_file = os.path.join(features_dir, 'train_features.h5')
    val_features_file = os.path.join(features_dir, 'val_features.h5')

    metadata_dir = os.path.join(ROOT_DIR, 'data', 'metadata')
    ensure_folder(metadata_dir)

    train_metadata_file = os.path.join(metadata_dir, 'train_metadata.h5')
    val_metadata_file = os.path.join(metadata_dir, 'val_metadata.h5')

    print("=== Feature Extraction Phase ===")
    extract_dinov2_features(train_df, image_path, train_features_file, dinov2_model_name)
    extract_dinov2_features(val_df, image_path, val_features_file, dinov2_model_name)
    
    if use_metadata:
        extract_metadata_features(train_df, train_metadata_file)
        extract_metadata_features(val_df, val_metadata_file)
        print("=== Feature Extraction Complete (DinoV2 + Metadata) ===")
    else:
        print("=== Feature Extraction Complete (DinoV2 only) ===")

    # Branch based on classifier type
    if classifier_type == 'xgboost':
        # Train XGBoost classifier
        train_xgboost_classifier(train_features_file, val_features_file, train_metadata_file, val_metadata_file, checkpoint_dir, use_metadata)
        return
    elif classifier_type == 'transformer':
        # Train Transformer classifier
        num_classes = len(train_df['taxonID_index'].unique())
        train_transformer_classifier(train_features_file, val_features_file, train_metadata_file, val_metadata_file, checkpoint_dir, num_classes, use_metadata)
        return
    elif classifier_type == 'linear':
        # Train Linear classifier
        num_classes = len(train_df['taxonID_index'].unique())
        train_linear_classifier(train_features_file, val_features_file, train_metadata_file, val_metadata_file, checkpoint_dir, num_classes, use_metadata)
        return
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name, dinov2_model_name='dinov2_vitg14', classifier_type='linear', use_metadata=False):
    """
    Evaluate classifier on the test set and save predictions to a CSV file.
    Supports both linear and XGBoost classifiers.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    df = load_and_preprocess_metadata(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    
    # Extract features for test set
    features_dir = os.path.join(ROOT_DIR, 'data', 'features', dinov2_model_name)
    ensure_folder(features_dir)
    test_features_file = os.path.join(features_dir, 'test_features.h5')

    metadata_dir = os.path.join(ROOT_DIR, 'data', 'metadata')
    ensure_folder(metadata_dir)
    test_metadata_file = os.path.join(metadata_dir, 'test_metadata.h5')
    
    print("=== Test Feature Extraction ===")
    extract_dinov2_features(test_df, image_path, test_features_file, dinov2_model_name)
    
    if use_metadata:
        extract_metadata_features(test_df, test_metadata_file)
        print("=== Test Feature Extraction Complete (DinoV2 + Metadata) ===")
    else:
        print("=== Test Feature Extraction Complete (DinoV2 only) ===")
    
    # Branch based on classifier type and feature mode
    if classifier_type == 'xgboost':
        # Evaluate XGBoost classifier
        evaluate_xgboost_on_test_set(test_features_file, test_metadata_file, checkpoint_dir, session_name, use_metadata)
        return
    elif classifier_type == 'transformer':
        # Evaluate Transformer classifier
        evaluate_transformer_on_test_set(test_features_file, test_metadata_file, checkpoint_dir, session_name, num_classes=183, use_metadata=use_metadata)
        return
    elif classifier_type == 'linear':
        # Evaluate Linear classifier
        evaluate_linear_on_test_set(test_features_file, test_metadata_file, checkpoint_dir, session_name, num_classes=183, use_metadata=use_metadata)
        return
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

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
    parser.add_argument('--use_metadata', action='store_true', default=False,
                        help='Use metadata features in addition to DinoV2 features (default: False, DinoV2 only)')
    
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
        if args.use_metadata:
            session = f"DinoV2{model_size}{classifier_name}_MM"
        else:
            session = f"DinoV2{model_size}{classifier_name}"
    else:
        session = args.session

    # Folder for results of this experiment based on session name:
    checkpoint_dir = ROOT_DIR / 'results' / session
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training with DINOv2 + {args.classifier.capitalize()} classifier...")
    print(f"DINOv2 Model: {args.dinov2_model}")
    print(f"Classifier: {args.classifier}")
    print(f"Use Metadata: {args.use_metadata}")
    print(f"Session: {session}")
    print(f"Data file: {data_file}")
    print(f"Image path: {image_path}")
    print(f"Results will be saved to: {checkpoint_dir}")
    
    train_fungi_network(data_file, image_path, checkpoint_dir, args.dinov2_model, args.classifier, args.use_metadata)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session, args.dinov2_model, args.classifier, args.use_metadata)
