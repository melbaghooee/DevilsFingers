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
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, INFO, FileHandler, Formatter, StreamHandler
import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter
import logging
import math


# -------------------- Utilities --------------------

def ensure_folder(folder):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def initialize_csv_logger(file_path):
    header = ["epoch", "time_sec", "val_loss", "val_accuracy", "train_loss", "train_accuracy", "lr"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)


def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy, lr):
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, f"{epoch_time:.2f}", f"{val_loss:.6f}", f"{val_accuracy:.6f}",
                         f"{train_loss:.6f}", f"{train_accuracy:.6f}", f"{lr:.8f}"])


def setup_logging(log_dir, name="train"):
    """Console + file logger with timestamps."""
    ensure_folder(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(INFO)
    logger.handlers.clear()

    fmt = Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmt)

    fh = FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    NOTE: Albumentations expects (height, width) order.
    """
    height, width = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop((height, width), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(height, width),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")


# -------------------- Dataset --------------------

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].iloc[idx]
        label = self.df['taxonID_index'].iloc[idx]
        label = -1 if pd.isnull(label) else int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = img.convert('RGB')
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, file_path


# --------------- FUSION -------------------

def prepare_metadata_encoders(train_df):
    # normalize categories
    def _norm_cat(s):
        return (s.astype("string").fillna("<UNK>").replace({"<NA>": "<UNK>"}))

    train_df = train_df.copy()
    train_df["Habitat"] = _norm_cat(train_df["Habitat"])
    train_df["Substrate"] = _norm_cat(train_df["Substrate"])

    # month from eventDate
    train_df["eventDate"] = pd.to_datetime(train_df["eventDate"], errors="coerce")
    train_df["month"] = train_df["eventDate"].dt.month

    # categorical vocab (include UNK at index 0)
    hab_cats = ["<UNK>"] + sorted([c for c in train_df["Habitat"].unique() if c != "<UNK>"])
    sub_cats = ["<UNK>"] + sorted([c for c in train_df["Substrate"].unique() if c != "<UNK>"])
    hab2i = {c: i for i, c in enumerate(hab_cats)}
    sub2i = {c: i for i, c in enumerate(sub_cats)}

    # numeric stats
    lat_mean = float(train_df["Latitude"].mean(skipna=True)) if train_df["Latitude"].notna().any() else 0.0
    lon_mean = float(train_df["Longitude"].mean(skipna=True)) if train_df["Longitude"].notna().any() else 0.0
    lat_std = float(train_df["Latitude"].std(skipna=True)) if train_df["Latitude"].notna().any() else 1.0
    lon_std = float(train_df["Longitude"].std(skipna=True)) if train_df["Longitude"].notna().any() else 1.0
    lat_std = lat_std if lat_std > 0 else 1.0
    lon_std = lon_std if lon_std > 0 else 1.0

    meta_cfg = {
        "hab_cats": hab_cats,
        "sub_cats": sub_cats,
        "lat_mean": lat_mean, "lat_std": lat_std,
        "lon_mean": lon_mean, "lon_std": lon_std
    }
    # lat, lon, lat_missing, lon_missing, month_sin, month_cos, month_missing
    meta_dim = len(hab_cats) + len(sub_cats) + 7
    return meta_cfg, hab2i, sub2i, meta_dim


def encode_meta_row(row, cfg, hab2i, sub2i):
    # categorical with UNK
    h = str(row.get("Habitat", "<UNK>") or "<UNK>")
    s = str(row.get("Substrate", "<UNK>") or "<UNK>")
    h_idx = hab2i.get(h, 0);
    s_idx = sub2i.get(s, 0)
    hab_oh = np.zeros(len(cfg["hab_cats"]), dtype=np.float32);
    hab_oh[h_idx] = 1.0
    sub_oh = np.zeros(len(cfg["sub_cats"]), dtype=np.float32);
    sub_oh[s_idx] = 1.0

    # lat/lon + missing flags
    lat = row.get("Latitude", np.nan);
    lon = row.get("Longitude", np.nan)
    lat_missing = float(pd.isna(lat));
    lon_missing = float(pd.isna(lon))
    if pd.isna(lat): lat = cfg["lat_mean"]
    if pd.isna(lon): lon = cfg["lon_mean"]
    lat = (float(lat) - cfg["lat_mean"]) / (cfg["lat_std"] or 1.0)
    lon = (float(lon) - cfg["lon_mean"]) / (cfg["lon_std"] or 1.0)

    # month → sin/cos + missing
    m = row.get("month", np.nan)
    if pd.isna(m):
        m_sin, m_cos, m_missing = 0.0, 0.0, 1.0
    else:
        ang = 2 * math.pi * (int(m) - 1) / 12.0
        m_sin, m_cos, m_missing = math.sin(ang), math.cos(ang), 0.0

    meta = np.concatenate([hab_oh, sub_oh, np.array(
        [lat, lon, lat_missing, lon_missing, m_sin, m_cos, m_missing], dtype=np.float32)])
    return torch.from_numpy(meta)


# ---- dataset: always returns meta tensor (empty if not used) ----
class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None,
                 use_metadata=False, meta_cfg=None, hab2i=None, sub2i=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path = path
        self.use_metadata = use_metadata
        self.meta_cfg, self.hab2i, self.sub2i = meta_cfg, hab2i, sub2i

        if self.use_metadata:
            # prepare month for all rows (val/test included)
            self.df["eventDate"] = pd.to_datetime(self.df["eventDate"], errors="coerce")
            self.df["month"] = self.df["eventDate"].dt.month

            # normalize cats for all rows
            def _norm_cat(s):
                return (s.astype("string").fillna("<UNK>").replace({"<NA>": "<UNK>"}))

            self.df["Habitat"] = _norm_cat(self.df["Habitat"])
            self.df["Substrate"] = _norm_cat(self.df["Substrate"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['filename_index']
        label = row['taxonID_index']
        label = -1 if pd.isna(label) else int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = np.array(img.convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']

        if self.use_metadata:
            meta = encode_meta_row(row, self.meta_cfg, self.hab2i, self.sub2i).float()
        else:
            meta = torch.empty(0, dtype=torch.float32)  # keeps interfaces uniform

        return image, label, file_path, meta


# ---- model: works with/without metadata ----
class FusionEfficientNet(nn.Module):
    def __init__(self, num_classes, meta_dim=0):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
            )
            self.head = nn.Linear(in_feats + 128, num_classes)
        else:
            self.head = nn.Linear(in_feats, num_classes)

    def forward(self, x_img, x_meta=None):
        f_img = self.backbone(x_img)  # [B, 1280]
        if self.meta_dim > 0 and x_meta is not None and x_meta.numel() > 0:
            f_meta = self.meta_mlp(x_meta)  # [B, 128]
            f = torch.cat([f_img, f_meta], dim=1)
        else:
            f = f_img
        return self.head(f)


# -------------------- Training --------------------
def train_fungi_network(data_file, image_path, checkpoint_dir, use_metadata=True):
    """
    Same training loop; toggle metadata by use_metadata flag.
    """
    ensure_folder(checkpoint_dir)
    logger = setup_logging(checkpoint_dir, "train")
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    seed_torch(777)

    # Load & split
    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')].copy()
    train_df = train_df[train_df['taxonID_index'].notna()].copy()
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['taxonID_index'])

    logger.info(f"Training samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # Encoders only if using metadata (fit on TRAIN)
    meta_cfg = hab2i = sub2i = None
    meta_dim = 0
    if use_metadata:
        meta_cfg, hab2i, sub2i, meta_dim = prepare_metadata_encoders(train_df)

    # Datasets / loaders (note: dataset always returns a meta tensor)
    train_dataset = FungiDataset(
        train_df, image_path, transform=get_transforms('train'),
        use_metadata=use_metadata, meta_cfg=meta_cfg, hab2i=hab2i, sub2i=sub2i)
    valid_dataset = FungiDataset(
        val_df, image_path, transform=get_transforms('valid'),
        use_metadata=use_metadata, meta_cfg=meta_cfg, hab2i=hab2i, sub2i=sub2i)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"CUDA available: {torch.cuda.is_available()} | Device: {device}")

    classes = sorted(train_df['taxonID_index'].astype(int).unique().tolist())
    num_classes = len(classes)
    logger.info(f"Number of classes: {num_classes} | use_metadata={use_metadata} | meta_dim={meta_dim}")

    model = FusionEfficientNet(num_classes=num_classes, meta_dim=meta_dim).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, eps=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    patience, patience_counter = 10, 0
    best_loss, best_accuracy = float('inf'), 0.0
    max_epochs = 100
    logger.info("Starting training...")

    try:
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss = running_correct = running_seen = 0

            current_lr = optimizer.param_groups[0]['lr']
            train_bar = tqdm.tqdm(train_loader, total=len(train_loader),
                                  desc=f"Epoch {epoch:03d}/{max_epochs} [train]", leave=False, ncols=0)

            for images, labels, _, meta in train_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True) if use_metadata else None

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images, meta)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_sz = labels.size(0)
                running_loss += loss.item()
                running_correct += (outputs.argmax(1) == labels).sum().item()
                running_seen += batch_sz

                avg_loss = running_loss / max(1, len(train_bar))
                train_acc = running_correct / max(1, running_seen)
                train_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}", lr=f"{current_lr:.2e}")

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = running_correct / max(1, running_seen)

            # ---- Validation ----
            model.eval()
            val_running_loss = val_running_correct = val_running_seen = 0
            val_bar = tqdm.tqdm(valid_loader, total=len(valid_loader),
                                desc=f"Epoch {epoch:03d}/{max_epochs} [valid]", leave=False, ncols=0)
            with torch.no_grad():
                for images, labels, _, meta in val_bar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    meta = meta.to(device, non_blocking=True) if use_metadata else None

                    outputs = model(images, meta)
                    loss = criterion(outputs, labels)

                    batch_sz = labels.size(0)
                    val_running_loss += loss.item()
                    val_running_correct += (outputs.argmax(1) == labels).sum().item()
                    val_running_seen += batch_sz

                    avg_val = val_running_loss / max(1, len(val_bar))
                    val_acc = val_running_correct / max(1, val_running_seen)
                    val_bar.set_postfix(loss=f"{avg_val:.4f}", acc=f"{val_acc:.4f}")

            avg_val_loss = val_running_loss / len(valid_loader)
            val_accuracy = val_running_correct / max(1, val_running_seen)

            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f}, train_acc={train_accuracy:.4f} | "
                        f"val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.4f} | lr={new_lr:.2e} | time={epoch_time:.2f}s")

            # CSV (adjust if your signature differs)
            log_epoch_to_csv(csv_file_path, epoch, epoch_time, avg_train_loss, train_accuracy, avg_val_loss,
                             val_accuracy, current_lr)

            # Checkpoints
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
                logger.info(f"↑ New best accuracy: {best_accuracy:.4f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
                logger.info(f"↓ New best loss: {best_loss:.4f}; patience reset.")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No val loss improvement. Patience {patience_counter}/{patience}.")
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {patience} epochs without val loss improvement.")
                    break

        logger.info("Training finished.")
        # Save encoders if used (so eval can reuse them)
        if use_metadata:
            torch.save({"meta_cfg": meta_cfg, "hab2i": hab2i, "sub2i": sub2i, "meta_dim": meta_dim},
                       os.path.join(checkpoint_dir, "metadata_encoders.pt"))

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving last checkpoint...")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "interrupted.pth"))


# -------------------- Evaluation --------------------

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    ensure_folder(checkpoint_dir)
    logger = setup_logging(checkpoint_dir, "test")

    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')].copy()
    logger.info(f"Test samples: {len(test_df)}")

    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading model from {best_trained_model}")
    model = models.efficientnet_b0(pretrained=True)
    # ⚠️ NOTE: Keep num classes consistent with training; adjust if needed.
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 183)  # <-- keep or compute dynamically to match training
    )
    model.load_state_dict(torch.load(best_trained_model, map_location=device))
    model.to(device)

    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating", ncols=0):
            images = images.to(device, non_blocking=True)
            outputs = model(images).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])
        writer.writerows(results)

    logger.info(f"Predictions saved to {output_csv_path}")


# -------------------- Main --------------------

if __name__ == "__main__":
    image_path = 'data/FungiImages/'
    data_file = 'data/metadata.csv'
    session = "FusionEfficientNet"
    checkpoint_dir = os.path.join(f"results/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir, True)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)
