#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

# --------------------- Utils ---------------------

def seed_all(seed: int = 777):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def norm_cat(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<UNK>").replace({"<NA>": "<UNK>"})

def month_col(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["eventDate"], errors="coerce")
    return dt.dt.month.fillna(0).astype(int)  # 0 = missing

def geo_bin(lat, lon, lat_deg=10.0, lon_deg=10.0) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "G:MISSING"
    lat_b = int(math.floor((float(lat) + 90.0) / lat_deg))
    lon_b = int(math.floor((float(lon) + 180.0) / lon_deg))
    return f"G:{lat_b}_{lon_b}"

# --------------------- Bayesian prior (late fusion) ---------------------

class BayesMetaPrior:
    """
    Naive-Bayes prior: sum_k log p(m_k | y).
    Trained on TRAIN split; applied as a bias to logits at eval.
    """
    def __init__(self, class_ids: List[int], alpha: float = 1.0, lat_deg: float = 10.0, lon_deg: float = 10.0):
        self.class_ids = list(map(int, class_ids))
        self.C = len(self.class_ids)
        self.y2i = {y:i for i,y in enumerate(self.class_ids)}
        self.alpha = float(alpha)
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.tables = {}         # name -> {domain, val2i, logP: [C,V]}
        self.filename_maps = {}  # name -> {filename -> value_index}

    def _fit_categorical(self, df: pd.DataFrame, col: str, domain: List) -> None:
        vals = list(domain)
        val2i = {v:i for i,v in enumerate(vals)}
        V = len(vals)
        rows = df[["taxonID_index", col, "filename_index"]].dropna(subset=["taxonID_index"]).copy()
        rows["taxonID_index"] = rows["taxonID_index"].astype(int)
        rows = rows[rows["taxonID_index"].isin(self.class_ids)]
        if rows.empty:
            logP = torch.full((self.C, V), -math.log(V), dtype=torch.float32)
        else:
            # counts
            counts = np.zeros((self.C, V), dtype=np.float64)
            for y, v in zip(rows["taxonID_index"].values, rows[col].values):
                if v not in val2i: v = vals[0]  # UNK or 0
                counts[self.y2i[int(y)], val2i[v]] += 1.0
            n_y = counts.sum(axis=1, keepdims=True)
            counts = counts + self.alpha
            denom = n_y + self.alpha * V
            logP = torch.from_numpy(np.log(counts / denom)).to(torch.float32)
        self.tables[col] = {"domain": vals, "val2i": val2i, "logP": logP}

    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["Habitat"] = norm_cat(df["Habitat"])
        df["Substrate"] = norm_cat(df["Substrate"])
        df["Month"] = month_col(df)
        df["GeoTok"] = [geo_bin(la, lo, self.lat_deg, self.lon_deg) for la, lo in zip(df["Latitude"], df["Longitude"])]

        hab_vals = ["<UNK>"] + sorted([v for v in df["Habitat"].dropna().unique() if v != "<UNK>"])
        sub_vals = ["<UNK>"] + sorted([v for v in df["Substrate"].dropna().unique() if v != "<UNK>"])
        mon_vals = list(range(0, 13))
        geo_vals = ["G:MISSING"] + sorted([v for v in df["GeoTok"].unique() if v != "G:MISSING"])

        self._fit_categorical(df, "Habitat", hab_vals)
        self._fit_categorical(df, "Substrate", sub_vals)
        self._fit_categorical(df, "Month", mon_vals)
        self._fit_categorical(df, "GeoTok", geo_vals)

        # build filename -> value-index maps
        def build_map(col: str, cast=None):
            tab = self.tables[col]; v2i = tab["val2i"]
            if col == "Habitat":
                series = norm_cat(train_df["Habitat"])
            elif col == "Substrate":
                series = norm_cat(train_df["Substrate"])
            elif col == "Month":
                series = month_col(train_df)
            elif col == "GeoTok":
                series = [geo_bin(la, lo, self.lat_deg, self.lon_deg)
                          for la, lo in zip(train_df["Latitude"], train_df["Longitude"])]
            else:
                series = train_df[col]
            mp = {}
            for fn, val in zip(train_df["filename_index"], series):
                if cast is not None:
                    val = cast(val)
                if val not in v2i:
                    val = list(v2i.keys())[0]
                mp[fn] = v2i[val]
            return mp

        self.filename_maps["Habitat"] = build_map("Habitat")
        self.filename_maps["Substrate"] = build_map("Substrate")
        self.filename_maps["Month"] = build_map("Month", cast=int)
        self.filename_maps["GeoTok"] = build_map("GeoTok")
        return self

    @torch.no_grad()
    def batch_bias(self, filenames: List[str], device=None) -> torch.Tensor:
        B = len(filenames)
        bias = torch.zeros((B, self.C), dtype=torch.float32)
        for name, tab in self.tables.items():
            v2i = tab["val2i"]; logP = tab["logP"]      # [C,V]
            fmap = self.filename_maps.get(name, {})
            idxs = torch.tensor([fmap.get(fn, 0) for fn in filenames], dtype=torch.long)  # [B]
            b = logP[:, idxs].T  # [B,C]
            bias += b
        if device is not None:
            bias = bias.to(device)
        return bias

# --------------------- Datasets ---------------------

class EvalDataset(Dataset):
    """Returns (image, label, filename)."""
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        fn = row["filename_index"]
        y  = int(row["taxonID_index"])
        with Image.open(os.path.join(self.root, fn)) as img:
            x = np.array(img.convert("RGB"))
        if self.transform:
            x = self.transform(image=x)["image"]
        return x, y, fn

# Use your original transforms (copy here if needed)
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
def get_transforms(data):
    width, height = 224, 224
    if data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    raise ValueError("Only 'valid' is used in this script.")

# --------------------- Models ---------------------

class FusionEfficientNet(nn.Module):
    """If meta_dim==0, behaves like vanilla EfficientNet-B0 head."""
    def __init__(self, num_classes: int, meta_dim: int = 0):
        super().__init__()
        base = models.efficientnet_b0(pretrained=True)
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 128), nn.ReLU(inplace=True),
                nn.BatchNorm1d(128), nn.Dropout(0.2)
            )
            self.head = nn.Linear(in_feats + 128, num_classes)
        else:
            self.head = nn.Linear(in_feats, num_classes)

    def forward(self, x_img, x_meta=None):
        f_img = self.backbone(x_img)
        if self.meta_dim > 0 and x_meta is not None:
            f_meta = self.meta_mlp(x_meta)
            f = torch.cat([f_img, f_meta], dim=1)
        else:
            f = f_img
        return self.head(f)

# --------------------- Confusion utilities ---------------------

def plot_confusion(cm: np.ndarray, class_names: List[str], out_png: str, normalize: str = None):
    """
    normalize in {None, 'true', 'pred', 'all'}
    """
    if normalize == 'true':
        cm_plot = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        title = "Confusion Matrix (row-normalized)"
    elif normalize == 'pred':
        cm_plot = cm / cm.sum(axis=0, keepdims=True).clip(min=1)
        title = "Confusion Matrix (col-normalized)"
    elif normalize == 'all':
        cm_plot = cm / cm.sum()
        title = "Confusion Matrix (global-normalized)"
    else:
        cm_plot = cm.astype(float)
        title = "Confusion Matrix (counts)"

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_plot, aspect='auto', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # tick sparsification if many classes
    n = len(class_names)
    step = max(1, n // 30)
    ticks = list(range(0, n, step))
    plt.xticks(ticks, [class_names[i] for i in ticks], rotation=90)
    plt.yticks(ticks, [class_names[i] for i in ticks])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def top_confusions(cm: np.ndarray, class_names: List[str], k: int = 20) -> pd.DataFrame:
    """Return top-k off-diagonal confusions by count."""
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i, j] > 0:
                pairs.append((int(cm2[i, j]), i, j))
    pairs.sort(reverse=True)  # by count
    rows = []
    for cnt, i, j in pairs[:k]:
        total_true = cm[i, :].sum()
        frac = cnt / total_true if total_true > 0 else 0.0
        rows.append({
            "true_id": class_names[i],
            "pred_id": class_names[j],
            "count": int(cnt),
            "frac_of_true": round(frac, 4)
        })
    return pd.DataFrame(rows)

# --------------------- Main eval ---------------------

def main():
    ap = argparse.ArgumentParser(description="Confusion matrix evaluation on validation split.")
    ap.add_argument("--data-file", default="data/metadata.csv")
    ap.add_argument("--image-path", default="data/FungiImages/")
    ap.add_argument("--checkpoint-dir", default="results/FusionEfficientNet/")
    ap.add_argument("--use-metadata", action="store_true", help="If your trained model used metadata fusion.")
    ap.add_argument("--use-meta-prior", action="store_true", help="Apply Bayesian prior correction at eval.")
    ap.add_argument("--lambda-prior", type=float, default=1.0, help="Weight for metadata prior bias.")
    ap.add_argument("--normalize", choices=[None, "true", "pred", "all"], default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    seed_all(777)
    ensure_dir(args.checkpoint_dir)

    # Load metadata CSV and build the SAME train/val split as training
    df = pd.read_csv(args.data_file)
    df_lab = df[df["filename_index"].str.startswith("fungi_train")].copy()
    df_lab = df_lab[df_lab["taxonID_index"].notna()].copy()
    # stratified split with random_state=42 (same as your training)
    train_df, val_df = train_test_split(
        df_lab, test_size=0.2, random_state=42, stratify=df_lab["taxonID_index"])

    classes = sorted(train_df["taxonID_index"].astype(int).unique().tolist())
    class2idx = {c:i for i,c in enumerate(classes)}
    class_names = [str(c) for c in classes]  # or map to species names if available

    # Build model (meta_dim==0 -> vanilla)
    meta_dim = 0  # keep 0; we are not reusing meta MLP here unless you want to
    model = FusionEfficientNet(num_classes=len(classes), meta_dim=meta_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint
    ckpt_acc = os.path.join(args.checkpoint_dir, "best_accuracy.pth")
    ckpt_loss = os.path.join(args.checkpoint_dir, "best_loss.pth")
    ckpt = ckpt_acc if os.path.exists(ckpt_acc) else ckpt_loss
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Data loader over VALIDATION split
    ds = EvalDataset(val_df, args.image_path, transform=get_transforms("valid"))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optional Bayesian prior correction (late fusion)
    meta_prior = None
    if args.use_meta_prior:
        meta_prior = BayesMetaPrior(class_ids=classes, alpha=1.0, lat_deg=10.0, lon_deg=10.0).fit(train_df)

    # Collect predictions
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels, filenames in dl:
            images = images.to(device, non_blocking=True)
            logits = model(images)  # [B,C]
            if meta_prior is not None:
                bias = meta_prior.batch_bias(list(filenames), device=device)  # [B,C]
                logits = logits + args.lambda_prior * bias
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend([classes[p] for p in preds])
            y_true.extend(labels.numpy().tolist())

    # Convert labels to consistent indices
    y_true_idx = np.array([class2idx[int(y)] for y in y_true])
    y_pred_idx = np.array([class2idx[int(y)] for y in y_pred])

    # Confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(classes))))
    report = classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=4, zero_division=0)

    # Save outputs
    out_dir = Path(args.checkpoint_dir)
    np.savetxt(out_dir / "confusion_matrix_val_counts.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, str(out_dir / "confusion_matrix_val.png"), normalize=args.normalize)

    # Normalized (row) version as CSV (handy to inspect)
    cm_row = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(cm_row, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix_val_row_normalized.csv")

    # Top confusions
    df_top = top_confusions(cm, class_names, k=30)
    df_top.to_csv(out_dir / "top_confusions_val.csv", index=False)

    # Print summary to console
    print("\n=== Classification report (validation) ===")
    print(report)
    print(f"\nSaved confusion matrix PNG to: {out_dir / 'confusion_matrix_val.png'}")
    print(f"Saved counts CSV to:          {out_dir / 'confusion_matrix_val_counts.csv'}")
    print(f"Saved row-normalized CSV to:  {out_dir / 'confusion_matrix_val_row_normalized.csv'}")
    print(f"Saved top confusions CSV to:  {out_dir / 'top_confusions_val.csv'}")

if __name__ == "__main__":
    main()
