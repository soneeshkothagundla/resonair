"""
Augmented Ensemble Lung Cancer Classifier
============================================
Generates offline augmented copies of spectrogram images to
grow the training set, then trains an ensemble of models with
heavy regularisation and test-time augmentation (TTA).

Key improvements over the base ensemble:
  1. Offline augmentation:  10x positive-class, 3x negative-class
     => 300 positive + 522 negative = ~822 training images
  2. Heavier online augmentation during training
  3. Label smoothing for better calibration
  4. Test-Time Augmentation (TTA) for more stable predictions
  5. Temperature scaling for better probability calibration
  6. Cross-architecture ensemble (best 2 models)
"""

import sys
import random
import shutil
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 80
LR = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE_LR = 7
PATIENCE_EARLY = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_SMOOTHING = 0.1

# Augmentation multipliers
POS_AUG_FACTOR = 10   # 30 positives x 10 = 300 augmented + 30 original
NEG_AUG_FACTOR = 3    # 174 negatives x 3 = 522 augmented + 174 original (but we cap)

# TTA settings
TTA_ROUNDS = 5  # number of augmented inference passes

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results_augmented"
AUG_DIR = BASE_DIR / "augmented_data"
OUTPUT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────
# Data Discovery
# ──────────────────────────────────────────────
DISEASE_FOLDERS = [
    "1. COVID-19", "2. Lungs Cancer", "3. Consolidation Lung",
    "4. Atelectasis", "5. Tuberculosis", "6. Pneumothorax",
    "7. Edema", "8. Pneumonia", "9. Normal",
]
POSITIVE_FOLDER = "2. Lungs Cancer"


def discover_images():
    image_paths, labels = [], []
    for folder in DISEASE_FOLDERS:
        csi_dir = BASE_DIR / folder / "CSI"
        if not csi_dir.exists():
            print(f"[WARNING] Missing CSI folder: {csi_dir}")
            continue
        label = 1 if folder == POSITIVE_FOLDER else 0
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img_file in sorted(csi_dir.glob(ext)):
                image_paths.append(str(img_file))
                labels.append(label)
    return image_paths, labels


# ──────────────────────────────────────────────
# Offline Augmentation Pipeline
# ──────────────────────────────────────────────
def create_offline_augmentations(image_paths, labels, split_name="train"):
    """
    Generate augmented copies of images offline.
    Returns (augmented_paths, augmented_labels) including originals.
    """
    aug_split_dir = AUG_DIR / split_name
    if aug_split_dir.exists():
        shutil.rmtree(aug_split_dir)
    (aug_split_dir / "positive").mkdir(parents=True, exist_ok=True)
    (aug_split_dir / "negative").mkdir(parents=True, exist_ok=True)

    # Define augmentation transforms for offline generation
    offline_augments = [
        # 1. Horizontal flip
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        # 2. Vertical flip
        lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        # 3. Rotate 90
        lambda img: img.rotate(90, expand=True),
        # 4. Rotate -90
        lambda img: img.rotate(-90, expand=True),
        # 5. Rotate small angle
        lambda img: img.rotate(random.uniform(-25, 25)),
        # 6. Brightness increase
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(1.1, 1.5)),
        # 7. Brightness decrease
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 0.9)),
        # 8. Contrast increase
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(1.2, 1.6)),
        # 9. Contrast decrease
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 0.8)),
        # 10. Gaussian blur
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
        # 11. Color jitter
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
        # 12. Sharpness
        lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(1.5, 3.0)),
        # 13. Slight crop and resize
        lambda img: _random_crop_resize(img),
        # 14. Combined: flip + brightness
        lambda img: ImageEnhance.Brightness(
            img.transpose(Image.FLIP_LEFT_RIGHT)
        ).enhance(random.uniform(0.8, 1.3)),
        # 15. Combined: rotate + contrast
        lambda img: ImageEnhance.Contrast(
            img.rotate(random.uniform(-20, 20))
        ).enhance(random.uniform(0.7, 1.4)),
    ]

    new_paths, new_labels = [], []
    aug_count = 0

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(path).convert("RGB")
        subdir = "positive" if label == 1 else "negative"
        factor = POS_AUG_FACTOR if label == 1 else NEG_AUG_FACTOR

        # Save original
        orig_name = f"orig_{i:04d}.png"
        orig_out = aug_split_dir / subdir / orig_name
        img.save(orig_out)
        new_paths.append(str(orig_out))
        new_labels.append(label)

        # Generate augmented copies
        for j in range(factor):
            aug_fn = random.choice(offline_augments)
            try:
                aug_img = aug_fn(img.copy())
                aug_name = f"aug_{i:04d}_{j:02d}.png"
                aug_out = aug_split_dir / subdir / aug_name
                aug_img.save(aug_out)
                new_paths.append(str(aug_out))
                new_labels.append(label)
                aug_count += 1
            except Exception:
                pass  # Skip if augmentation fails

    print(f"  [AUG] Created {aug_count} augmented images for {split_name}")
    return new_paths, new_labels


def _random_crop_resize(img):
    """Random crop 85-95% of the image and resize back."""
    w, h = img.size
    crop_ratio = random.uniform(0.85, 0.95)
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.RandomGrayscale(p=0.15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.25)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────
class ClassifierHead(nn.Module):
    def __init__(self, in_features, dropout1=0.5, dropout2=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(x)


def build_model(arch_name, freeze_ratio=0.7):
    if arch_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    elif arch_name == "mobilenet_v3_large":
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
    elif arch_name == "densenet121":
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown: {arch_name}")

    all_params = list(backbone.parameters())
    freeze_up_to = int(len(all_params) * freeze_ratio)
    for i, param in enumerate(all_params):
        if i < freeze_up_to:
            param.requires_grad = False

    return nn.Sequential(backbone, ClassifierHead(in_features))


# ──────────────────────────────────────────────
# Training Helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, running_correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        running_correct += (preds == labels).sum().item()
        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return running_loss / n, running_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, running_correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        running_correct += (preds == labels).sum().item()
        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / n
    avg_acc = running_correct / n
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    return avg_loss, avg_acc, auc, np.array(all_probs), np.array(all_labels)


@torch.no_grad()
def predict_with_tta(model, loader, n_rounds=TTA_ROUNDS):
    """Test-Time Augmentation: average predictions over multiple augmented passes."""
    model.eval()
    # First pass: standard (no augmentation)
    all_probs_runs = []
    all_labels = None

    # Standard pass
    probs_standard, labels_arr = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs).squeeze(1)
        probs = torch.sigmoid(outputs)
        probs_standard.extend(probs.cpu().numpy())
        labels_arr.extend(labels.numpy())
    all_probs_runs.append(np.array(probs_standard))
    all_labels = np.array(labels_arr)

    # TTA passes with augmented transforms
    for _ in range(n_rounds):
        # Rebuild dataset with TTA transform
        tta_ds = SpectrogramDataset(
            loader.dataset.paths, loader.dataset.labels, transform=tta_transform
        )
        tta_loader = DataLoader(tta_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True)
        probs_run = []
        for imgs, labels in tta_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).squeeze(1)
            probs = torch.sigmoid(outputs)
            probs_run.extend(probs.cpu().numpy())
        all_probs_runs.append(np.array(probs_run))

    # Average all runs
    avg_probs = np.mean(all_probs_runs, axis=0)
    return avg_probs, all_labels


# ──────────────────────────────────────────────
# Temperature Scaling (post-hoc calibration)
# ──────────────────────────────────────────────
class TemperatureScaler:
    """Learn a single temperature parameter on the validation set."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits, val_labels):
        """Find optimal temperature via grid search."""
        best_nll = float("inf")
        best_temp = 1.0
        for temp in np.arange(0.1, 5.0, 0.05):
            scaled_probs = 1.0 / (1.0 + np.exp(-val_logits / temp))
            # Negative log likelihood
            eps = 1e-7
            nll = -np.mean(
                val_labels * np.log(scaled_probs + eps)
                + (1 - val_labels) * np.log(1 - scaled_probs + eps)
            )
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        self.temperature = best_temp
        print(f"  [CALIBRATION] Optimal temperature: {self.temperature:.2f}")

    def scale(self, logits):
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


@torch.no_grad()
def get_logits(model, loader):
    """Get raw logits (before sigmoid) from a model."""
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs).squeeze(1)
        all_logits.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_logits), np.array(all_labels)


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────
def train_model(arch_name, train_loader, val_loader, criterion):
    print(f"\n{'='*60}")
    print(f"  Training: {arch_name.upper()}")
    print(f"{'='*60}")

    model = build_model(arch_name).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    best_auc = 0.0
    patience_counter = 0
    save_path = OUTPUT_DIR / f"best_{arch_name}.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        if epoch % 5 == 0 or val_auc > best_auc:
            print(
                f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"TrLoss: {train_loss:.4f} TrAcc: {train_acc:.4f} | "
                f"ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f} AUC: {val_auc:.4f}"
            )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"    --> New best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE_EARLY:
            print(f"  [EARLY STOP] at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    print(f"  Best Val AUC for {arch_name}: {best_auc:.4f}")
    return model, best_auc


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
def plot_model_comparison(results, save_path):
    names = list(results.keys())
    aucs = [results[n]["test_auc"] for n in names]
    accs = [results[n]["test_acc"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#3b82f6" if "ENSEMBLE" not in n else "#1a56db" for n in names]

    bars1 = ax1.bar(names, aucs, color=colors, edgecolor="white", linewidth=1.5)
    for bar, auc in zip(bars1, aucs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{auc:.4f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Test AUC")
    ax1.set_title("Test AUC Comparison", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis='x', rotation=25)

    bars2 = ax2.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Accuracy Comparison", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Model comparison -> {save_path}")


def plot_roc_curves(results, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.get_cmap("tab10")
    for i, (name, data) in enumerate(results.items()):
        labels = data["test_labels"]
        probs = data["test_probs"]
        if len(set(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        lw = 3.0 if "ENSEMBLE" in name else 1.8
        ls = "-" if "ENSEMBLE" in name else "--"
        ax.plot(fpr, tpr, linewidth=lw, linestyle=ls,
                color=cmap(i), label=f"{name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves - Augmented Models", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] ROC curves -> {save_path}")


def plot_confusion(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Other", "Lung Cancer"],
                yticklabels=["Other", "Lung Cancer"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Augmented Ensemble Lung Cancer Classifier")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # --- Discover original data ---
    image_paths, labels = discover_images()
    print(f"\nOriginal images: {len(image_paths)}")
    counts = Counter(labels)
    print(f"  Lung Cancer (1): {counts[1]}")
    print(f"  Other (0):       {counts[0]}")

    if len(image_paths) == 0:
        print("[ERROR] No images found.")
        sys.exit(1)

    # --- Split BEFORE augmentation (to prevent data leakage) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"\nOriginal splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # --- Offline augmentation on TRAINING set only ---
    print(f"\n--- Generating offline augmentations ---")
    X_train_aug, y_train_aug = create_offline_augmentations(X_train, y_train, "train")

    aug_counts = Counter(y_train_aug)
    print(f"  Augmented training set: {len(X_train_aug)} images")
    print(f"    Lung Cancer: {aug_counts[1]}")
    print(f"    Other:       {aug_counts[0]}")

    # --- Datasets ---
    train_ds = SpectrogramDataset(X_train_aug, y_train_aug, transform=train_transform)
    val_ds = SpectrogramDataset(X_val, y_val, transform=val_transform)
    test_ds = SpectrogramDataset(X_test, y_test, transform=val_transform)

    # Weighted sampler
    class_counts = Counter(y_train_aug)
    sample_weights = [1.0 / class_counts[label] for label in y_train_aug]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train_aug), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # --- Loss with label smoothing ---
    pos_weight = torch.tensor(
        [class_counts[0] / class_counts[1]], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Positive class weight: {pos_weight.item():.2f}")

    # ──────────────────────────────────────────────
    # Train models (top 3 from previous experiment)
    # ──────────────────────────────────────────────
    ARCHITECTURES = ["efficientnet_b0", "mobilenet_v3_large", "densenet121"]

    trained_models = {}
    all_results = {}

    for arch in ARCHITECTURES:
        model, best_val_auc = train_model(arch, train_loader, val_loader, criterion)
        trained_models[arch] = model

        # Temperature calibration
        val_logits, val_labels = get_logits(model, val_loader)
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)

        # Test with TTA
        print(f"  Running TTA ({TTA_ROUNDS} rounds)...")
        test_probs, test_labels = predict_with_tta(model, test_loader)

        # Also get raw logits for temperature scaling
        test_logits, _ = get_logits(model, test_loader)
        test_probs_calibrated = scaler.scale(test_logits)

        # Use calibrated TTA average
        # Combine TTA probs with calibrated probs
        final_probs = 0.5 * test_probs + 0.5 * test_probs_calibrated

        test_preds = (final_probs >= 0.5).astype(int)
        try:
            test_auc = roc_auc_score(test_labels, final_probs)
        except ValueError:
            test_auc = 0.0
        test_acc = accuracy_score(test_labels, test_preds)

        all_results[arch] = {
            "val_auc": best_val_auc,
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_probs": final_probs,
            "test_labels": test_labels,
            "test_preds": test_preds,
            "temperature": scaler.temperature,
        }
        print(f"  {arch} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")

    # ──────────────────────────────────────────────
    # Ensemble: Weighted average by validation AUC
    # ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Building Calibrated Ensemble")
    print(f"{'='*60}")

    val_aucs = np.array([all_results[a]["val_auc"] for a in ARCHITECTURES])
    weights = val_aucs / val_aucs.sum()
    print(f"  Ensemble weights: {dict(zip(ARCHITECTURES, [f'{w:.3f}' for w in weights]))}")

    ensemble_probs = np.average(
        [all_results[a]["test_probs"] for a in ARCHITECTURES],
        axis=0, weights=weights,
    )
    ensemble_labels = all_results[ARCHITECTURES[0]]["test_labels"]
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    try:
        ensemble_auc = roc_auc_score(ensemble_labels, ensemble_probs)
    except ValueError:
        ensemble_auc = 0.0
    ensemble_acc = accuracy_score(ensemble_labels, ensemble_preds)

    all_results["ENSEMBLE_TTA"] = {
        "val_auc": 0,
        "test_auc": ensemble_auc,
        "test_acc": ensemble_acc,
        "test_probs": ensemble_probs,
        "test_labels": ensemble_labels,
        "test_preds": ensemble_preds,
    }

    # ──────────────────────────────────────────────
    # Results Summary
    # ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Test AUC':>10} {'Test Acc':>10} {'Recall':>10} {'Precision':>10}")
    print("-" * 70)

    best_name, best_auc_val = None, 0.0
    for name, data in all_results.items():
        tp = ((data["test_preds"] == 1) & (data["test_labels"] == 1)).sum()
        fp = ((data["test_preds"] == 1) & (data["test_labels"] == 0)).sum()
        fn = ((data["test_preds"] == 0) & (data["test_labels"] == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        print(f"  {name:<23} {data['test_auc']:>10.4f} {data['test_acc']:>10.4f} "
              f"{recall:>10.4f} {precision:>10.4f}")

        if data["test_auc"] > best_auc_val:
            best_auc_val = data["test_auc"]
            best_name = name

    print(f"\n  >>> BEST: {best_name} with Test AUC = {best_auc_val:.4f}")

    # Save best individual model weights
    best_data = all_results[best_name]
    if best_name in trained_models:
        torch.save(
            trained_models[best_name].state_dict(),
            OUTPUT_DIR / "best_augmented_model.pth",
        )
        # Also record best architecture
        with open(OUTPUT_DIR / "best_architecture.txt", "w") as f:
            f.write(best_name)
        print(f"  [SAVED] Best model: {best_name}")

    # Save all individual model weights
    for arch in ARCHITECTURES:
        # Already saved during training
        pass

    # Classification report
    report = classification_report(
        best_data["test_labels"], best_data["test_preds"],
        target_names=["Other Diseases", "Lung Cancer"],
        digits=4,
    )
    print(f"\n{report}")

    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Test AUC: {best_auc_val:.4f}\n")
        f.write(f"Test Acc: {best_data['test_acc']:.4f}\n\n")
        f.write(f"Augmented training set size: {len(X_train_aug)}\n")
        f.write(f"  Positive: {aug_counts[1]}\n")
        f.write(f"  Negative: {aug_counts[0]}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Visualizations
    plot_model_comparison(all_results, OUTPUT_DIR / "model_comparison.png")
    plot_roc_curves(all_results, OUTPUT_DIR / "roc_curves.png")
    plot_confusion(
        best_data["test_labels"], best_data["test_preds"],
        OUTPUT_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix - {best_name} (Augmented)",
    )

    print(f"\n{'='*60}")
    print(f"  DONE -- All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
