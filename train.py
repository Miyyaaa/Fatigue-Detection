"""
SCRIPT D'ENTRAÎNEMENT POUR LE MODÈLE DE DÉTECTION DE FATIGUE (VISAGE COMPLET)
=============================================================================

Ce script entraîne le modèle FatigueCNN (MobileNetV2) sur un dataset de visages.

Datasets recommandés:
- UTA-RLDD: https://sites.google.com/view/utarldd/home
- DROZY: https://www.epsto.be/datasets/drozy/
- Créer son propre dataset avec la webcam (voir generate_dataset.py)

Structure attendue:
    data/
    ├── train/
    │   ├── alert/      # Visages alertes (yeux ouverts, pas de bâillement)
    │   └── fatigued/   # Visages fatigués (yeux fermés, bâillements, somnolence)
    └── val/
        ├── alert/
        └── fatigued/

Usage:
    python train.py --data_dir ./data --epochs 30 --batch_size 16
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from fatigue_detector import FatigueCNN


class FaceDataset(Dataset):
    """
    Dataset pour images de visages (Alert/Fatigued).
    """
    
    def __init__(self, root_dir: str, transform=None, img_size: int = 224, use_rgb: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_size = img_size
        self.use_rgb = use_rgb
        self.samples = []
        self.class_to_idx = {'alert': 0, 'fatigued': 1}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((str(img_path), class_idx))
        
        print(f"Dataset: {len(self.samples)} images depuis {root_dir}")
        for name, idx in self.class_to_idx.items():
            count = sum(1 for _, c in self.samples if c == idx)
            print(f"  - {name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        if self.use_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    if len(loader) == 0:
        return 0.0, 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    if total == 0:
        return 0.0, 0.0
    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="Entraînement CNN Fatigue")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='fatigue_model.pth')
    parser.add_argument('--freeze', type=int, default=5, help='Epochs avec backbone gelé')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data augmentation pour MobileNetV2
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        normalize
    ])
    val_transform = normalize
    img_size = 224
    use_rgb = True
    # Datasets
    train_set = FaceDataset(os.path.join(args.data_dir, 'train'), train_transform, img_size, use_rgb)
    val_set = FaceDataset(os.path.join(args.data_dir, 'val'), val_transform, img_size, use_rgb)
    
    # Si validation vide, split automatique 80/20
    if len(val_set) == 0 and len(train_set) > 0:
        print("\n⚠️ Validation vide - Split automatique 80/20 du train")
        from torch.utils.data import random_split
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size
        train_set, val_set = random_split(train_set, [train_size, val_size])
        print(f"  Train: {len(train_set)}, Val: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2)
    
    # Modèle MobileNetV2 avec Transfer Learning
    model = FatigueCNN(pretrained=True).to(device)
    print("Architecture: MobileNetV2 + Transfer Learning (RGB 224x224)")
    model.freeze_backbone()
    print(f"Backbone gelé pour les {args.freeze} premières epochs")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Dégeler le backbone après quelques epochs
        if epoch == args.freeze:
            model.unfreeze_backbone()
            print("🔓 Backbone dégelé - Fine-tuning complet")
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"✅ Modèle sauvegardé: {args.output} (Acc: {val_acc:.2f}%)")
    
    print(f"\n🎉 Entraînement terminé! Meilleure accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
