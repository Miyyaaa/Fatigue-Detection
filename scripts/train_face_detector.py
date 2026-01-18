"""
SCRIPT D'ENTRAÎNEMENT POUR LE CNN DE DÉTECTION DE VISAGES
==========================================================

Ce script entraîne le SimpleFaceCNN pour classifier les fenêtres d'image
comme "visage" ou "non-visage".

NOUVEAU: Téléchargement automatique du dataset!
    python train_face_detector.py --download --epochs 20

Le script télécharge automatiquement:
- LFW (Labeled Faces in the Wild) pour les visages
- Génère des images "non-visage" (bruit, patterns, textures)

Usage:
    python train_face_detector.py --download              # Télécharge + entraîne
    python train_face_detector.py --data_dir ./data/faces # Utilise dataset existant

Note: Ce modèle est conçu pour être simple et explicable dans un rapport.
      Pour la production, utilisez MediaPipe.
"""

import os
import sys
import argparse
import random
import tarfile
import urllib.request
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detectors.custom_cnn import SimpleFaceCNN

# =============================================================================
# TÉLÉCHARGEMENT AUTOMATIQUE DU DATASET
# =============================================================================

# URLs de fallback pour LFW
LFW_URLS = [
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
    "https://ndownloader.figshare.com/files/5976018",  # Mirror
]
DEFAULT_DATA_DIR = "./data/faces"


def download_file(url: str, dest_path: str, desc: str = "Downloading", timeout: int = 30):
    """Télécharge un fichier avec barre de progression et timeout."""
    print(f"📥 {desc}...")
    print(f"   URL: {url}")
    
    try:
        import ssl
        # Créer un contexte SSL permissif pour éviter les erreurs de certificat
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        print(f"\r   Progression: {percent}% ({downloaded//1024//1024}MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n   ❌ Erreur: {e}")
        return False


def try_download_lfw(dest_path: str) -> bool:
    """Essaie de télécharger LFW depuis plusieurs sources."""
    for url in LFW_URLS:
        if download_file(url, dest_path, f"Tentative depuis {url.split('/')[2]}"):
            return True
    return False


def generate_synthetic_faces(output_dir: Path, num_samples: int = 2000, img_size: int = 64):
    """
    Génère des visages synthétiques simples (ellipses avec features).
    Utilisé quand le téléchargement échoue.
    """
    print(f"🎭 Génération de {num_samples} visages synthétiques...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating faces"):
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Couleur de peau aléatoire
        skin_tones = [
            (180, 150, 120), (200, 170, 140), (220, 190, 160),
            (160, 120, 90), (140, 100, 70), (100, 70, 50)
        ]
        skin = random.choice(skin_tones)
        
        # Fond aléatoire
        bg_color = tuple(random.randint(50, 200) for _ in range(3))
        img[:] = bg_color
        
        # Visage (ellipse)
        cx, cy = img_size // 2, img_size // 2
        face_w = random.randint(20, 28)
        face_h = random.randint(24, 32)
        cv2.ellipse(img, (cx, cy), (face_w, face_h), 0, 0, 360, skin, -1)
        
        # Yeux
        eye_y = cy - random.randint(4, 8)
        eye_offset = random.randint(6, 10)
        eye_size = random.randint(2, 4)
        cv2.circle(img, (cx - eye_offset, eye_y), eye_size, (40, 40, 40), -1)
        cv2.circle(img, (cx + eye_offset, eye_y), eye_size, (40, 40, 40), -1)
        
        # Nez
        nose_len = random.randint(3, 6)
        cv2.line(img, (cx, cy - 2), (cx, cy + nose_len), tuple(max(0, c-30) for c in skin), 1)
        
        # Bouche
        mouth_y = cy + random.randint(8, 12)
        mouth_w = random.randint(6, 10)
        cv2.ellipse(img, (cx, mouth_y), (mouth_w, 2), 0, 0, 180, (100, 50, 50), -1)
        
        # Cheveux (optionnel)
        if random.random() > 0.3:
            hair_color = tuple(random.randint(20, 100) for _ in range(3))
            hair_y = cy - face_h + random.randint(-5, 5)
            cv2.ellipse(img, (cx, hair_y), (face_w + 2, 10), 0, 180, 360, hair_color, -1)
        
        # Ajouter du bruit pour réalisme
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"face_{i:05d}.jpg"), img)
    
    print(f"   ✅ {num_samples} visages synthétiques générés")


def generate_negative_samples(output_dir: Path, num_samples: int = 2000, img_size: int = 64):
    """
    Génère des images "non-visage" variées:
    - Bruit aléatoire
    - Patterns géométriques
    - Gradients
    - Textures
    """
    print(f"🎲 Génération de {num_samples} images négatives...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating negatives"):
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        pattern_type = random.choice(['noise', 'gradient', 'circles', 'lines', 'checker', 'solid'])
        
        if pattern_type == 'noise':
            # Bruit aléatoire
            img = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        
        elif pattern_type == 'gradient':
            # Gradient horizontal ou vertical
            for c in range(3):
                if random.random() > 0.5:
                    img[:, :, c] = np.tile(np.linspace(0, 255, img_size), (img_size, 1)).astype(np.uint8)
                else:
                    img[:, :, c] = np.tile(np.linspace(0, 255, img_size), (img_size, 1)).T.astype(np.uint8)
        
        elif pattern_type == 'circles':
            # Cercles aléatoires
            color = tuple(random.randint(0, 255) for _ in range(3))
            bg_color = tuple(random.randint(0, 255) for _ in range(3))
            img[:] = bg_color
            for _ in range(random.randint(2, 8)):
                cx, cy = random.randint(0, img_size), random.randint(0, img_size)
                r = random.randint(5, 30)
                cv2.circle(img, (cx, cy), r, color, -1)
        
        elif pattern_type == 'lines':
            # Lignes aléatoires
            bg_color = tuple(random.randint(0, 255) for _ in range(3))
            img[:] = bg_color
            for _ in range(random.randint(3, 10)):
                pt1 = (random.randint(0, img_size), random.randint(0, img_size))
                pt2 = (random.randint(0, img_size), random.randint(0, img_size))
                color = tuple(random.randint(0, 255) for _ in range(3))
                cv2.line(img, pt1, pt2, color, random.randint(1, 5))
        
        elif pattern_type == 'checker':
            # Damier
            cell_size = random.randint(4, 16)
            c1 = tuple(random.randint(0, 255) for _ in range(3))
            c2 = tuple(random.randint(0, 255) for _ in range(3))
            for y in range(0, img_size, cell_size):
                for x in range(0, img_size, cell_size):
                    color = c1 if (x // cell_size + y // cell_size) % 2 == 0 else c2
                    img[y:y+cell_size, x:x+cell_size] = color
        
        else:  # solid
            # Couleur unie avec variation
            color = tuple(random.randint(0, 255) for _ in range(3))
            img[:] = color
            # Ajouter un peu de bruit
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"neg_{i:05d}.jpg"), img)


def prepare_face_samples(lfw_dir: Path, output_dir: Path, max_samples: int = 2000, img_size: int = 64):
    """Prépare les images de visages depuis LFW."""
    print(f"👤 Préparation des images de visages depuis LFW...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = list(lfw_dir.rglob("*.jpg"))
    random.shuffle(all_images)
    
    count = 0
    for img_path in tqdm(all_images[:max_samples], desc="Processing faces"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # LFW images are 250x250 with face centered
        # Crop center and resize
        h, w = img.shape[:2]
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        img = cv2.resize(img, (img_size, img_size))
        
        cv2.imwrite(str(output_dir / f"face_{count:05d}.jpg"), img)
        count += 1
    
    print(f"   {count} images de visages préparées")
    return count


def download_and_prepare_dataset(data_dir: str = DEFAULT_DATA_DIR, num_samples: int = 2000, offline: bool = False):
    """
    Télécharge LFW et prépare le dataset complet.
    Si offline=True ou si le téléchargement échoue, génère des visages synthétiques.
    """
    data_path = Path(data_dir)
    train_face = data_path / "train" / "face"
    train_noface = data_path / "train" / "no_face"
    val_face = data_path / "val" / "face"
    val_noface = data_path / "val" / "no_face"
    
    # Vérifier si déjà téléchargé
    if train_face.exists() and len(list(train_face.glob("*.jpg"))) > 100:
        print("✅ Dataset déjà préparé!")
        return
    
    print("\n" + "="*60)
    print("TÉLÉCHARGEMENT ET PRÉPARATION DU DATASET")
    print("="*60 + "\n")
    
    lfw_tgz = Path("/tmp/lfw.tgz")
    lfw_dir = Path("/tmp/lfw")
    use_synthetic = offline
    
    # Essayer de télécharger LFW (sauf si mode offline)
    if not offline and not lfw_dir.exists():
        if not lfw_tgz.exists() or lfw_tgz.stat().st_size < 1000000:
            print("🌐 Tentative de téléchargement de LFW...")
            if not try_download_lfw(str(lfw_tgz)):
                print("\n⚠️ Téléchargement impossible. Passage en mode OFFLINE.")
                print("   (Génération de visages synthétiques)\n")
                use_synthetic = True
        
        if not use_synthetic:
            print("📦 Extraction de LFW...")
            try:
                with tarfile.open(lfw_tgz, "r:gz") as tar:
                    tar.extractall("/tmp")
            except Exception as e:
                print(f"❌ Erreur extraction: {e}")
                use_synthetic = True
    
    # Préparer les visages
    temp_faces = Path("/tmp/faces_temp")
    if temp_faces.exists():
        shutil.rmtree(temp_faces)
    temp_faces.mkdir()
    
    if use_synthetic or not lfw_dir.exists():
        # Mode OFFLINE: générer des visages synthétiques
        generate_synthetic_faces(temp_faces, num_samples)
    else:
        # Mode ONLINE: utiliser LFW
        prepare_face_samples(lfw_dir, temp_faces, num_samples)
    
    # Diviser en train/val
    all_faces = list(temp_faces.glob("*.jpg"))
    random.shuffle(all_faces)
    split_idx = int(len(all_faces) * 0.8)
    
    train_face.mkdir(parents=True, exist_ok=True)
    val_face.mkdir(parents=True, exist_ok=True)
    
    for img_path in all_faces[:split_idx]:
        shutil.copy(img_path, train_face / img_path.name)
    for img_path in all_faces[split_idx:]:
        shutil.copy(img_path, val_face / img_path.name)
    
    # 3. Générer les non-visages
    train_neg_count = int(num_samples * 0.8)
    val_neg_count = num_samples - train_neg_count
    
    generate_negative_samples(train_noface, train_neg_count)
    generate_negative_samples(val_noface, val_neg_count)
    
    # Nettoyage
    shutil.rmtree(temp_faces)
    
    print(f"\n✅ Dataset préparé dans: {data_dir}")
    print(f"   Train: {len(list(train_face.glob('*.jpg')))} faces, {len(list(train_noface.glob('*.jpg')))} negatives")
    print(f"   Val:   {len(list(val_face.glob('*.jpg')))} faces, {len(list(val_noface.glob('*.jpg')))} negatives")


# =============================================================================
# DATASET PYTORCH
# =============================================================================

class FaceDetectionDataset(Dataset):
    """
    Dataset pour la détection binaire de visages.
    Charge des images depuis deux dossiers: 'face' et 'no_face'.
    """
    
    def __init__(self, root_dir: str, transform=None, img_size: int = 64):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self.class_to_idx = {'no_face': 0, 'face': 1}
        
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement du CNN de détection de visages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python train_face_detector.py --download              # Télécharge LFW + entraîne
  python train_face_detector.py --download --offline    # Mode hors-ligne (synthétique)
  python train_face_detector.py --download --epochs 30  # Plus d'epochs
  python train_face_detector.py --data_dir ./my_data    # Dataset personnalisé
"""
    )
    parser.add_argument('--download', action='store_true',
                       help="Prépare automatiquement le dataset")
    parser.add_argument('--offline', action='store_true',
                       help="Mode hors-ligne: génère des visages synthétiques au lieu de télécharger")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                       help=f"Dossier du dataset (défaut: {DEFAULT_DATA_DIR})")
    parser.add_argument('--num_samples', type=int, default=2000,
                       help="Nombre d'images par classe (défaut: 2000)")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='face_detector_model.pth')
    args = parser.parse_args()
    
    # Téléchargement/génération automatique si demandé
    if args.download:
        download_and_prepare_dataset(args.data_dir, args.num_samples, offline=args.offline)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
    ])
    
    # Datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    train_set = FaceDetectionDataset(train_dir, train_transform)
    val_set = FaceDetectionDataset(val_dir)
    
    if len(train_set) == 0:
        print("\n❌ Aucune donnée d'entraînement trouvée!")
        print("   Utilisez --download pour télécharger automatiquement le dataset LFW")
        print(f"   Ou créez manuellement: {train_dir}/face/ et {train_dir}/no_face/")
        return
    
    # Si validation vide, split automatique
    if len(val_set) == 0 and len(train_set) > 0:
        print("\n⚠️ Validation vide - Split automatique 80/20")
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size
        train_set, val_set = random_split(train_set, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2)
    
    # Modèle
    model = SimpleFaceCNN().to(device)
    num_params = count_parameters(model)
    print(f"\n" + "="*50)
    print(f"Architecture: SimpleFaceCNN")
    print(f"Paramètres: {num_params:,} (~{num_params/1000:.1f}K)")
    print(f"Input: 64x64 RGB | Output: 2 classes")
    print("="*50 + "\n")
    
    # Entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  ✅ Modèle sauvegardé: {args.output}")
    
    print(f"\n🎉 Entraînement terminé!")
    print(f"   Meilleure accuracy: {best_acc:.2f}%")
    print(f"   Modèle: {args.output}")
    print(f"\n   Pour utiliser: python fatigue_detector.py --detector custom --face-model {args.output}")


if __name__ == "__main__":
    main()

