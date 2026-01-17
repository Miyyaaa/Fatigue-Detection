# 🚗 Système de Détection de Fatigue en Temps Réel

**Projet DNN - Détection de somnolence au volant par Deep Learning**

Ce système utilise un réseau de neurones convolutif (CNN) basé sur **MobileNetV2** avec **transfer learning** (pré-entraîné sur ImageNet) pour détecter la fatigue à partir du visage capturé par webcam.

---

## 📋 Table des Matières

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation Rapide](#-utilisation-rapide)
- [Créer son Dataset](#-créer-son-dataset)
- [Entraîner le Modèle](#-entraîner-le-modèle)
- [Structure du Projet](#-structure-du-projet)
- [Explication Technique](#-explication-technique)

---

## 🧠 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────┐
│   Webcam    │ ──► │  MediaPipe   │ ──► │  MobileNetV2    │ ──► │ Alerte  │
│   (Frame)   │     │  (Face ROI)  │     │  (CNN ImageNet) │     │ Sonore  │
└─────────────┘     └──────────────┘     └─────────────────┘     └─────────┘
```

| Composant | Description |
|-----------|-------------|
| **MediaPipe** | Détecte le visage et extrait la région d'intérêt (ROI) |
| **MobileNetV2** | CNN pré-entraîné sur ImageNet (1.4M d'images, 1000 classes) |
| **Transfer Learning** | Fine-tuning du modèle pour 2 classes : Alerte / Fatigué |

---

## � Installation

```bash
# Cloner ou accéder au projet
cd /home/matthias/epita/ing2/dnn/fatigue

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**
- `torch` & `torchvision` - Deep Learning
- `opencv-python` - Traitement vidéo
- `mediapipe` - Détection de visage
- `pygame` - Alarme sonore

---

## 🚀 Utilisation Rapide

### Option 1 : Sans entraînement (backbone ImageNet)
```bash
python fatigue_detector.py
```
Le modèle utilise directement les features ImageNet pour évaluer la fatigue.

### Option 2 : Avec modèle entraîné
```bash
python fatigue_detector.py --model fatigue_model.pth
```

### Contrôles
| Touche | Action |
|--------|--------|
| `Q` | Quitter |
| `R` | Réinitialiser les scores |

---

## 📸 Créer son Dataset

Pour entraîner un modèle personnalisé, capturez des images de votre visage :

```bash
python generate_dataset.py --output ./data --samples 200
```

### Contrôles pendant la capture
| Touche | Action |
|--------|--------|
| `A` | Capturer visage **Alerte** (yeux ouverts, attentif) |
| `F` | Capturer visage **Fatigué** (yeux mi-clos, bâillements) |
| `Q` | Terminer la capture |

**Conseils pour un bon dataset :**
- Variez les expressions et angles
- Capturez dans différentes conditions d'éclairage
- Pour "Fatigué" : fermez les yeux, bâillez, inclinez la tête
- Minimum recommandé : 200 images par classe

---

## 🎓 Entraîner le Modèle

```bash
python train.py --data_dir ./data --epochs 20
```

### Paramètres disponibles
| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--data_dir` | (requis) | Dossier contenant train/ et val/ |
| `--epochs` | 30 | Nombre d'époques |
| `--batch_size` | 16 | Taille des batchs |
| `--lr` | 0.001 | Learning rate |
| `--freeze` | 5 | Epochs avec backbone gelé |
| `--output` | fatigue_model.pth | Fichier de sortie |

### Stratégie de Transfer Learning
1. **Epochs 1-5** : Backbone MobileNetV2 gelé, seule la tête de classification apprend
2. **Epochs 6+** : Backbone dégelé, fine-tuning complet avec LR réduit (×0.1)

---

## � Structure du Projet

```
fatigue/
├── fatigue_detector.py   # Système de détection temps réel
├── train.py              # Script d'entraînement
├── generate_dataset.py   # Capture d'images via webcam
├── requirements.txt      # Dépendances Python
├── fatigue_model.pth     # Modèle entraîné (généré)
├── README.md             # Ce fichier
└── data/                 # Dataset (généré)
    ├── train/
    │   ├── alert/        # Visages alertes
    │   └── fatigued/     # Visages fatigués
    └── val/
        ├── alert/
        └── fatigued/
```

---

## 🔬 Explication Technique

### MobileNetV2

Architecture légère optimisée pour le mobile/embarqué :
- **Inverted Residual Blocks** avec expansion/projection
- **Depthwise Separable Convolutions** pour réduire les paramètres
- Seulement **3.4M de paramètres** (vs 138M pour VGG16)

```
Input (224×224×3)
    │
    ▼
┌─────────────────┐
│  Conv 3×3       │ ── 32 filtres
│  + 17 Blocs IR  │ ── Inverted Residual
│  Conv 1×1       │ ── 1280 features
└────────┬────────┘
         │
    ▼────┴────▼
┌─────────────────┐
│ Global AvgPool  │
│ Dropout (0.3)   │
│ FC 1280→256     │
│ ReLU + Dropout  │
│ FC 256→2        │
└─────────────────┘
    │
    ▼
[Alerte, Fatigué]
```

### Pourquoi Transfer Learning ?

1. **ImageNet features** : Le backbone a appris des features visuelles universelles (bords, textures, formes)
2. **Peu de données nécessaires** : 200-500 images suffisent vs 10k+ pour train from scratch
3. **Entraînement rapide** : Convergence en 10-20 epochs

### Pipeline de Détection

```python
# 1. Capture frame
frame = webcam.read()

# 2. Extraction visage (MediaPipe)
face_roi = mediapipe.detect_face(frame)  # 224×224 RGB

# 3. Prétraitement ImageNet
tensor = normalize(face_roi, mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])

# 4. Inférence CNN
prob_fatigue = model(tensor).softmax()[1]  # Probabilité classe "fatigué"

# 5. Décision
if prob_fatigue > 0.5 pendant 2 secondes:
    trigger_alarm()
```

---

## 📊 Métriques de Sortie

| Métrique | Description |
|----------|-------------|
| **Fatigue %** | Probabilité de fatigue (sortie softmax du CNN) |
| **Status** | OK / ATTENTION / ALERTE selon le seuil |
| **FPS** | Images par seconde traitées |

---

## 🎯 Améliorations Possibles

- [ ] Ajouter des features géométriques (EAR, MAR, pose de tête)
- [ ] Implémenter PERCLOS (% temps yeux fermés)
- [ ] Data augmentation plus agressive
- [ ] Exporter en ONNX pour déploiement embarqué
- [ ] Tester d'autres backbones (EfficientNet, ResNet18)

---

## 📚 Références

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

**Projet DNN - EPITA 2026**
