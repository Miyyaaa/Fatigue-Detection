# 🚗 Système de Détection de Fatigue en Temps Réel

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev/)

**Projet DNN - Détection de somnolence au volant par Deep Learning**

Ce système utilise un CNN (MobileNetV2 + Transfer Learning) pour détecter la fatigue en temps réel via webcam. Le projet supporte **deux modes de détection de visages** : MediaPipe (production) et CNN personnalisé (démonstration académique).

---

## 📋 Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation Rapide](#-utilisation-rapide)
- [Modes de Détection](#-modes-de-détection)
- [Entraînement](#-entraînement)
- [Structure du Projet](#-structure-du-projet)
- [Architecture Technique](#-architecture-technique)

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 🎯 **Détection temps réel** | Analyse vidéo via webcam |
| 🔄 **Double backend** | MediaPipe (rapide) ou CNN custom (académique) |
| 🧠 **Transfer Learning** | MobileNetV2 pré-entraîné sur ImageNet |
| 🔊 **Alarme sonore** | Alerte en cas de somnolence |
| 📊 **HUD informatif** | Score de fatigue, FPS, statut |

---

## 🔧 Installation

```bash
# Cloner le projet
git clone <repo_url>
cd Fatigue-Detection

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**
- `torch` & `torchvision` - Deep Learning
- `opencv-python` - Traitement vidéo
- `mediapipe` - Détection de visage (optionnel)
- `pygame` - Alarme sonore

---

## 🚀 Utilisation Rapide

### Lancer la détection (Mode MediaPipe - défaut)
```bash
python main.py
```

### Avec modèle de fatigue entraîné
```bash
python main.py --model models/fatigue_model.pth
```

### Mode CNN personnalisé
```bash
python main.py --detector custom --face-model models/face_detector_model.pth
```

### Options disponibles
```bash
python main.py --help
```

| Option | Description |
|--------|-------------|
| `-d, --detector` | `mediapipe` (défaut) ou `custom` |
| `-m, --model` | Chemin vers le modèle de fatigue |
| `--face-model` | Chemin vers le modèle de détection de visages |
| `-c, --camera` | ID de la caméra (défaut: 0) |

---

## 🔀 Modes de Détection

### Mode A : MediaPipe (Production)
```bash
python main.py --detector mediapipe
```
- ✅ Rapide (~30+ FPS)
- ✅ Précis
- ❌ Dépend de Google MediaPipe

### Mode B : CNN Personnalisé (Académique)
```bash
python main.py --detector custom --face-model models/face_detector_model.pth
```
- ✅ Architecture maîtrisée (pour rapport)
- ✅ ~37K paramètres (explicable)
- ❌ Plus lent (sliding window)

Le CNN personnalisé utilise une approche **sliding window** avec un classificateur binaire (Face vs Non-Face).

---

## 🎓 Entraînement

### 1. Entraîner le détecteur de visages (Custom CNN)
```bash
# Télécharge automatiquement LFW + génère le dataset
python scripts/train_face_detector.py --download --epochs 20

# Mode hors-ligne (visages synthétiques)
python scripts/train_face_detector.py --download --offline --epochs 20
```

### 2. Créer un dataset de fatigue
```bash
python scripts/generate_dataset.py --output ./data/fatigue --samples 200
```
| Touche | Action |
|--------|--------|
| `A` | Capturer visage **Alerte** |
| `F` | Capturer visage **Fatigué** |
| `Q` | Quitter |

### 3. Entraîner le modèle de fatigue
```bash
python scripts/train_fatigue.py --data_dir ./data/fatigue --epochs 20
```

---

## 📁 Structure du Projet

```
Fatigue-Detection/
├── main.py                     # Point d'entrée principal
├── requirements.txt
├── README.md
│
├── src/                        # Code source modulaire
│   ├── detectors/              # Détecteurs de visages
│   │   ├── base.py             # Classe abstraite
│   │   ├── mediapipe_detector.py
│   │   └── custom_cnn.py       # SimpleFaceCNN
│   ├── models/
│   │   └── fatigue_cnn.py      # MobileNetV2
│   └── core/
│       ├── scorer.py           # Scoring de fatigue
│       └── alarm.py            # Gestion alarmes
│
├── scripts/                    # Scripts utilitaires
│   ├── train_fatigue.py        # Entraîner modèle fatigue
│   ├── train_face_detector.py  # Entraîner CNN custom
│   └── generate_dataset.py     # Capturer dataset
│
├── models/                     # Modèles sauvegardés (.pth)
│   ├── fatigue_model.pth
│   └── face_detector_model.pth
│
└── data/                       # Datasets
    ├── fatigue/                # Dataset de fatigue
    └── faces/                  # Dataset de visages
```

---

## 🔬 Architecture Technique

### Pipeline de Détection

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────┐
│   Webcam    │ ──► │  Face Detector   │ ──► │  MobileNetV2    │ ──► │ Alerte  │
│   (Frame)   │     │  (MediaPipe/CNN) │     │  (Fatigue CNN)  │     │ Sonore  │
└─────────────┘     └──────────────────┘     └─────────────────┘     └─────────┘
```

### SimpleFaceCNN (~37K paramètres)
```
Input: 64×64×3 RGB
    ↓
Conv(3→16) + BN + ReLU + MaxPool  →  32×32×16
Conv(16→32) + BN + ReLU + MaxPool →  16×16×32
Conv(32→64) + BN + ReLU + MaxPool →  8×8×64
    ↓
Flatten → FC(4096→128) → ReLU → Dropout
FC(128→2) → Output [no_face, face]
```

### FatigueCNN (MobileNetV2)
- **Backbone**: MobileNetV2 pré-entraîné (ImageNet)
- **Head**: FC(1280→256→2)
- **Transfer Learning**: Backbone gelé puis fine-tuning

---

## 📚 Références

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)

---

**Projet DNN - EPITA 2026**
