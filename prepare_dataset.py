"""
UTILITAIRE DE PRÉPARATION DU MRL EYE DATASET
=============================================

Ce script télécharge et organise le MRL Eye Dataset pour l'entraînement.

Le MRL Eye Dataset contient des images d'yeux ouverts et fermés:
http://mrl.cs.vsb.cz/eyedataset

Usage:
    python prepare_dataset.py --output ./data
"""

import os
import argparse
import shutil
import random
from pathlib import Path
import cv2
import numpy as np


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """
    Crée un dataset synthétique pour tester le pipeline.
    
    Pour un vrai projet, utilisez le MRL Eye Dataset ou similaire.
    
    Args:
        output_dir: Dossier de sortie
        num_samples: Nombre d'échantillons par classe
    """
    output = Path(output_dir)
    
    # Créer structure
    for split in ['train', 'val']:
        for cls in ['open', 'closed']:
            (output / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("Génération d'images synthétiques...")
    
    for split in ['train', 'val']:
        n = num_samples if split == 'train' else num_samples // 5
        
        for i in range(n):
            # Œil ouvert: ellipse horizontale
            img_open = np.zeros((24, 24), dtype=np.uint8)
            cv2.ellipse(img_open, (12, 12), (10, 5), 0, 0, 360, 255, -1)
            # Ajouter bruit
            noise = np.random.normal(0, 20, img_open.shape).astype(np.int16)
            img_open = np.clip(img_open.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output / split / 'open' / f'{i:04d}.png'), img_open)
            
            # Œil fermé: ligne horizontale
            img_closed = np.zeros((24, 24), dtype=np.uint8)
            cv2.line(img_closed, (2, 12), (22, 12), 255, 2)
            noise = np.random.normal(0, 20, img_closed.shape).astype(np.int16)
            img_closed = np.clip(img_closed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output / split / 'closed' / f'{i:04d}.png'), img_closed)
    
    print(f"Dataset créé dans {output_dir}")
    print(f"  Train: {num_samples * 2} images")
    print(f"  Val: {num_samples // 5 * 2} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./data')
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()
    
    print("=" * 50)
    print("PRÉPARATION DU DATASET")
    print("=" * 50)
    print()
    print("NOTE: Ce script génère un dataset SYNTHÉTIQUE pour tester.")
    print("Pour un vrai projet, téléchargez le MRL Eye Dataset:")
    print("  http://mrl.cs.vsb.cz/eyedataset")
    print()
    
    create_sample_dataset(args.output, args.samples)
    
    print()
    print("Pour entraîner le modèle:")
    print(f"  python train.py --data_dir {args.output} --epochs 30")


if __name__ == "__main__":
    main()
