#!/usr/bin/env python3
"""
SYSTÈME DE DÉTECTION DE FATIGUE EN TEMPS RÉEL
==============================================

Projet DNN - EPITA

Usage:
    python main.py                              # Mode MediaPipe (défaut)
    python main.py --detector mediapipe         # Mode MediaPipe explicite
    python main.py --detector custom            # Mode CNN personnalisé
    python main.py --model models/fatigue.pth   # Avec modèle entraîné

Pour l'aide complète:
    python main.py --help
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import time
import argparse

from src.detectors import create_face_detector
from src.models import FatigueCNN
from src.core import FatigueScorer, AlarmManager


class FatigueDetector:
    """
    Système de détection de fatigue en temps réel.
    
    Supporte deux modes de détection de visages:
    - 'mediapipe': Rapide et précis (production)
    - 'custom': CNN personnalisé avec sliding window (académique)
    """
    
    def __init__(
        self, 
        model_path: str = None, 
        camera_id: int = 0,
        detector_type: str = "mediapipe",
        face_model_path: str = None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Modèle de classification de fatigue
        self.model = FatigueCNN(pretrained=True)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Modèle fatigue chargé: {model_path}")
            except Exception as e:
                print(f"Modèle fatigue non chargé ({e}), utilisation backbone ImageNet")
        self.model.to(self.device).eval()
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        # Détecteur de visages
        self.detector = create_face_detector(
            detector_type=detector_type,
            model_path=face_model_path
        )
        print(f"Détecteur de visages: {self.detector.name}")
        
        self.scorer = FatigueScorer()
        self.alarm = AlarmManager()
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_times = deque(maxlen=30)
    
    def preprocess(self, face: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        tensor = self.normalize(tensor)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, face: np.ndarray) -> float:
        with torch.no_grad():
            output = self.model(self.preprocess(face))
            return F.softmax(output, dim=1)[0, 1].item()
    
    def draw_hud(self, frame, metrics, bbox):
        h, w = frame.shape[:2]
        GREEN, ORANGE, RED, WHITE = (0,255,0), (0,165,255), (0,0,255), (255,255,255)
        level = metrics.get('level', 'normal')
        color = RED if level=='severe' else ORANGE if level=='light' else GREEN
        
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        cv2.rectangle(frame, (10, 10), (280, 120), (0,0,0), -1)
        cv2.rectangle(frame, (10, 10), (280, 120), color, 2)
        
        detector_label = self.detector.name if self.detector else "CNN"
        cv2.putText(frame, f"FATIGUE [{detector_label}]", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
        
        cnn_prob = metrics.get('cnn_prob', 0)
        cv2.putText(frame, f"Fatigue: {cnn_prob:.1%}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        bar_w = 200
        cv2.rectangle(frame, (20, 75), (20+bar_w, 90), (50,50,50), -1)
        cv2.rectangle(frame, (20, 75), (20+int(bar_w*cnn_prob), 90), color, -1)
        
        status = "ALERTE!" if level=='severe' else "ATTENTION" if level=='light' else "OK"
        fps = metrics.get('fps', 0)
        cv2.putText(frame, f"Status: {status}  |  FPS: {fps:.0f}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        
        if metrics.get('alert'):
            cv2.rectangle(frame, (0,0), (w-1,h-1), RED, 10)
            cv2.putText(frame, "!!! ALERTE SOMNOLENCE !!!", (w//2-180, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)
        
        return frame
    
    def run(self):
        print("\nDémarrage... Appuyez sur 'Q' pour quitter.\n")
        
        while True:
            start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            
            face, bbox = self.detector.detect(frame)
            
            if face is not None:
                cnn_prob = self.predict(face)
                metrics = self.scorer.update(cnn_prob)
                
                self.frame_times.append(time.time() - start)
                metrics['fps'] = 1.0 / np.mean(self.frame_times)
                
                if metrics['alert']:
                    self.alarm.trigger()
                
                frame = self.draw_hud(frame, metrics, bbox)
            else:
                cv2.putText(frame, "Aucun visage", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow("Fatigue Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.close()
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        self.alarm.close()


def main():
    parser = argparse.ArgumentParser(
        description="Système de détection de fatigue en temps réel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py                                    # MediaPipe (défaut)
  python main.py --detector custom                  # CNN personnalisé
  python main.py --model models/fatigue_model.pth   # Avec modèle entraîné
  python main.py -d custom --face-model models/face_detector_model.pth
"""
    )
    parser.add_argument('--model', '-m', type=str, default=None,
                       help="Chemin vers le modèle de fatigue (.pth)")
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help="ID de la caméra (défaut: 0)")
    parser.add_argument('--detector', '-d', type=str, default='mediapipe',
                       choices=['mediapipe', 'custom'],
                       help="Type de détecteur de visages (défaut: mediapipe)")
    parser.add_argument('--face-model', type=str, default=None,
                       help="Chemin vers le modèle de détection de visages (.pth)")
    args = parser.parse_args()
    
    FatigueDetector(
        model_path=args.model, 
        camera_id=args.camera,
        detector_type=args.detector,
        face_model_path=args.face_model
    ).run()


if __name__ == "__main__":
    main()
