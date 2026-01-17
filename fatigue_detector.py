"""
SYSTÈME DE DÉTECTION DE FATIGUE - CNN MobileNetV2 (ImageNet)
=============================================================

Utilise MobileNetV2 pré-entraîné sur ImageNet pour analyser le visage complet.
Le modèle apprend automatiquement les features pertinentes pour la fatigue.

Pipeline: Frame -> Face Detection -> CNN Classification -> Alert
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from collections import deque
import time
from typing import Tuple, Optional
import pygame


# =============================================================================
# MODÈLE CNN (MobileNetV2 ImageNet)
# =============================================================================

class FatigueCNN(nn.Module):
    """
    CNN MobileNetV2 pré-entraîné sur ImageNet.
    Le backbone extrait les features visuelles du visage.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(FatigueCNN, self).__init__()
        
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True


# =============================================================================
# EXTRACTEUR DE VISAGE (MEDIAPIPE)
# =============================================================================

class FaceExtractor:
    """Extrait le visage via MediaPipe."""
    
    def __init__(self, face_size: Tuple[int, int] = (224, 224)):
        self.face_size = face_size
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            import urllib.request
            import os
            
            model_path = '/tmp/face_landmarker.task'
            if not os.path.exists(model_path):
                print("Téléchargement du modèle MediaPipe...")
                url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
                urllib.request.urlretrieve(url, model_path)
            
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.use_new_api = True
            print("MediaPipe: OK")
        except Exception as e:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
            self.use_new_api = False
    
    def extract(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Extrait le ROI du visage. Retourne (face_roi, bbox)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.use_new_api:
            import mediapipe as mp_img
            mp_image = mp_img.Image(image_format=mp_img.ImageFormat.SRGB, data=rgb)
            results = self.landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None, None
            lm = results.face_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        else:
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None, None
            lm = results.multi_face_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        
        # Bounding box
        xs, ys = zip(*pts)
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        pad = int((x2 - x1) * 0.15)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None, None
        
        face_roi = cv2.resize(face_roi, self.face_size)
        return face_roi, (x1, y1, x2, y2)
    
    def close(self):
        if self.use_new_api:
            self.landmarker.close()
        else:
            self.face_mesh.close()


# =============================================================================
# SCOREUR SIMPLE
# =============================================================================

class FatigueScorer:
    """Score de fatigue basé uniquement sur le CNN."""
    
    def __init__(self, window_size: int = 150, fps: int = 30):
        self.predictions = deque(maxlen=window_size)
        self.consec_fatigue = 0
        self.alert_threshold = int(fps * 2)
    
    def update(self, cnn_prob: float) -> dict:
        self.predictions.append(cnn_prob)
        
        is_fatigued = cnn_prob > 0.5
        self.consec_fatigue = self.consec_fatigue + 1 if is_fatigued else 0
        
        avg_prob = np.mean(list(self.predictions)[-30:]) if self.predictions else 0
        
        if avg_prob > 0.6:
            level = 'severe'
        elif avg_prob > 0.4:
            level = 'light'
        else:
            level = 'normal'
        
        return {
            'cnn_prob': cnn_prob,
            'avg_prob': avg_prob,
            'level': level,
            'consec': self.consec_fatigue,
            'alert': level == 'severe' or self.consec_fatigue >= self.alert_threshold
        }
    
    def reset(self):
        self.predictions.clear()
        self.consec_fatigue = 0


# =============================================================================
# ALARME
# =============================================================================

class AlarmManager:
    def __init__(self, cooldown: float = 3.0):
        self.cooldown = cooldown
        self.last_time = 0
        self.ready = False
        try:
            pygame.mixer.init()
            sr, dur, freq = 44100, 0.5, 880
            t = np.linspace(0, dur, int(sr * dur), False)
            wave = (np.sin(2 * np.pi * freq * t) * 16383).astype(np.int16)
            self.sound = pygame.sndarray.make_sound(np.column_stack((wave, wave)))
            self.ready = True
        except: pass
    
    def trigger(self):
        if self.ready and time.time() - self.last_time >= self.cooldown:
            self.sound.play()
            self.last_time = time.time()
    
    def close(self):
        if self.ready: pygame.mixer.quit()


# =============================================================================
# SYSTÈME PRINCIPAL
# =============================================================================

class FatigueDetector:
    """Détection de fatigue par CNN MobileNetV2."""
    
    def __init__(self, model_path: str = None, camera_id: int = 0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model = FatigueCNN(pretrained=True)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Modèle chargé: {model_path}")
            except Exception as e:
                print(f"Modèle non chargé ({e}), utilisation backbone ImageNet")
        self.model.to(self.device).eval()
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        self.extractor = FaceExtractor()
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
        
        # Cadre visage
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # HUD
        cv2.rectangle(frame, (10, 10), (280, 120), (0,0,0), -1)
        cv2.rectangle(frame, (10, 10), (280, 120), color, 2)
        
        cv2.putText(frame, "FATIGUE DETECTOR (CNN)", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
        
        # Score
        cnn_prob = metrics.get('cnn_prob', 0)
        cv2.putText(frame, f"Fatigue: {cnn_prob:.1%}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Barre
        bar_w = 200
        cv2.rectangle(frame, (20, 75), (20+bar_w, 90), (50,50,50), -1)
        cv2.rectangle(frame, (20, 75), (20+int(bar_w*cnn_prob), 90), color, -1)
        
        # Status
        status = "ALERTE!" if level=='severe' else "ATTENTION" if level=='light' else "OK"
        fps = metrics.get('fps', 0)
        cv2.putText(frame, f"Status: {status}  |  FPS: {fps:.0f}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        
        # Alerte
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
            if not ret: continue
            frame = cv2.flip(frame, 1)
            
            face, bbox = self.extractor.extract(frame)
            
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
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.close()
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        self.alarm.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--camera', '-c', type=int, default=0)
    args = parser.parse_args()
    
    FatigueDetector(model_path=args.model, camera_id=args.camera).run()
