"""
Détecteur de visages MediaPipe (Production).
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .base import BaseFaceDetector


class MediaPipeFaceDetector(BaseFaceDetector):
    """
    Détecteur de visages utilisant MediaPipe Face Landmarker.
    Rapide et précis - recommandé pour la production.
    """
    
    def __init__(self, face_size: Tuple[int, int] = (224, 224)):
        self.face_size = face_size
        self._init_mediapipe()
    
    @property
    def name(self) -> str:
        return "MediaPipe"
    
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
            print("MediaPipe Face Landmarker: OK")
        except Exception as e:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, 
                min_detection_confidence=0.5
            )
            self.use_new_api = False
            print("MediaPipe Face Mesh (legacy): OK")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        """Détecte et extrait le ROI du visage via MediaPipe."""
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
        
        # Bounding box avec padding
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
    
    def close(self) -> None:
        if self.use_new_api:
            self.landmarker.close()
        else:
            self.face_mesh.close()
