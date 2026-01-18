"""
Détecteur de visages CNN personnalisé (Académique).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base import BaseFaceDetector


class SimpleFaceCNN(nn.Module):
    """
    CNN léger pour classification binaire (Visage vs Non-Visage).
    
    Architecture:
    - 3 blocs convolutionnels (Conv + BatchNorm + ReLU + MaxPool)
    - 2 couches fully-connected avec dropout
    
    Input: 64x64 RGB images
    Output: 2 classes (no_face, face)
    Paramètres: ~37K (facile à expliquer dans un rapport étudiant)
    """
    
    def __init__(self):
        super(SimpleFaceCNN, self).__init__()
        
        # Bloc 1: 64x64x3 -> 32x32x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Bloc 2: 32x32x16 -> 16x16x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Bloc 3: 16x16x32 -> 8x8x64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Classifier: 8x8x64 = 4096 -> 128 -> 2
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la probabilité de la classe 'face'."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1]


class CustomCNNFaceDetector(BaseFaceDetector):
    """
    Détecteur de visages utilisant un CNN personnalisé avec sliding window.
    
    Approche:
    1. Parcourt l'image avec des fenêtres de différentes tailles
    2. Pour chaque fenêtre, le CNN prédit si c'est un visage
    3. Non-Maximum Suppression pour éliminer les doublons
    
    Note: Plus lent que MediaPipe mais démontre la compréhension des CNN.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        face_size: Tuple[int, int] = (224, 224),
        cnn_input_size: int = 64,
        confidence_threshold: float = 0.7,
        window_scales: Tuple[float, ...] = (0.3, 0.4, 0.5),
        stride_ratio: float = 0.2
    ):
        self.face_size = face_size
        self.cnn_input_size = cnn_input_size
        self.confidence_threshold = confidence_threshold
        self.window_scales = window_scales
        self.stride_ratio = stride_ratio
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SimpleFaceCNN()
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Custom Face CNN chargé: {model_path}")
            except Exception as e:
                print(f"⚠️ Modèle non trouvé ({e}), utilisation de poids aléatoires")
        else:
            print("⚠️ Aucun modèle spécifié, utilisation de poids aléatoires")
        
        self.model.to(self.device).eval()
        print(f"Custom CNN Face Detector: OK (device: {self.device})")
    
    @property
    def name(self) -> str:
        return "Custom CNN"
    
    def _preprocess_window(self, window: np.ndarray) -> torch.Tensor:
        window = cv2.resize(window, (self.cnn_input_size, self.cnn_input_size))
        window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
        window = window.astype(np.float32) / 255.0
        window = np.transpose(window, (2, 0, 1))
        return torch.from_numpy(window).unsqueeze(0).to(self.device)
    
    def _sliding_window(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        windows = []
        for scale in self.window_scales:
            win_size = int(min(h, w) * scale)
            stride = int(win_size * self.stride_ratio)
            for y in range(0, h - win_size + 1, stride):
                for x in range(0, w - win_size + 1, stride):
                    window = frame[y:y+win_size, x:x+win_size]
                    windows.append((x, y, x+win_size, y+win_size, window))
        return windows
    
    def _non_max_suppression(self, detections: list) -> Optional[Tuple]:
        if not detections:
            return None
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        best = detections[0]
        return (best[0], best[1], best[2], best[3])
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
        h, w = frame.shape[:2]
        detections = []
        windows = self._sliding_window(frame)
        
        with torch.no_grad():
            for x1, y1, x2, y2, window in windows:
                tensor = self._preprocess_window(window)
                prob = self.model.predict_proba(tensor).item()
                if prob >= self.confidence_threshold:
                    detections.append((x1, y1, x2, y2, prob))
        
        bbox = self._non_max_suppression(detections)
        if bbox is None:
            return None, None
        
        x1, y1, x2, y2 = bbox
        pad = int((x2 - x1) * 0.15)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None, None
        
        face_roi = cv2.resize(face_roi, self.face_size)
        return face_roi, (x1, y1, x2, y2)
    
    def close(self) -> None:
        pass
