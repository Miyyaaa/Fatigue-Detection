"""
Classe abstraite pour les détecteurs de visages.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseFaceDetector(ABC):
    """
    Classe abstraite pour les détecteurs de visages.
    Permet d'interchanger facilement les backends de détection.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du détecteur (pour affichage dans le HUD)."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Détecte un visage dans une frame.
        
        Args:
            frame: Image BGR (numpy array)
            
        Returns:
            Tuple (face_roi, bbox) où:
            - face_roi: ROI du visage redimensionné (224x224) ou None
            - bbox: (x1, y1, x2, y2) ou None si aucun visage
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Libère les ressources."""
        pass
