# Face Detectors Module
from .base import BaseFaceDetector
from .mediapipe_detector import MediaPipeFaceDetector
from .custom_cnn import SimpleFaceCNN, CustomCNNFaceDetector

def create_face_detector(detector_type: str = "mediapipe", model_path=None, **kwargs):
    """Factory pour créer le détecteur approprié."""
    detector_type = detector_type.lower()
    if detector_type == "mediapipe":
        return MediaPipeFaceDetector(**kwargs)
    elif detector_type == "custom":
        return CustomCNNFaceDetector(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Type inconnu: '{detector_type}'. Choix: 'mediapipe', 'custom'")

__all__ = [
    'BaseFaceDetector', 
    'MediaPipeFaceDetector', 
    'SimpleFaceCNN', 
    'CustomCNNFaceDetector',
    'create_face_detector'
]
