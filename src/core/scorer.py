"""
Système de scoring de fatigue.
"""

import numpy as np
from collections import deque


class FatigueScorer:
    """Score de fatigue basé sur les prédictions du CNN."""
    
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
