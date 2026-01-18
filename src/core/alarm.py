"""
Gestionnaire d'alarme sonore.
"""

import time
import numpy as np


class AlarmManager:
    """Gère les alarmes sonores avec cooldown."""
    
    def __init__(self, cooldown: float = 3.0):
        self.cooldown = cooldown
        self.last_time = 0
        self.ready = False
        try:
            import pygame
            pygame.mixer.init()
            sr, dur, freq = 44100, 0.5, 880
            t = np.linspace(0, dur, int(sr * dur), False)
            wave = (np.sin(2 * np.pi * freq * t) * 16383).astype(np.int16)
            self.sound = pygame.sndarray.make_sound(np.column_stack((wave, wave)))
            self.ready = True
        except:
            pass
    
    def trigger(self):
        if self.ready and time.time() - self.last_time >= self.cooldown:
            self.sound.play()
            self.last_time = time.time()
    
    def close(self):
        if self.ready:
            import pygame
            pygame.mixer.quit()
