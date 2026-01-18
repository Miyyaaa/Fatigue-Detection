"""
Modèle CNN de détection de fatigue (MobileNetV2 + Transfer Learning).
"""

import torch
import torch.nn as nn
from torchvision import models


class FatigueCNN(nn.Module):
    """
    CNN MobileNetV2 pré-entraîné sur ImageNet.
    Le backbone extrait les features visuelles du visage.
    
    Architecture:
    - Backbone: MobileNetV2 (ImageNet weights)
    - Head: Dropout -> FC(1280->256) -> ReLU -> Dropout -> FC(256->2)
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
        """Gèle le backbone pour le transfer learning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Dégèle le backbone pour le fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
