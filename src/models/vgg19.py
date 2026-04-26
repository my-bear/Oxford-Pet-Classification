import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import cast

class VGG19(nn.Module):
    """
    VGG19 с заморозкой всех сверточных слоев.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(VGG19, self).__init__()
        
        # Загрузка предобученной модели VGG19
        self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Заморозка всех слоев
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Сохраняем feature extractor (все сверточные слои)
        self.features = cast(nn.Sequential, self.backbone.features)
        
        # Создаем новый классификатор
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
        # Размораживаем классификатор
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Инициализируем веса
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов классификатора"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.features(x)           # [batch, 512, 7, 7]
        x = x.view(x.size(0), -1)      # [batch, 25088]
        x = self.classifier(x)         # [batch, num_classes]
        return x
    
    def get_optimizer(self, lr_classifier: float = 1e-3, weight_decay: float = 0.01):
        """Создание оптимизатора только для классификатора"""
        optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=lr_classifier,
            weight_decay=weight_decay
        )
        return optimizer
