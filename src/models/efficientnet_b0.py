import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import cast

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 с полной заморозкой всех слоев.
    Обучается только классификатор.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.2):
        super(EfficientNetB0, self).__init__()
        
        # Загрузка предобученной модели EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Заморозка ВСЕХ слоев
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Получаем размер признаков перед классификатором
        # EfficientNet-B0 имеет classifier: Dropout -> Linear
        in_features = cast(int, self.backbone.classifier[1].in_features)  # 1280 для EfficientNet-B0
        
        # Сохраняем feature extractor (все слои до классификатора)
        self.features = cast(nn.Sequential, self.backbone.features)
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Адаптивный пулинг до 1x1

        # Создаем новый классификатор
        # Оригинальная структура: Dropout(p=0.2) -> Linear(1280, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        # Классификатор обучается
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Инициализация весов классификатора
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов классификатора"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.features(x)           # [batch, 1280, H, W] (H и W зависят от входа)
        x = self.gap(x)                # Adaptive Global Average Pooling: [batch, 1280, 1, 1]
        x = x.flatten(1)               # [batch, 1280]
        x = self.classifier(x)         # [batch, num_classes]
        return x
    
    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 0.01):
        """Создание оптимизатора только для классификатора"""
        optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        return optimizer
    
    def freeze_batch_norm(self):
        """Заморозка BatchNorm слоев в features (если нужно)"""
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False