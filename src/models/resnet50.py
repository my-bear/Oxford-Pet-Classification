import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR

class ResNet50(nn.Module):
    """
    ResNet50 с заморозкой всех слоев, кроме layer4 и FC.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(ResNet50, self).__init__()
        
        # Загрузка предобученной модели
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Заморозка всех слоев
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Разморозка layer4
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Получаем размер признаков
        in_features = self.backbone.fc.in_features
        
        # Сохраняем все слои ДО fc (включая conv1, bn1, relu, maxpool, layer1-4, avgpool)
        # Но без самого fc
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        # Теперь self.features выдает тензор [batch, 2048, 1, 1] после avgpool
        
        # Создаем новый классификатор: Flatten → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # [batch, 2048, 1, 1] → [batch, 2048]
            nn.Dropout(dropout_rate),       # Dropout ДО линейного слоя!
            nn.Linear(in_features, num_classes)
        )
        
        # Инициализация весов - правильный способ
        self._init_weights()
        
        # Классификатор обучается
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def _init_weights(self):
        """Инициализация весов классификатора"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.features(x)      # [batch, 2048, 1, 1]
        x = self.classifier(x)    # [batch, num_classes]
        return x
    
    def get_optimizer(self, lr_layer4: float = 1e-4, lr_classifier: float = 1e-3, weight_decay: float = 0.01):
        """Создание оптимизатора с разными learning rates"""
        optimizer = optim.AdamW([
            {'params': self.backbone.layer4.parameters(), 'lr': lr_layer4},
            {'params': self.classifier.parameters(), 'lr': lr_classifier}
        ], weight_decay=weight_decay)
        return optimizer
    
    def freeze_batch_norm(self):
        """Заморозка BatchNorm слоев в layer4"""
        for module in self.backbone.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False



