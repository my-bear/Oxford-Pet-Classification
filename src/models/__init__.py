# Импортируем классы прямо в __init__.py
from .models import load_model, confusion_matrix
from .ModernCNN import ModernCNN
from .trainer import Trainer
from .resnet50 import ResNet50
from .vgg19 import VGG19