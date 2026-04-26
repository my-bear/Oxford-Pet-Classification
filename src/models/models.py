"""
Model definitions.
"""
import torch
import torch.nn as nn
from .resnet50 import ResNet50
from .vgg19 import VGG19
from .efficientnet_b0 import EfficientNetB0
from .ModernCNN import ModernCNN
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import polars as pl


def load_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Create a model for image classification.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
    
    model_name = model_name.lower()

    match model_name:
        case 'vgg19':
            model = VGG19(num_classes)
        case 'resnet50':
            model = ResNet50(num_classes)
        case 'efficientnet_b0':
            model = EfficientNetB0(num_classes)
        case 'modern_cnn':
            model = ModernCNN(num_classes)
        case _:
            raise ValueError(
                f"Модель '{model_name}' не поддерживается. "
                f"Доступные модели: vgg19, resnet50, efficientnet_b0, modern_cnn"
            )
    
    return model


def confusion_matrix(model, data_loader, device, num_classes, class_names=None):
    """
    Строит и выводит нормализованную матрицу неточностей.
    
    Args:
        model: обученная модель
        data_loader: DataLoader с данными
        device: устройство ('cuda' или 'cpu')
        num_classes: количество классов
        class_names: список названий классов (опционально)
    
    Returns:
        polars.DataFrame: нормализованная матрица неточностей
    """
    model.eval()
    
    # Собираем предсказания и метки
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Создаем матрицу неточностей
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    
    # Нормализуем по строкам
    cm_normalized = cm.astype('float64') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # заменяем NaN на 0
    
    # Создаем polars DataFrame
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    df_cm = pl.DataFrame(
        cm_normalized,
        schema=class_names,
        orient='row'
    ).with_columns(
        pl.Series('True Label', class_names)
    ).select(['True Label'] + class_names)
    
    # Визуализация
    # Создаем цветовую карту
    colors = ['white', '#d0e1f9', '#4a90e2', '#1a4d8c', '#0a2a4a']
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=256)
    
    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.8), max(8, num_classes * 0.7)))
    
    # Рисуем матрицу
    im = ax.imshow(cm_normalized, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    # Настройка осей
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Добавляем значения в ячейки
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_normalized[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            
            if value >= 0.01:
                text = f'{value:.1%}' if value < 0.995 else '100'
            else:
                text = '0'
            
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Value (per row)', fontsize=10)
    
    # Сетка
    ax.set_xticks(np.arange(num_classes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_classes + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    plt.tight_layout()
    plt.show()