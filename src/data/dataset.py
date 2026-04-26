"""
Dataset classes and data loading utilities using torchvision.datasets.OxfordIIITPet.
"""
import torch
import torchvision
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from .CachedOxfordIIITPet import CachedOxfordIIITPet
from torchvision import transforms
from collections import Counter
import polars as pl

def _get_transforms(mode: str = 'train', image_size: int = 224) -> transforms.Compose:
    """Get transforms for train or validation."""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
        ])


def load_datasets(
    root_dir: str = './data/processed',
    image_size: int = 224
) -> Tuple[Dataset, Dataset]:
    """
    Create train, val, test datasets from Oxford-IIIT Pet Dataset.
    
    Args:
        root_dir: Root directory for dataset
        image_size: Target image size
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """

    # Для обучения используем train с аугментациями
    train_dataset = CachedOxfordIIITPet(
        root=root_dir,
        split='trainval',
        transform=_get_transforms('train', image_size),
        cache_in_ram=True  # теперь это параметр вашего класса
    )
    
    val_dataset = CachedOxfordIIITPet(
        root=root_dir,
        split='test',
        transform=_get_transforms('val', image_size),
        cache_in_ram=True  # теперь это параметр вашего класса
    )

 
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    pin_memory=True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train, validation and test dataloaders.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # отбрасываем последний неполный батч для стабильности
        persistent_workers=True,  # воркеры не пересоздаются каждую эпоху
        prefetch_factor=2,  # предзагрузка батчей
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,  # воркеры не пересоздаются каждую эпоху
        prefetch_factor=2,  # предзагрузка батчей
    )
    
    return train_loader, val_loader


def get_dataset_info(dataset) -> dict:
    """Get information about the dataset."""
    
    # Получаем размер (работает для всех Dataset)
    size = len(dataset)  # это всегда работает
    
    info = {
        'size': size,
    }
    
    # Пытаемся получить классы из разных источников
    if hasattr(dataset, 'classes'):
        # Для OxfordIIITPet
        # info['classes'] = dataset.classes
        info['num_classes'] = len(dataset.classes)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
        # Для Subset, который оборачивает OxfordIIITPet
        # info['classes'] = dataset.dataset.classes
        info['num_classes'] = len(dataset.dataset.classes)
    
    return info

def denormalize_image(tensor):
    """
    Денормализация одного изображения
    tensor: (C, H, W) в нормализованном виде
    возвращает: (H, W, C) в диапазоне [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Отменяем нормализацию
    tensor = tensor * std + mean
    
    # Клиппируем в диапазон [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Меняем порядок осей (C, H, W) -> (H, W, C)
    tensor = tensor.permute(1, 2, 0)
    
    return tensor.numpy()

def create_class_distribution_table(train_dataset, test_dataset=None):
    """
    Создает таблицу распределения классов для train и test выборок
    
    Parameters:
    -----------
    train_dataset : Dataset
        Обучающий датасет (должен иметь атрибуты _labels и classes)
    test_dataset : Dataset, optional
        Тестовый датасет (если None, выводит только train)
    
    Returns:
    --------
    pl.DataFrame
        Таблица с распределением классов
    """
    # Считаем распределение для train
    train_counts = Counter(train_dataset._labels)
    
    # Создаем базовый DataFrame
    data = []
    for idx in sorted(train_counts.keys()):
        row = {
            "class_name": train_dataset.classes[idx],
            "train_count": train_counts[idx]
        }
        
        # Добавляем test_count если есть test_dataset
        if test_dataset is not None:
            test_counts = Counter(test_dataset._labels)
            row["test_count"] = test_counts.get(idx, 0)
        
        data.append(row)
    
    df = pl.DataFrame(data)
    
    # Добавляем столбцы с процентами и итогами
    if test_dataset is not None:
        df = df.with_columns([
            (pl.col("train_count") / pl.col("train_count").sum() * 100).round(2).alias("train_%"),
            (pl.col("test_count") / pl.col("test_count").sum() * 100).round(2).alias("test_%")
        ])
        
        # Добавляем строку с итогами
        total_row = {
            "class_name": "TOTAL",
            "train_count": df["train_count"].sum(),
            "test_count": df["test_count"].sum(),
            "train_%": 100.0,
            "test_%": 100.0
        }
        df = pl.concat([df, pl.DataFrame([total_row])])
        
    else:
        df = df.with_columns([
            (pl.col("train_count") / pl.col("train_count").sum() * 100).round(2).alias("percentage")
        ])
        
        total_row = {
            "class_name": "TOTAL",
            "train_count": df["train_count"].sum(),
            "percentage": 100.0
        }
        df = pl.concat([df, pl.DataFrame([total_row])])
    
    return df