from torch.utils.data import Dataset
import torch
import torchvision

class CachedOxfordIIITPet(Dataset):
    def __init__(self, root, split, transform, cache_in_ram=False):
        # Загружаем оригинальный датасет БЕЗ трансформаций
        self.original_dataset = torchvision.datasets.OxfordIIITPet(
            root=root, 
            split=split, 
            target_types='category',
            download=True,
            transform=None  # важно: None
        )
        self.transform = transform
        self.cache_in_ram = cache_in_ram
        self.cache = {}  # словарь для кэша
        
    def __getitem__(self, idx):
        # Если есть в кэше - берём оттуда
        if idx in self.cache:
            img, target = self.cache[idx]
        else:
            # Загружаем из оригинального датасета
            img, target = self.original_dataset[idx]
            
            # Если включено кэширование в RAM - сохраняем
            if self.cache_in_ram:
                self.cache[idx] = (img, target)
        
        # Применяем трансформации (аугментации)
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.original_dataset)