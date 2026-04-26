"""
Training loop logic.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from src.data import BatchTimer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class Trainer:
    """
    Класс для обучения и валидации моделей глубокого обучения.
    
    Обеспечивает полный цикл обучения: forward pass, backward pass,
    обновление весов, валидацию, сохранение чекпоинтов и логирование.
    
    Attributes:
        model (nn.Module): Нейронная сеть для обучения
        train_loader (DataLoader): DataLoader для обучающей выборки
        val_loader (DataLoader): DataLoader для валидационной выборки
        optimizer (torch.optim.Optimizer): Оптимизатор (Adam, SGD и т.д.)
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): 
            Планировщик скорости обучения (например, CosineAnnealingLR)
        criterion (Optional[nn.Module]): Функция потерь (CrossEntropyLoss, MSELoss и т.д.)
        device (torch.device): Устройство для обучения ('cuda' или 'cpu')
        output_dir (Path): Директория для сохранения моделей и логов
        logger (Optional[logging.Logger]): Логгер для записи процесса обучения
    
    Methods:
        train_epoch(): Обучает модель одну эпоху
        validate(): Выполняет валидацию модели
        save_checkpoint(): Сохраняет состояние модели и оптимизатора
        load_checkpoint(): Загружает сохранённое состояние
        train(): Запускает полный цикл обучения
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device('cuda'),
        output_dir: Path = Path('./models'),
        logger: Optional[logging.Logger] = None,
    ):
        '''
        Инициализация тренировочного класса.
        
        Args:
            model: Нейронная сеть для обучения
            train_loader: DataLoader с обучающими данными
            val_loader: DataLoader с валидационными данными
            optimizer: Оптимизатор (Adam, SGD, AdamW и т.д.)
            scheduler: Планировщик learning rate (опционально)
            criterion: Функция потерь (по умолчанию CrossEntropyLoss)
            device: Устройство ('cuda' или 'cpu'), по умолчанию 'cuda'
            output_dir: Директория для сохранения моделей
            logger: Логгер для записи процесса (опционально)
        '''
        self.model = model
        self.model = self.model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.batchtimer = BatchTimer()
        
        self.batch_mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.batch_std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

        # Создаем директорию
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Функция потерь
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Метрики
        self.best_metric = 0.0
    
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера по умолчанию."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()

        # Переменные для накопления метрик на GPU (без синхронизации!)
        total_loss = torch.tensor(0.0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total = 0

        # Частота обновления прогресс-бар
        log_interval = max(1, len(self.train_loader) // 10)  # примерно 10 раз за эпоху

        pbar = tqdm(self.train_loader,
                    ncols=60,
                    bar_format='{l_bar}{bar:40}{r_bar}',
                    desc=f'Epoch {epoch} [train]')

        self.batchtimer.last_time = time.time()

        for batch_idx, (data, target) in enumerate(pbar):
            self.batchtimer.tick()

            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Нормализация прямо на GPU
            data = (data - self.batch_mean) / self.batch_std

            output = self.model(data)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach()
            # Количество правильных ответов в батче
            correct += (output.argmax(dim=1) == target).sum()
            total += target.size(0)

            self.batchtimer.tick_end_of_batch()

            if (batch_idx + 1) % log_interval == 0 or (batch_idx == len(self.train_loader) - 1):
                # Единственное место, где CPU дожидается GPU
                avg_loss = total_loss.item() / (batch_idx + 1)
                accuracy = 100. * correct.item() / total

                pbar.set_postfix({
                    'loss': avg_loss,
                    'acc': accuracy,
                })
                pbar.set_description(
                    f'Epoch {epoch} [train]: {self.batchtimer.avg*1000:.1f}ms'
                )

        # Финальное обновление
        final_acc = correct.item() / total
        return final_acc

    # def train_epoch(self, epoch: int) -> float:
    #     """
    #     Обучает модель одну эпох.

    #     Args:
    #         epoch: Текущий номер эпохи
    #     Returns:
    #         float: Средняя потеря на эпоху
    #     """
    #     self.model.train()
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0

    #     pbar = tqdm(self.train_loader, 
    #                 ncols=60, 
    #                 bar_format='{l_bar}{bar:40}{r_bar}', 
    #                 desc=f'Epoch {epoch} [train]'
    #     )

    #     self.batchtimer.last_time = time.time()  # инициализация

    #     for batch_idx, (data, target) in enumerate(pbar):
    #         self.batchtimer.tick()  # замеряем время между батчами
    #         data, target = data.to(self.device), target.to(self.device)
            
    #         # Применяем нормализацию на GPU
    #         data = (data - self.batch_mean) / self.batch_std

    #         # Forward
    #         # тензор формы (batch_size, num_classes) с сырыми логитами
    #         output = self.model(data)
    #         # тензор PyTorch, содержащий скалярное значение функции потерь
    #         # (например, CrossEntropyLoss) для текущего батча.
    #         loss = self.criterion(output, target)
            
    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    #         # Обновляем описание tqdm с текущим средним временем
    #         pbar.set_description(f'Epoch {epoch} [train]: {self.batchtimer.avg*1000:.1f}ms')
            
    #         # Metrics
    #         # .item() - метод, извлекающий Python‑число из тензора с одним элементом.
    #         # Накапливаем суммарную потерю (loss) по всем батчам
    #         total_loss += loss.item()
    #         # Получаем предсказанные классы для всех примеров в батче.
    #         pred = output.argmax(dim=1)
    #         # Подсчитываем число правильно классифицированных примеров в батче
    #         correct += pred.eq(target).sum().item()
    #         # Считаем общее число обработанных примеров (размер датасета) за эпоху.
    #         total += target.size(0)
            
    #         # Update progress bar
    #         pbar.set_postfix({
    #             'loss': total_loss / (batch_idx + 1),
    #             'acc': 100. * correct / total,
    #         })

    #          # В конце батча - сбрасываем таймер для следующего замера
    #         self.batchtimer.tick_end_of_batch()
        
    #     return correct / total
    
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Выполняет быструю валидацию модели.
        
        Returns:
            dict: {'loss': float, 'accuracy': float}
        """
        loader = loader or self.val_loader
        self.model.eval()
        
        # Накопление метрик на GPU для скорости
        total_loss = torch.tensor(0.0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total = 0
        
        log_interval = max(1, len(loader) // 5)  # обновляем прогресс-бар 5 раз за валидацию
        
        pbar = tqdm(loader, 
                    ncols=60, 
                    bar_format='{l_bar}{bar:40}{r_bar}', 
                    desc=f'[val]')
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                # Асинхронная загрузка на GPU
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Нормализация на GPU
                data = (data - self.batch_mean) / self.batch_std
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Накопление метрик на GPU (без синхронизации)
                total_loss += loss.detach()
                correct += (output.argmax(dim=1) == target).sum()
                total += target.size(0)
                
                # Редкое обновление прогресс-бара
                if (batch_idx + 1) % log_interval == 0 or batch_idx == len(loader) - 1:
                    # Единственная синхронизация — только когда обновляем бар
                    avg_loss = total_loss.item() / (batch_idx + 1)
                    avg_acc = 100. * correct.item() / total
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'acc': f'{avg_acc:.2f}%'
                    })
        
        # Финальные метрики
        final_loss = total_loss.item() / len(loader)
        final_acc = correct.item() / total
        
        return {
            'loss': final_loss,
            'accuracy': final_acc,
        }

    # def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
    #     """
    #     Выполняет валидацию модели.
        
    #     Returns:
    #         tuple[float, float]: (средняя потеря, точность в процентах)
    #     """
    #     loader = loader or self.val_loader
    #     self.model.eval()
        
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0
        

    #     pbar = tqdm(loader, 
    #         ncols=60, 
    #         bar_format='{l_bar}{bar:40}{r_bar}', 
    #         desc=f'[val]'
    #     )
    #     with torch.no_grad():
    #         for data, target in pbar:
    #             data, target = data.to(self.device), target.to(self.device)

    #             # Применяем нормализацию на GPU
    #             data = (data - self.batch_mean) / self.batch_std
                
    #             output = self.model(data)
    #             loss = self.criterion(output, target)
                
    #             total_loss += loss.item()
    #             pred = output.argmax(dim=1)
    #             correct += pred.eq(target).sum().item()
    #             total += target.size(0)
        
    #     return {
    #         'loss': total_loss / len(loader),
    #         'accuracy': correct / total,
    #     }
    
    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """
        Сохраняет чекпоинт модели.
        
        Args:
            epoch: Номер эпохи
            metric: Точность предсказания
            is_best: Если True, сохраняет как лучшую модель
        """

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save last checkpoint
        torch.save(checkpoint, self.output_dir / 'last_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            self.logger.info(f"New best model saved with metric: {metric:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Any:
        """
        Загружает чекпоинт модели.
        
        Args:
            checkpoint_path: Путь к файлу чекпоинта
        """

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('metric', 0.0)

        return checkpoint

    
    def train(
        self, 
        num_epochs: int, 
        start_epoch: int = 0, 
        patience: int = 10,
        early_stopping_metric: str = 'loss',  # 'loss' или 'accuracy'
        min_delta: float = 1e-2
    ) -> float:
        """
        Запускает полный цикл обучения с поддержкой ранней остановки.
        
        Args:
            num_epochs: Количество эпох для обучения
            start_epoch: С какой эпохи продолжить обучение
            patience: Количество эпох без улучшения для ранней остановки (по умолчанию 10)
            early_stopping_metric: Метрика для отслеживания ('loss' или 'accuracy')
            min_delta: Минимальное изменение для определения улучшения
        
        Returns:
            Лучшее значение метрики
        """
        
        # Инициализация счетчика ранней остановки
        early_stopping_counter = 0
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_acc = self.train_epoch(epoch)
            # self.logger.info(f"Epoch {epoch}: Train accuracy: {train_acc:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch} [val]: loss: {val_metrics['loss']:.4f}, "
                f"accuracy: {val_metrics['accuracy']:.4f}\n")
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['accuracy']
            
            self.save_checkpoint(epoch, val_metrics['accuracy'], is_best)
            
            # ========== РАННЯЯ ОСТАНОВКА ==========
            should_stop = False
            
            if early_stopping_metric == 'loss':
                # Ранняя остановка по validation loss
                current_val_loss = val_metrics['loss']
                
                if current_val_loss < best_val_loss - min_delta:
                    # Улучшение найдено
                    best_val_loss = current_val_loss
                    early_stopping_counter = 0
                    self.logger.info(f"Improvement in validation loss: {best_val_loss:.6f}")
                else:
                    # Нет улучшения
                    early_stopping_counter += 1
                    self.logger.info(
                        f"No improvement in loss for {early_stopping_counter}/{patience} epochs. "
                        f"Best val loss: {best_val_loss:.6f}"
                    )
                    
                    if early_stopping_counter >= patience:
                        self.logger.info(
                            f"Early stopping triggered! Stopping training after {epoch + 1} epochs. "
                            f"Best validation loss: {best_val_loss:.6f}, "
                            f"Best accuracy: {self.best_metric:.4f}"
                        )
                        should_stop = True
            
            elif early_stopping_metric == 'accuracy':
                # Ранняя остановка по validation accuracy
                current_val_accuracy = val_metrics['accuracy']
                
                if current_val_accuracy > best_val_accuracy + min_delta:
                    # Улучшение найдено
                    best_val_accuracy = current_val_accuracy
                    early_stopping_counter = 0
                    self.logger.info(f" Improvement in validation accuracy: {best_val_accuracy:.4f}")
                else:
                    # Нет улучшения
                    early_stopping_counter += 1
                    self.logger.info(
                        f"No improvement in accuracy for {early_stopping_counter}/{patience} epochs. "
                        f"Best val accuracy: {best_val_accuracy:.4f}"
                    )
                    
                    if early_stopping_counter >= patience:
                        self.logger.info(
                            f"Early stopping triggered! Stopping training after {epoch + 1} epochs. "
                            f"Best validation accuracy: {best_val_accuracy:.4f}, "
                            f"Best accuracy: {self.best_metric:.4f}"
                        )
                        should_stop = True
            
            if should_stop:
                break
        
        return self.best_metric