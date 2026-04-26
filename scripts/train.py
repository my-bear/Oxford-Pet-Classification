import logging
import sys
from pathlib import Path
import click
import torch
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Добавляем корень проекта в PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.models import load_model, Trainer
from src.data import load_datasets, create_dataloaders
from src.utils import setup_logging, set_seed, load_config


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to config file (YAML)')
@click.option('--data-dir', type=click.Path(exists=True),
              help='Path to data directory')
@click.option('--batch-size', '-b', type=int, default=32,
              show_default=True, help='Batch size')
@click.option('--epochs', '-e', type=int, default=100,
              show_default=True, help='Number of epochs')
@click.option('--lr', type=float, default=1e-3,
              show_default=True, help='Learning rate')
@click.option('--weight-decay', type=float, default=0.0,
              show_default=True, help='Weight decay')
@click.option('--model', '-m', 'model_name', type=str, default='resnet18',
              show_default=True, help='Model architecture')
@click.option('--num-classes', type=int, default=37,
              show_default=True, help='Number of classes')
@click.option('--seed', type=int, default=42,
              show_default=True, help='Random seed')
@click.option('--output-dir', type=click.Path(), default='../data/models',
              show_default=True, help='Output directory')
@click.option('--resume', type=click.Path(exists=True),
              help='Resume from checkpoint')
@click.option('--device', type=str, default='cuda',
              help='Device (cuda/cpu)')
@click.option('--num-workers', type=int, default=4,
              show_default=True, help='Number of data workers')
@click.option('--dry-run', is_flag=True,
              help='Dry run (test only)')
@click.option('--patience', type=int, default=10,
              show_default=True, help='Number of early stop iterations')
@click.option('--early-stopping-metric', 'early_stopping_metric', type=str, default='loss',
              show_default=True, help='Tracking metric (loss or accuracy)')

def train(
    config,
    data_dir,
    batch_size,
    epochs,
    lr,
    weight_decay,
    model_name,
    num_classes,
    seed,
    output_dir,
    resume,
    device,
    num_workers,
    dry_run,
    patience,
    early_stopping_metric
):
    """Train a model for image classification."""
    
    # 1. Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 2. Загрузка конфигурации
    # Собираем параметры из CLI
    cli_params = {
        'data_dir': data_dir,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'model_name': model_name,
        'num_classes': num_classes,
        'seed': seed,
        'output_dir': output_dir,
        'resume': resume,
        'device': device,
        'num_workers': num_workers,
        'patience': patience,
        'early_stopping_metric': early_stopping_metric
    }
    
    # Загружаем конфиг
    config_dict = load_config(config, cli_params)
    config_model = config_dict['model']

    logger.info("=" * 60)
    logger.info("Training configuration:")
    for key, value in config_dict["model"].items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # 3. Установка seed
    set_seed(config_model['seed'])
    
    # 4. Определение устройства
    device_obj = torch.device(
        config_model['device'] if torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"Using device: {device_obj}")
    
    data_config = load_config(config_dict['data_config'])

    # 5. Создание датасетов
    logger.info("Loading datasets...")
    train_dataset, val_dataset = load_datasets(data_config['root_dir'], image_size = data_config["image_size"])
    
    # 6. Создание даталоадеров
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
    )
    
    logger.info(f"Train samples: {len(train_loader)}, Val samples: {len(val_loader)}")
    
    # Dry run
    if dry_run:
        logger.info("Dry run completed. Exiting.")
        return
    
    # 7. Создание модели
    logger.info(f"Creating model: {config_model['name']}")
    model = load_model(
        model_name=config_model['name'],
        num_classes=data_config['num_classes'],
    )
    
    # 8. Оптимизатор
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_model['lr'],
        weight_decay=config_model['weight_decay'],
    )
    
    # 9. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config_model['epochs'],
        eta_min=1e-6
    )
    
    # 10. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device_obj,
        output_dir=Path(config_model['output_dir']),
        logger=logger,
    )
    
    # 11. Resume
    start_epoch:int = 0
    checkpoint_path = Path(config_dict['resume']) if config_dict.get('resume') else None
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = trainer.load_checkpoint(str(checkpoint_path))
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # 12. Обучение
    logger.info("Starting training...")
    best_metric = trainer.train(
        num_epochs=config_model['epochs'],
        start_epoch=start_epoch,
        patience = config_dict.get('patience', 10),
        early_stopping_metric = config_dict.get('early_stopping_metric', 'loss')
    )
    
    logger.info(f"Training completed. Best metric: {best_metric:.4f}")
    logger.info(f"Model saved to: {config_dict['output_dir']}")


if __name__ == '__main__':
    train()