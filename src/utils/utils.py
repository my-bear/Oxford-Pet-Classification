"""
Utility functions.
"""

import logging
import random
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, cli_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML and override with CLI parameters.
    
    Args:
        config_path: Path to YAML config file
        cli_params: Dictionary of CLI parameters (only non-None values override)
    
    Returns:
        Merged configuration dictionary
    """
    config = {}
    
    # Load from YAML
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with CLI parameters
    if cli_params:
        for key, value in cli_params.items():
            if value is not None:
                config[key] = value
    
    return config