"""
Modern CNN for image classification.

Architecture features:
- Conv-BN-GELU blocks (modern activation)
- Residual connections
- SE blocks for channel attention
- Adaptive pooling for flexible input sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ConvConfig:
    """Configuration for convolution block."""
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: Optional[int] = None
    use_bn: bool = True
    use_se: bool = True  # Add SE block after convolution
    dropout_rate: float = 0.0

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Adaptively recalibrates channel-wise feature responses.
    Modern approach: uses GELU.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Explicit GAP layer
        # AdaptiveAvgPool2d(1) works with ANY input spatial size
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)

        # Squeeze + Excitation with modern activation
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),  # Modern: GELU instead of ReLU
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # Keep sigmoid for scaling
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Scaled tensor of same shape (B, C, H, W)
        """
        batch, channels, height, width = x.shape
        
        # Step 1: Squeeze (GAP)
        # (B, C, H, W) -> (B, C, 1, 1)
        y = self.gap(x)
        
        # Step 2: Flatten for Linear layers
        # (B, C, 1, 1) -> (B, C)
        y = y.view(batch, channels)
        
        # Step 3: Excitation (learn channel weights)
        # (B, C) -> (B, C)
        y = self.fc(y)
        
        # Step 4: Reshape for broadcasting
        # (B, C) -> (B, C, 1, 1)
        y = y.view(batch, channels, 1, 1)
        
        # Step 5: Scale original features
        # Broadcast multiplication: each channel multiplied by its weight
        return x * y

class ConvBlock(nn.Module):
    """
    Modern convolution block:
    1. Conv2d
    2. BatchNorm (optional)
    3. GELU activation (modern, smoother than ReLU)
    4. SE block (optional, for channel attention)
    5. Dropout (optional)
    
    Uses Kaiming initialization for GELU compatibility.
    """
    
    def __init__(self, config: ConvConfig):
        super().__init__()
        
        # Calculate padding if not provided
        if config.padding is None:
            config.padding = config.kernel_size // 2  # Same padding
        
        # Когда in_channels=3, ядро имеет форму:
        # (out_channels, 3, kernel_size, kernel_size)
        # Main convolution
        self.conv = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=not config.use_bn  # No bias if using BN
        )

        # I use batch normalization after each convolution 
        # and before activation, without bias in the convolution layers.
        # Batch normalization 
        # nn.Identity() — это слой, который возвращает вход без изменений. 
        self.bn = nn.BatchNorm2d(config.out_channels) if config.use_bn else nn.Identity()
        
        # Activation: GELU
        self.activation = nn.ReLU()
        
        # Optional SE block for channel attention
        self.se = SEBlock(config.out_channels) if config.use_se else nn.Identity()
        
        # Optional dropout
        self.dropout = nn.Dropout2d(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for GELU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Заполняет веса случайными числами из нормального распределения
                # Дисперсия подбирается так, чтобы сохранять дисперсию сигнала на каждом слое
                # В сверточных слоях важно сохранять дисперсию на выходе
                # fan_out = out_channels * kernel_size * kernel_size
                # Это помогает стабилизировать градиенты при обратном распространении
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Изначально BatchNorm не должен менять входные данные и добавлять смещение
                # γ = 1 означает, что нормализованные данные проходят без масштабирования
                # β = 0 означает, что нормализованные данные не сдвигаются
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Two conv blocks with optional projection
    for dimension matching.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
    ):
        super().__init__()
        
        # Main path: two convolutions
        self.conv1 = ConvBlock(ConvConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            use_se=False  # SE after both convs
        ))
        
        self.conv2 = ConvBlock(ConvConfig(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_se=False
        ))
        
        # SE after residual sum
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Skip connection with projection if dimensions change
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = ConvBlock(ConvConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                use_bn=True,
                use_se=False
            ))
        
        # Activation after residual sum
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity  # Residual connection #  + или __add__ 
        out = self.activation(out)
        return out


class ModernCNN(nn.Module):
    """
    Modern CNN for image classification.
    
    Architecture:
    - Stem: Conv-BN-GELU (no aggressive downsampling)
    - Stage 1: 2 residual blocks, 64 channels
    - Stage 2: 2 residual blocks, 128 channels
    - Stage 3: 2 residual blocks, 256 channels
    - Head: Adaptive pooling → Classifier
    
    Total depth: ~15-20 layers (not too deep)
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        input_channels: int = 3,
        base_channels: int = 64,
        dropout_rate: float = 0.3,
        use_se: bool = True,
    ):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels
            dropout_rate: Dropout rate for classifier
            use_se: Whether to use SE blocks
        """
        super().__init__()
        
        # Store config for debugging
        self.config = {
            'num_classes': num_classes,
            'base_channels': base_channels,
            'dropout_rate': dropout_rate,
            'use_se': use_se
        }
        
        # ========== STEM ==========
        # Modern stem: 7x7 with stride 2 is too aggressive for small images
        # Using 3x3 with stride 1 for better detail preservation
        self.stem = ConvBlock(ConvConfig(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            use_se=False
        ))
        
        # ========== STAGE 1 ==========
        # 64 channels, preserve spatial dimensions
        self.stage1 = self._make_stage(
            in_channels=base_channels,
            out_channels=base_channels,
            num_blocks=2,
            stride=1,
            use_se=use_se
        )
        
        # ========== STAGE 2 ==========
        # 128 channels, downsample by factor 2
        self.stage2 = self._make_stage(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            num_blocks=2,
            stride=2,
            use_se=use_se
        )
        
        # ========== STAGE 3 ==========
        # 256 channels, downsample by factor 2
        self.stage3 = self._make_stage(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            num_blocks=2,
            stride=2,
            use_se=use_se
        )
        
        # ========== HEAD ==========
        # Adaptive pooling: works with any input size
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 4, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_se: bool,
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        blocks = []
        
        # First block may downsample
        blocks.append(ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            use_se=use_se,
        ))
        
        # Remaining blocks keep dimensions
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                use_se=use_se,
            ))
        
        return nn.Sequential(*blocks)
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Debug: print shape if needed
        # print(f"Input: {x.shape}")
        
        x = self.stem(x)
        # print(f"After stem: {x.shape}")
        
        x = self.stage1(x)
        # print(f"After stage1: {x.shape}")
        
        x = self.stage2(x)
        # print(f"After stage2: {x.shape}")
        
        x = self.stage3(x)
        # print(f"After stage3: {x.shape}")
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.classifier(x)
        
        return x
    

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract intermediate features (useful for debugging/visualization).
        
        Returns:
            List of feature maps from each stage
        """
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        x = self.stage1(x)
        features.append(x)
        
        x = self.stage2(x)
        features.append(x)
        
        x = self.stage3(x)
        features.append(x)
        
        return features
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return self.config
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self, input_size: Tuple[int, int, int] = (3, 256, 256)):
        """Generate model summary."""
        device = next(self.parameters()).device
        x = torch.randn(1, *input_size).to(device)
        
        output = self.forward(x)
        
        summary_str = f"""
        ========================================
        Model: ModernCNN
        ========================================
        Input shape:  {input_size}
        Output shape: {tuple(output.shape)}
        
        Total parameters: {self.count_parameters():,}
        Trainable parameters: {self.count_parameters():,}
        
        Architecture:
        - Stem: Conv3x3-BN-ReLU
        - Stage 1: 2x Residual blocks, {self.config['base_channels']} channels
        - Stage 2: 2x Residual blocks, {self.config['base_channels'] * 2} channels
        - Stage 3: 4x Residual blocks, {self.config['base_channels'] * 4} channels
        - Classifier: AdaptiveAvgPool + Dropout + Linear
        
        Features:
        - ReLU activation
        - SE blocks for channel attention
        - Residual connections
        - Kaiming initialization
        ========================================
        """
        return print(summary_str)
