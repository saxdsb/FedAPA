import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Dict, Type

class BaseModel(nn.Module):
    """Base class for all neural network models"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Forward method must be implemented")

class SimpleCNN(BaseModel):
    """FedAvgCNN implementation with configurable dimensions"""
    def __init__(self, in_channels=1, num_classes=10, dim=1024):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class LeNet5(BaseModel):
    """LeNet-5 architecture implementation"""
    def __init__(self, in_channels, num_classes, dim):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class HeadMergedModel(BaseModel):
    """Model combining base feature extractor and classification head"""
    def __init__(self, base: nn.Module, head: nn.Module):
        super().__init__()
        self.base = base
        self.head = head
    
    def forward(self, x: Tensor) -> Tensor:
        features = self.base(x)
        return self.head(features)

class ResNetBlock(nn.Module):
    """Base class for ResNet building blocks"""
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_bn: bool = True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Common components
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # BN configurable
        self.bn_layer = norm_layer if use_bn else nn.Identity

class BasicResBlock(ResNetBlock):
    """Basic ResNet block with two 3x3 convolutions"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        stride = kwargs['stride']
        norm_layer = kwargs.get('norm_layer', nn.BatchNorm2d)
        use_bn = kwargs.get('use_bn', True)
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = self.bn_layer(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = self.bn_layer(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)

class BottleneckResBlock(ResNetBlock):
    """Bottleneck ResNet block with 1x1-3x3-1x1 convolutions"""
    expansion: int = 4
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        stride = kwargs['stride']
        groups = kwargs.get('groups', 1)
        base_width = kwargs.get('base_width', 64)
        dilation = kwargs.get('dilation', 1)
        norm_layer = kwargs.get('norm_layer', nn.BatchNorm2d)
        use_bn = kwargs.get('use_bn', True)
        
        width = int(out_channels * (base_width / 64.0)) * groups
        
        self.conv1 = nn.Conv2d(
            in_channels, width, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = self.bn_layer(width)
        
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, 
            padding=dilation, groups=groups, dilation=dilation, bias=False
        )
        self.bn2 = self.bn_layer(width)
        
        self.conv3 = nn.Conv2d(
            width, out_channels * self.expansion, 
            kernel_size=1, stride=1, bias=False
        )
        self.bn3 = self.bn_layer(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)

class ResNetModel(BaseModel):
    """Flexible ResNet implementation supporting various architectures"""
    def __init__(
        self,
        block_type: Type[ResNetBlock],
        layer_config: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        use_bn: bool = True,
        groups: int = 1,
        base_width: int = 64,
        dilation_config: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bn_blocks: int = 4,
        feature_channels: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.in_channels = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = base_width
        
        # Default dilation configuration
        if dilation_config is None:
            dilation_config = [False, False, False]
        
        # Initial layers
        self.init_conv = nn.Conv2d(
            in_channels, self.in_channels, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.init_bn = norm_layer(self.in_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build residual layers
        self.res_layers = nn.ModuleList()
        self._build_res_layers(
            block_type, layer_config, feature_channels, 
            dilation_config, use_bn, bn_blocks
        )
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            feature_channels[-1] * block_type.expansion, 
            num_classes
        )
        
        # Weight initialization
        self._init_weights()

    def _build_res_layers(
        self,
        block_type: Type[ResNetBlock],
        layer_config: List[int],
        feature_channels: List[int],
        dilation_config: List[bool],
        use_bn: bool,
        bn_blocks: int
    ):
        """Construct residual layers based on configuration"""
        layers = []
        # First layer
        layers.extend(self._make_res_layer(
            block_type, feature_channels[0], layer_config[0], 
            stride=1, use_bn=use_bn and (bn_blocks > 0)
        ))
        
        # Subsequent layers
        for idx, block_count in enumerate(layer_config[1:], start=1):
            layers.extend(self._make_res_layer(
                block_type, feature_channels[idx], block_count,
                stride=2, dilate=dilation_config[idx-1],
                use_bn=use_bn and (idx < bn_blocks)
            ))
        
        self.res_layers = nn.Sequential(*layers)

    def _make_res_layer(
        self,
        block_type: Type[ResNetBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dilate: bool = False,
        use_bn: bool = True
    ) -> List[nn.Module]:
        """Create a residual layer with multiple blocks"""
        norm_layer = self._norm_layer
        downsample = None
        prev_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # Downsample if needed
        if stride != 1 or self.in_channels != out_channels * block_type.expansion:
            downsample_layers = [
                nn.Conv2d(
                    self.in_channels, out_channels * block_type.expansion,
                    kernel_size=1, stride=stride, bias=False
                )
            ]
            if use_bn:
                downsample_layers.append(norm_layer(out_channels * block_type.expansion))
            downsample = nn.Sequential(*downsample_layers)
        
        # Build blocks
        blocks = []
        blocks.append(block_type(
            self.in_channels, out_channels, stride, downsample, 
            self.groups, self.base_width, prev_dilation, 
            norm_layer, use_bn
        ))
        
        self.in_channels = out_channels * block_type.expansion
        for _ in range(1, num_blocks):
            blocks.append(block_type(
                self.in_channels, out_channels, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer, use_bn=use_bn
            ))
        
        return blocks

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.res_layers(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ResNet Architecture Factory Functions
def create_resnet4(**kwargs) -> ResNetModel:
    return ResNetModel(BasicResBlock, [1], **kwargs)

def create_resnet6(**kwargs) -> ResNetModel:
    return ResNetModel(BasicResBlock, [1, 1], **kwargs)

def create_resnet8(**kwargs) -> ResNetModel:
    return ResNetModel(BasicResBlock, [1, 1, 1], **kwargs)

def create_resnet10(**kwargs) -> ResNetModel:
    return ResNetModel(BasicResBlock, [1, 1, 1, 1], **kwargs)

def create_resnet18(**kwargs) -> ResNetModel:
    return ResNetModel(BasicResBlock, [2, 2, 2, 2], **kwargs)

def create_resnet34(**kwargs) -> ResNetModel:
    return ResNetModel(BasicResBlock, [3, 4, 6, 3], **kwargs)

def create_resnet50(**kwargs) -> ResNetModel:
    return ResNetModel(BottleneckResBlock, [3, 4, 6, 3], **kwargs)

def create_resnet101(**kwargs) -> ResNetModel:
    return ResNetModel(BottleneckResBlock, [3, 4, 23, 3], **kwargs)

def create_resnet152(**kwargs) -> ResNetModel:
    return ResNetModel(BottleneckResBlock, [3, 8, 36, 3], **kwargs)

# Aliases for compatibility
ResNet18 = create_resnet18
resnet4 = create_resnet4