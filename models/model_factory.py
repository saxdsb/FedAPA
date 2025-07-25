import torch
import torch.nn as nn
from typing import Dict
from models.models import SimpleCNN, LeNet5, create_resnet18, create_resnet4


class BaseHeadMerge(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


class Net(nn.Module):
    def __init__(self, in_channels, num_classes, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=6,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 120),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(84, num_classes)
        # self.fc = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class ModelFactory:
    """Factory for creating neural network models based on configuration"""
    def __init__(self, config):
        self.config = config
        self.dataset_params = self._get_dataset_params()
    
    def _get_dataset_params(self) -> Dict:
        """Get dataset-specific parameters"""
        return {
            "mnist": {"in_channels": 1, "dim": 16*4*4, "num_classes": 10},
            "fmnist": {"in_channels": 1, "dim": 16*4*4, "num_classes": 10},
            "cifar10": {"in_channels": 3, "dim": 16*5*5, "num_classes": 10},
            "cifar100": {"in_channels": 3, "dim": 16*5*5, "num_classes": 100},
            "kronoDroid": {"in_channels": 3, "dim": 16*1*1, "num_classes": 13}
        }.get(self.config.dataset, {})
    
    def create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_name = self.config.model_name
        params = self.dataset_params
        
        

        if model_name =='LeNet5':
           return( Net(in_channels=params["in_channels"], num_classes=params["num_classes"], dim=params["dim"]) ) # 基础分类模型，对这个模型后续训练过程不能有任何改动

        elif model_name == "ResNet18":
            model = create_resnet18(num_classes=params["num_classes"])
            
            # 精确修改第一层卷积+BN
            model.init_conv = nn.Conv2d(
                in_channels=params["in_channels"],
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            # 同步更新BN层
            model.init_bn = nn.BatchNorm2d(64) if model.use_bn else nn.Identity()
            
            # 必须修改全连接层（原始行为）
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, params["num_classes"])
            
            return model
        
        elif model_name == "resnet4":
            return create_resnet4(
                num_classes=params["num_classes"],
                use_bn=True  # 假设原始实现使用BN
            )
        
        elif model_name == "FedAvgCNN":
            return SimpleCNN(
                in_features=params["in_channels"],  # 使用新参数名
                num_classes=params["num_classes"],
                dim=1600
            )