from src.FedAPA.server_FedAPA import ServerFedAPA

import os
import torch
from typing import Dict

class ServerAdapter:
    """完整的服务器适配器，包含所有 server_base 需要的属性"""
    def __init__(self, server_class, algorithm, config, model, device):
        # 创建与原始 args 兼容的对象
        self.args = type('Args', (), {})()
        
        # 基本配置
        self.args.algorithm = algorithm
        self.args.dataset = config.dataset
        self.args.distribution = config.distribution
        self.args.alpha = config.alpha
        self.args.modelName = config.model_name
        self.args.benignAlone = config.benign_alone
        self.args.device = device
        self.args.model = model
        self.args.num_repeat_times = config.num_repeat_times
        
        # 服务器配置
        self.args.server = config.server
        self.args.client = config.client
        
        # 路径配置 (server_base.py 需要)
        self.args.paths = self._generate_paths()
        
        # 数据集配置 (server_base.py 需要)
        self.args.train_config = {
            "dataset": config.dataset,
            "distribution": config.distribution,
            "alpha": config.alpha,
            "server": config.server,
            "client": config.client,
        }
        self.args.statistic = []  # 将在数据集生成后填充
        
        # 类别数和特征维度 (server_base.py 需要)
        self.args.num_classes = self._get_num_classes()
        self.args.feature_dim = self._get_feature_dim()

        # 保存服务器类和初始化状态
        self.server_class = server_class
        self.server = None  # 延迟初始化
        self.initialized = False
        
    
    def _generate_paths(self) -> Dict[str, str]:
        """生成数据集路径，模拟原始 generate_paths 函数"""
        rawdata_path = os.path.join("dataset", "rawdata", self.args.dataset)
        data_path = os.path.join(rawdata_path, self.args.distribution)
        return {
            "rawdata": rawdata_path,
            "data": data_path,
            "config": os.path.join(data_path, "config.json"),
            "train": os.path.join(data_path, "trainsets.pth"),
            "test": os.path.join(data_path, "testsets.pth")
        }
    
    def _get_num_classes(self) -> int:
        """根据数据集确定类别数"""
        dataset_classes = {
            "mnist": 10, "fmnist": 10, "cifar10": 10,
            "cifar100": 100, "kronoDroid": 13
        }
        return dataset_classes.get(self.args.dataset, 10)
    
    def _get_feature_dim(self) -> int:
        """根据数据集确定类别数"""
        dataset_classes = {
            "mnist": 84, "fmnist": 84, "cifar10": 84,
            "cifar100": 84, "kronoDroid": 84
        }
        return dataset_classes.get(self.args.dataset, 84)
    
    def initialize_server(self):
        """在设置好所有参数后初始化服务器"""
        if not self.initialized:
            self.server = self.server_class(self.args)
            self.initialized = True

    def train(self):
        """调用原始服务器的 train 方法"""
        if not self.initialized:
            self.initialize_server()
        return self.server.train()
    


    def __getattr__(self, name):
        """将所有未定义的方法转发给原始服务器"""
        if self.initialized and hasattr(self.server, name):
            return getattr(self.server, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class ServerFactory:
    # 服务器类映射
    SERVER_CLASSES = {
        "FedAPA": ServerFedAPA
    }
    
    @staticmethod
    def create_server(algorithm: str, config, model, device):
        if algorithm not in ServerFactory.SERVER_CLASSES:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        server_class = ServerFactory.SERVER_CLASSES[algorithm]
        return ServerAdapter(server_class, algorithm, config, model, device)