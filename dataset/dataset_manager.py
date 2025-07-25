import os
import ujson
from typing import Dict, Tuple
from dataset.data_utils import prepare_federated_dataset, get_dataset_paths

class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.paths = get_dataset_paths(config.dataset, config.distribution)
        os.makedirs(os.path.dirname(self.paths["config"]), exist_ok=True)
    
    def prepare_dataset(self) -> Tuple[Dict, Dict]:
        """准备联邦学习数据集"""
        if self._is_config_valid():
            return self._load_existing_config()
        return prepare_federated_dataset(self.config)
    
    def _is_config_valid(self) -> bool:
        """检查现有配置是否有效"""
        from dataset.data_utils import validate_dataset_config
        return validate_dataset_config(self.config, self.paths["config"])
    
    def _load_existing_config(self) -> Tuple[Dict, Dict]:
        """加载现有的数据集配置"""
        with open(self.paths["config"], "r") as f:
            config_data = ujson.load(f)
        return config_data["train_config"], config_data["statistic"]