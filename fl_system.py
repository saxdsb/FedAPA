import time
import numpy as np
import torch
from typing import Dict, List, Tuple
from utils.result_saver import ResultSaver
from dataset.dataset_manager import DatasetManager
from models.model_factory import ModelFactory
from models.model_factory import BaseHeadMerge
from sever.server_factory import ServerFactory
from torch import nn
import copy

class FederatedLearningSystem:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.server["device"])
        self.result_saver = ResultSaver(config)
        self.dataset_manager = DatasetManager(config)
        self.model_factory = ModelFactory(config)
    
    def run(self):
        results = {
            "experiments": [],
            "metrics": {
                "best_acc": [],
                "upload": [],
                "download": [],
                "total_time": [],
                "iteration_time": []
            }
        }
        
        for repeat in range(self.config.num_repeat_times):
            print(f"=== Experiment Repetition {repeat + 1}/{self.config.num_repeat_times} ===")
            
            # 准备数据集
            train_config, statistics = self.dataset_manager.prepare_dataset()
            
            # 准备模型
            model = self.model_factory.create_model()
            head = copy.deepcopy(model.fc)
            model.fc = nn.Identity()


            model_new = BaseHeadMerge(base=model, head=head).to(self.device)

            # 创建服务器
            server = ServerFactory.create_server(
                algorithm=self.config.algorithm,
                config=self.config,
                model=model_new,
                device=self.device
            )


            # 将数据集信息添加到服务器参数中
            if hasattr(server, 'args') and hasattr(server.args, 'train_config'):
                server.args.train_config = train_config
                server.args.statistic = statistics



            
            # 训练模型
            experiment_result = self.run_experiment(server, train_config)
            
            # 保存结果
            results["experiments"].append(experiment_result)
            results["metrics"]["best_acc"].append(experiment_result["best_acc"])
            results["metrics"]["upload"].append(experiment_result["upload"])
            results["metrics"]["download"].append(experiment_result["download"])
            results["metrics"]["total_time"].append(experiment_result["total_time"])
            results["metrics"]["iteration_time"].append(experiment_result["iteration_time"])
        
        # 计算并保存最终结果
        self.result_saver.save(results, statistics)

    def run_experiment(self, server, train_config) -> Dict:
        start_time = time.time()
        
        # 训练模型
        accuracy_history, upload_size, download_size = server.train()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            "config": train_config,
            "accuracy_history": accuracy_history,
            "best_acc": max(accuracy_history.values()),
            "upload": upload_size,
            "download": download_size,
            "total_time": total_time,
            "iteration_time": total_time / self.config.server["global_rounds"]
        }