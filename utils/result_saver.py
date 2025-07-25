import os
import ujson
import numpy as np
from utils.tools import readable_size

class ResultSaver:
    def __init__(self, config):
        self.config = config
    
    def save(self, results, statistics):
        # 计算汇总统计
        metrics = results["metrics"]
        summary = {
            "mean_accuracy": np.mean(metrics["best_acc"]).item(),
            "std_accuracy": np.var(metrics["best_acc"]).item(),
            "mean_upload": readable_size(np.mean(metrics["upload"])),
            "mean_download": readable_size(np.mean(metrics["download"])),
            "mean_total_time": np.mean(metrics["total_time"]).item(),
            "mean_iteration_time": np.mean(metrics["iteration_time"]).item(),
            "model": self.config.model_name
        }
        
        # 准备完整结果
        full_result = {
            "experiments": results["experiments"],
            "summary": summary,
            "statistics": statistics,
            "config": {
                "dataset": self.config.dataset,
                "distribution": self.config.distribution,
                "algorithm": self.config.algorithm,
                "num_clients": self.config.client["num_clients"],
                "model": self.config.model_name,
                "alpha": self.config.alpha
            }
        }
        
        # 保存结果
        result_dir = f"./results/{self.config.dataset}/{self.config.distribution}"
        os.makedirs(result_dir, exist_ok=True)
        
        file_prefix = f"{self.config.prefix}-" if self.config.prefix else ""
        alpha = self.config.alpha
        
        filename = f"{file_prefix}{self.config.algorithm}-{self.config.client['num_clients']}-{alpha}.json"
        
        with open(os.path.join(result_dir, filename), "w") as f:
            ujson.dump(full_result, f, indent=2)
        
        print("Results saved successfully.")