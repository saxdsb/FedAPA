import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    dataset: str
    algorithm: str
    model_name: str
    distribution: str
    alpha: float
    num_repeat_times: int
    debug: bool
    prefix: str
    benign_alone: bool
    server: Dict[str, Any]
    client: Dict[str, Any]

    @classmethod
    def load(cls, config_path: str) -> "ExperimentConfig":
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        
        experiment = config_data["experiment"]
        return cls(
            dataset=experiment["dataset"],
            algorithm=experiment["algorithm"],
            model_name=experiment["model_name"],
            distribution=experiment["distribution"],
            alpha=experiment["alpha"],
            num_repeat_times=experiment["num_repeat_times"],
            debug=experiment["debug"],
            prefix=experiment["prefix"],
            benign_alone=experiment["benign_alone"],
            server=config_data["server"],
            client=config_data["client"]
        )