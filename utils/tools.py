import os
import random

import numpy as np
import torch
import ujson


def set_random_seed(seed):
    """
    设置随机种子
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_result(dataset, distribution, algorithm, num_clients, result, prefix):
    result_dir = f"./results/{dataset}/{distribution}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_prefix = f"{prefix}-" if prefix else ""
    alpha = result["config"]["alpha"]
    if algorithm=='FedCPCL':
        lambada = result["config"]["client"]["lambda"]
        belta = result["config"]["client"]["belt"]
        result_path = os.path.join(result_dir, file_prefix + f"{algorithm}-{num_clients}-{alpha}-{lambada}--{belta}.json")
    else:
        result_path = os.path.join(result_dir, file_prefix + f"{algorithm}-{num_clients}-{alpha}.json")
    with open(result_path, "w") as f:
        ujson.dump(result, f, indent=2)


def readable_size(num_bytes: int):
    """Convert bytes to a readable format."""
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"