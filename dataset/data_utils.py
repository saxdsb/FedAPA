import os
import numpy as np
import ujson
import torch
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import transforms, datasets

class KronoDroidDataset(torch.utils.data.Dataset):
    def __init__(self, root="/root/autodl-fs/kronoDroid", data_type='train'):
        self.data_path = f"{root}/{data_type}_time_junyihua.npy"
        data = np.load(self.data_path, allow_pickle=True).item()
        self.x = data['X']
        self.targets = data['Y'].astype('int64')
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = np.stack((self.x[index],) * 3, axis=-1).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        return img, self.targets[index]

    def __len__(self):
        return len(self.x)

def validate_dataset_config(args, config_path):
    if not args.debug or not os.path.exists(config_path):
        return False
    
    with open(config_path, 'r') as f:
        config = ujson.load(f)
    
    train_cfg = config["train_config"]
    if train_cfg["distribution"] != args.distribution:
        return False
    
    client_num = args.client["num_clients"]
    if args.distribution == "dirichlet":
        return ("alpha" in train_cfg and 
                train_cfg["alpha"] == args.alpha and
                train_cfg["client"]["num_clients"] == client_num)
    
    elif args.distribution == "non_balanced":
        return (train_cfg["client"]["num_clients"] == client_num and
                train_cfg["client"]["num_classes_per_client"] == 
                args.client["num_classes_per_client"])
    
    elif args.distribution == "iid":
        return train_cfg["client"]["num_clients"] == client_num
    
    return False

def save_dataset(train_config, statistics, train_subsets, test_subsets, config_path, train_path, test_path):
    config = {"train_config": train_config, "statistic": statistics}
    with open(config_path, 'w') as f:
        ujson.dump(config, f, indent=2)
    torch.save(train_subsets, train_path)
    torch.save(test_subsets, test_path)

def load_dataset(train_path, test_path):
    return torch.load(train_path), torch.load(test_path)

def get_client_data(dataset, client_id, is_train=True):
    data_dir = "dataset/Cifar100/" + ('train/' if is_train else 'test/')
    file_path = f"{data_dir}{client_id}.npz"
    
    with open(file_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    
    features = torch.tensor(data['x'], dtype=torch.float32)
    labels = torch.tensor(data['y'], dtype=torch.int64)
    return [(x, y) for x, y in zip(features, labels)]

def create_data_partition(num_clients, num_classes, distribution_type, alpha=None, classes_per_client=None):
    if distribution_type == "dirichlet":
        return np.random.dirichlet([alpha] * num_clients, size=num_classes)
    
    client_class_counts = [classes_per_client] * num_clients
    clients_per_class = num_clients * classes_per_client // num_classes
    partition = np.zeros((num_classes, num_clients))
    
    for cls_idx in range(num_classes):
        selected_clients = []
        for client_id in range(num_clients):
            if client_class_counts[client_id] > 0 and len(selected_clients) < clients_per_class:
                selected_clients.append(client_id)
                client_class_counts[client_id] -= 1
        
        proportions = np.ones(clients_per_class) if distribution_type == "iid" else \
                     np.random.uniform(0.3, 0.7, clients_per_class)
        proportions /= proportions.sum()
        
        for client_id in selected_clients:
            partition[cls_idx, client_id] = proportions.pop()
    
    return partition

def distribute_dataset(dataset, num_classes, num_clients, partition_matrix, benign_alone=False):
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == cls)[0] for cls in range(num_classes)]
    class_counts = [len(indices) for indices in class_indices]
    
    client_indices = defaultdict(list)
    client_stats = [{} for _ in range(num_clients)]
    
    # 处理恶意样本分布
    start_cls = 1 if benign_alone else 0
    for cls in range(start_cls, num_classes):
        cls_idx = cls if not benign_alone else cls - 1
        clients = np.nonzero(partition_matrix[cls_idx])[0]
        
        for client in clients:
            sample_count = int(partition_matrix[cls_idx, client] * class_counts[cls])
            if sample_count < 1: 
                continue
                
            client_indices[client].extend(class_indices[cls][:sample_count])
            class_indices[cls] = class_indices[cls][sample_count:]
            client_stats[client][str(cls)] = sample_count
    
    # 单独处理良性样本
    if benign_alone:
        total_malware = sum(len(v) for v in client_indices.values())
        for client in range(num_clients):
            malware_count = len(client_indices[client])
            benign_count = int((malware_count / total_malware) * class_counts[0])
            selected = np.random.choice(class_indices[0], benign_count, replace=False)
            client_indices[client].extend(selected)
            client_stats[client]["0"] = benign_count
    
    return [Subset(dataset, client_indices[i]) for i in range(num_clients)], client_stats

def get_dataset_paths(dataset_name, distribution):
    base_path = f"dataset/rawdata/{dataset_name}/{distribution}"
    return {
        "config": f"{base_path}/config.json",
        "train": f"{base_path}/trainsets.pth",
        "test": f"{base_path}/testsets.pth"
    }

def load_raw_dataset(dataset_name, data_paths):
    common_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == "mnist":
        DatasetClass = datasets.MNIST
        norm = transforms.Normalize([0.5], [0.5])
    elif dataset_name == "fmnist":
        DatasetClass = datasets.FashionMNIST
        norm = transforms.Normalize([0.5], [0.5])
    elif dataset_name == "cifar10":
        DatasetClass = datasets.CIFAR10
    elif dataset_name == "cifar100":
        DatasetClass = datasets.CIFAR100
    elif dataset_name == "kronoDroid":
        return KronoDroidDataset(data_type='train'), KronoDroidDataset(data_type='test')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    transform = common_transforms if dataset_name in ["cifar10", "cifar100"] else \
               transforms.Compose([transforms.ToTensor(), norm])
    
    train_set = DatasetClass(root=data_paths["rawdata"], train=True, download=True, transform=transform)
    test_set = DatasetClass(root=data_paths["rawdata"], train=False, download=True, transform=transform)
    return train_set, test_set

def prepare_federated_dataset(args):
    paths = get_dataset_paths(args.dataset, args.distribution)
    os.makedirs(os.path.dirname(paths["config"]), exist_ok=True)
    
    if validate_dataset_config(args, paths["config"]):
        with open(paths["config"], 'r') as f:
            config = ujson.load(f)
        return config["train_config"], config["statistic"]
    
    print("Generating new federated dataset...")
    train_set, test_set = load_raw_dataset(args.dataset, {
        "rawdata": f"dataset/rawdata/{args.dataset}"
    })
    
    num_clients = args.client["num_clients"]
    num_classes = len(np.unique(train_set.targets))
    partition_classes = num_classes - 1 if args.benign_alone else num_classes
    
    while True:
        partition = create_data_partition(
            num_clients=num_clients,
            num_classes=partition_classes,
            distribution_type=args.distribution,
            alpha=args.alpha,
            classes_per_client=args.client.get("num_classes_per_client", 10)
        )
        
        train_subsets, stats = distribute_dataset(
            train_set, num_classes, num_clients, partition, args.benign_alone
        )
        test_subsets, _ = distribute_dataset(
            test_set, num_classes, num_clients, partition, args.benign_alone
        )
        
        if all(len(subset) > 128 for subset in train_subsets) and \
           all(len(subset) > 2 for subset in test_subsets):
            break
        print("Regenerating due to insufficient samples...")
    
    config = {
        "dataset": args.dataset,
        "distribution": args.distribution,
        "alpha": args.alpha,
        "server": args.server,
        "client": args.client
    }
    save_dataset(config, stats, train_subsets, test_subsets, 
                paths["config"], paths["train"], paths["test"])
    
    return config, stats