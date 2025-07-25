import torch.optim
from torch import nn
from torch.utils.data import DataLoader


class ClientBase:
    def __init__(self, args, client_id, trainset, testset, statistic):
        self.args = args
        self.algorithm = args.algorithm  # 算法名称
        self.device = args.device

        self.client_id = client_id  # 用户 id
        self.num_clients = args.client["num_clients"]  # 用户数
        self.num_classes = args.num_classes  # 类别数
        self.trainset = trainset  # 训练集
        self.testset = testset  # 测试集
        self.sample_size = len(self.trainset)  # 训练集样本数
        self.statistic = statistic  # 各类标签的样本量统计

        self.batch_size = args.client["batch_size"]  # 本地训练批次大小
        self.local_epochs = args.client["local_epochs"]  # 本地训练轮数
        self.learning_rate = args.client["learning_rate"]  # 本地学习率
        self.loss = nn.CrossEntropyLoss()  # 损失函数

    def get_train_loader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def get_test_loader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, drop_last=False, shuffle=True)

    def evaluate_test(self):
        test_loader = self.get_test_loader()


        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.inference_mode():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                total_correct += torch.sum(torch.argmax(output, dim=1) == y).item()
                total_samples += len(y)
        return total_correct, total_samples
