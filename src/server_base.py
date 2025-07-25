import numpy as np
from dataset.data_utils import load_dataset  # 更新导入

class ServerBase(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.global_rounds = args.server["global_rounds"]  # 全局通信次数
        self.join_clients = None  # 某轮参加训练的用户

        self.num_clients = args.client["num_clients"]  # 用户数
        self.num_classes = args.num_classes  # 类别数
        self.distribution = args.train_config["distribution"]  # 用户数据分布

        self.clients = []  # 所有用户列表
        try:
            self.lowest_join_rate = args.server["lowest_join_rate"]  # 最低参与率
        except:
            self.lowest_join_rate = 1  # 如果没有设置最低参与率，则默认为 1

    def initialize_clients(self, ClientClass):
        # 使用新的数据集加载函数
        train_subsets, test_subsets = load_dataset(
            self.args.paths["train"], 
            self.args.paths["test"]
        )
        
        for client_id in range(self.num_clients):
            client = ClientClass(
                args=self.args,
                client_id=client_id,
                trainset=train_subsets[client_id],
                testset=test_subsets[client_id],
                statistic=self.args.statistic[client_id]
            )
            self.clients.append(client)

    def select_join_clients(self):
        minimal_join_clients = max(1, int(self.num_clients * self.lowest_join_rate))
        num_join_clients = np.random.randint(
            low=minimal_join_clients, 
            high=self.num_clients + 1
        )
        selected_clients = np.random.choice(
            self.clients,
            size=num_join_clients,
            replace=False
        ).tolist()
        return selected_clients

    def evaluate(self):
        total_samples = 0
        total_correct = 0
        for client in self.clients:
            num_correct, num_samples = client.evaluate_test()
            total_samples += num_samples
            total_correct += num_correct
            print(f'Client {client.client_id} accuracy: {100 * num_correct / num_samples:.2f}%')
        
        acc = total_correct / total_samples if total_samples > 0 else 0
        print(f'Global accuracy: {100 * acc:.2f}%')
        return acc