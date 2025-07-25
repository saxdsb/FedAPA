from fl_system import FederatedLearningSystem
from configs.config_loader import ExperimentConfig

def main():
    config = ExperimentConfig.load("configs/FedAPA.yaml")
    fl_system = FederatedLearningSystem(config)
    fl_system.run()

if __name__ == "__main__":
    main()