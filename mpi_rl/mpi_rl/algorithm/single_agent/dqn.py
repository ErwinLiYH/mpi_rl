from mpi_rl.algorithm.base_algorithm import BaseAlgorithm, BaseAlgorithmConfig
from dataclasses import dataclass


@dataclass
class DQNConfig(BaseAlgorithmConfig):
    pass


class DQN(BaseAlgorithm):
    def __init__(
            self,
            ):
        pass

    def check_config(self):
        pass

    def compute_action(self, obs):
        pass

    def train(self, episodes):
        pass