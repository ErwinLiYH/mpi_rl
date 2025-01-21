# whether to use mpi or not ------------
from mpi_rl import config
__IF_MPI = config.use_mpi

if __IF_MPI:
    from mpi4py import MPI
# --------------------------------------

from mpi_rl.mpi_utils import MPIArchitectureConfig


class BaseAlgorithm:
    def __init__(
            self,
            env,
            model_map:dict = None,
            mpi_architecture_config:MPIArchitectureConfig = None,
            ):
        self.env = env
        self.env_type = None
        self.mpi_architecture_config = mpi_architecture_config
        self.model_map = model_map

    def compute_action(self, state):
        raise NotImplementedError
    
    def compute_loss(self):
        raise NotImplementedError
    
    def compute_logit(self):
        raise NotImplementedError
    
    def pre_communication(self):
        raise NotImplementedError
    
    def post_communication(self):
        raise NotImplementedError
    
    def rollout(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError
    
    def train_iteration(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
    
    def reload_from(self):
        raise NotImplementedError
