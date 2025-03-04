from mpi_rl.mpi_rl import algorithm
from mpi_rl.mpi_utils.mpi_architecture_config import MPIArchitectureConfig

# ----------------------------------------------------------------------------------------------

# **example 1**:

# centralized training with parallel train and eval
# run by mpirun -n 12 python test_architecture_config.py
# How to calculate the archetecture configuration?
# total workers = num_training_workers + num_evaluation_workers = 6 + 6 = 12, so we need to run `mpirun -n 12`
# num_workers_per_group = None and centralized_training = True, so we have 1 group
# so the archetecture configuration output should be:
# Global Rank: 0, Group Index: 0, Group Rank: 0, Role: ['main', 'training']
# Global Rank: 1, Group Index: 0, Group Rank: 1, Role: ['training']
# Global Rank: 2, Group Index: 0, Group Rank: 2, Role: ['training']
# Global Rank: 3, Group Index: 0, Group Rank: 3, Role: ['training']
# Global Rank: 4, Group Index: 0, Group Rank: 4, Role: ['training']
# Global Rank: 5, Group Index: 0, Group Rank: 5, Role: ['training']
# Global Rank: 6, Group Index: 0, Group Rank: 6, Role: ['evaluation']
# Global Rank: 7, Group Index: 0, Group Rank: 7, Role: ['evaluation']
# Global Rank: 8, Group Index: 0, Group Rank: 8, Role: ['evaluation']
# Global Rank: 9, Group Index: 0, Group Rank: 9, Role: ['evaluation']
# Global Rank: 10, Group Index: 0, Group Rank: 10, Role: ['evaluation']
# Global Rank: 11, Group Index: 0, Group Rank: 11, Role: ['evaluation']

arch_config = MPIArchitectureConfig(
    num_training_workers = 6,
    num_evaluation_workers = 6,
    parallel_train_eval = True,
    centralized_training = True,
    num_workers_per_group = None,
)

arch_config.mpi_print_role()

# ----------------------------------------------------------------------------------------------

# **example 2**:

# decentralized training with parallel train and eval
# run by mpirun -n 12 python test_architecture_config.py
# How to calculate the archetecture configuration?
# total workers = num_training_workers + num_evaluation_workers = 6 + 6 = 12, so we need to run `mpirun -n 12`
# num_workers_per_group = 4, so we have total works / num_workers_per_group = 12 / 4 = 3 groups
# num_training_workers = 6, so we have 6 / 3 = 2 training workers per group
# num_evaluation_workers = 6, so we have 6 / 3 = 2 evaluation workers per group
# then the archetecture configuration output should be:
# Global Rank: 0, Group Index: 0, Group Rank: 0, Role: ['main', 'training']
# Global Rank: 1, Group Index: 0, Group Rank: 1, Role: ['training']
# Global Rank: 2, Group Index: 0, Group Rank: 2, Role: ['evaluation']
# Global Rank: 3, Group Index: 0, Group Rank: 3, Role: ['evaluation']
# Global Rank: 4, Group Index: 1, Group Rank: 0, Role: ['main', 'training']
# Global Rank: 5, Group Index: 1, Group Rank: 1, Role: ['training']
# Global Rank: 6, Group Index: 1, Group Rank: 2, Role: ['evaluation']
# Global Rank: 7, Group Index: 1, Group Rank: 3, Role: ['evaluation']
# Global Rank: 8, Group Index: 2, Group Rank: 0, Role: ['main', 'training']
# Global Rank: 9, Group Index: 2, Group Rank: 1, Role: ['training']
# Global Rank: 10, Group Index: 2, Group Rank: 2, Role: ['evaluation']
# Global Rank: 11, Group Index: 2, Group Rank: 3, Role: ['evaluation']

# arch_config = MPIArchitectureConfig(
#     num_training_workers = 6,
#     num_evaluation_workers = 6,
#     parallel_train_eval = True,
#     centralized_training = False,
#     num_workers_per_group = 4,
# )

# arch_config.mpi_print_role()