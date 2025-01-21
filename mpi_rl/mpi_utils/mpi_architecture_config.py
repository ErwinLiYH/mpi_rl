import logging
import pprint
from mpi4py import MPI


class MPIArchitectureConfig:
    def __init__(
            self,
                num_training_workers: int,
                num_evaluation_workers: int,
                parallel_train_eval: bool,
                centralized_training: bool,
                num_workers_per_group: int = None,
            ):
        self.global_comm = MPI.COMM_WORLD
        self.global_size = self.MPI_comm.Get_size()
        self.global_rank = self.MPI_comm.Get_rank()
        self.centralized_training = centralized_training
        self.parallel_train_eval = parallel_train_eval
        self.num_workers_per_group = num_workers_per_group
        self.num_training_workers = num_training_workers
        self.num_evaluation_workers = num_evaluation_workers

        # check the configuration
        self.check()

        # generate self.role, self.group, self.role_map
        self.gen_role_group()

    def check(self):
        if self.parallel_train_eval:
            assert self.num_evaluation_workers > 0, (
                "num_evaluation_workers should be greater than 0 when parallel_train_eval is True."
            )
            assert self.MPI_size == self.num_training_workers + self.num_evaluation_workers, (
                "MPI_size should be equal to num_training_workers + num_evaluation_workers.\n"+
                "MPI_size: {}, num_training_workers: {}, num_evaluation_workers: {}".format(
                    self.MPI_size, self.num_training_workers, self.num_evaluation_workers
            )
        )
        else:
            assert self.num_training_workers == self.MPI_size, (
                "num_training_workers should be equal to MPI_size when parallel_train_eval is False.\n"+
                "num_training_workers: {}, MPI_size: {}".format(
                    self.num_training_workers, self.MPI_size
                )
            )
            assert self.num_evaluation_workers == 0 or self.num_evaluation_workers == self.num_training_workers, (
                "num_evaluation_workers should be 0 or equal to num_training_workers when parallel_train_eval is False.\n"+
                "num_evaluation_workers: {}, num_training_workers: {}".format(
                    self.num_evaluation_workers, self.num_training_workers
                )
            )
        if self.centralized_training == False:
            assert self.num_evaluation_workers % self.num_training_workers == 0, (
                "num_evaluation_workers should be divisible by num_training_workers.\n"+
                "num_evaluation_workers: {}, num_training_workers: {}".format(
                    self.num_evaluation_workers, self.num_training_workers
                )
            )
            assert self.MPI_size % self.num_workers_per_group == 0, (
                "MPI_size should be divisible by num_workers_per_group.\n"+
                "MPI_size: {}, num_workers_per_group: {}".format(
                    self.MPI_size, self.num_workers_per_group
                )
            )
            if self.parallel_train_eval:
                assert (
                    (self.num_training_workers / (self.MPI_size/self.num_workers_per_group)) +
                    (self.num_evaluation_workers / (self.MPI_size/self.num_workers_per_group))
                ) == self.num_workers_per_group , (
                    "num of evaluation workers per group and num of training workers per group should sum up to num_workers_per_group.\n"+
                    "num_training_workers: {}, num_evaluation_workers: {}, num_workers_per_group: {}".format(
                        self.num_training_workers, self.num_evaluation_workers, self.num_workers_per_group
                    )
                )
        if self.centralized_training and self.num_workers_per_group is not None:
            if self.MPI_rank == 0:
                logging.warning("num_workers_per_group is not used in centralized training, ignored.")

    def gen_role_group(self):
        # role_map: {MPI_rank: {'role': [], 'group': 0}}
        role_map = {i:{'role': [], 'group': 0} for i in range(self.MPI_size)}
        if self.centralized_training:
            # centralized training
            role_map[0]['role'].append('main')
            if self.parallel_train_eval:
                # parallel training and evaluation
                for i in range(self.MPI_size):
                    if i < self.num_training_workers:
                        role_map[i]['role'].append('train')
                    else:
                        role_map[i]['role'].append('eval')
            else:
                # evaluation after training
                for i in range(self.MPI_size):
                    role_map[i]['role'].append('train')
                    if self.num_evaluation_workers > 0:
                        role_map[i]['role'].append('eval')
        else:
            # decentralized training
            if self.parallel_train_eval:
                # parallel training and evaluation
                for i in range(self.MPI_size):
                    in_group_index = i % self.num_workers_per_group
                    group = i // self.num_workers_per_group
                    if in_group_index == 0:
                        role_map[i]['role'].append('main')
                    if in_group_index < (self.num_training_workers / self.num_workers_per_group):
                        role_map[i]['role'].append('train')
                    else:
                        role_map[i]['role'].append('eval')
                    role_map[i]['group'] = group
            else:
                # evaluation after training
                for i in range(self.MPI_size):
                    in_group_index = i % self.num_workers_per_group
                    group = i // self.num_workers_per_group
                    if in_group_index == 0:
                        role_map[i]['role'].append('main')
                    role_map[i]['role'].append('train')
                    if self.num_evaluation_workers > 0:
                        role_map[i]['role'].append('eval')
                    role_map[i]['group'] = group

        self.role = role_map[self.MPI_rank]['role']
        self.group = role_map[self.MPI_rank]['group']
        self.role_map = role_map

    def print_role_map(self):
        if self.MPI_rank == 0:
            pprint.pprint(self.role_map)

    def logging_role(self):
        for i in range(self.MPI_size):
            if i == self.MPI_rank:
                logging.info("MPI_rank: {}, role: {}, group: {}".format(i, self.role, self.group))
            self.MPI_comm.Barrier()
