import logging
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
        self.global_size = self.global_comm.Get_size()
        self.global_rank = self.global_comm.Get_rank()
        self.centralized_training = centralized_training
        self.parallel_train_eval = parallel_train_eval
        self.num_workers_per_group = num_workers_per_group
        self.num_training_workers = num_training_workers
        self.num_evaluation_workers = num_evaluation_workers

        # whether evaluation is needed
        self.with_evaluation = True
        if self.parallel_train_eval == False and self.num_evaluation_workers == 0:
            self.with_evaluation = False

        # check the configuration
        self.__check()

        # in centralized training, all workers are in the same group
        if self.num_workers_per_group is None:
            self.num_workers_per_group = self.global_size

        # generate sub-communicators
        self.group = self.global_rank // self.num_workers_per_group
        self.group_comm = self.global_comm.Split(self.group, self.global_rank)
        self.group_size = self.group_comm.Get_size()
        self.group_rank = self.group_comm.Get_rank()

        # generate role
        self.role = []  # "main", "training", "evaluation"
        if self.group_rank == 0:
            self.role.append("main")
        num_training_workers_per_group = (   # after checking, this is always an integer
            self.num_training_workers / 
            (self.global_size/self.num_workers_per_group)
            )
        if parallel_train_eval:
            if self.group_rank < num_training_workers_per_group:
                self.role.append("training")
            else:
                self.role.append("evaluation")
        else:
            self.role.append("training")
            if num_evaluation_workers > 0:
                self.role.append("evaluation")

        # split the group into training and evaluation groups
        if self.parallel_train_eval:
            train_eval_color = int("training" in self.role)   # 1 if training role, then evaluation role is 0
            self.group_role_comm = self.group_comm.Split(train_eval_color, self.group_rank)
        else:
            self.group_role_comm = self.group_comm

    def __check(self):
        # check type
        assert isinstance(self.num_training_workers, int), (
            "num_training_workers should be an integer."
        )
        assert isinstance(self.num_evaluation_workers, int), (
            "num_evaluation_workers should be an integer."
        )
        assert isinstance(self.parallel_train_eval, bool), (
            "parallel_train_eval should be a boolean."
        )
        assert isinstance(self.num_workers_per_group, int) or self.num_workers_per_group is None, (
            "num_workers_per_group should be an integer or None."
        )

        # check configuration interaction
        if self.parallel_train_eval:
            assert self.num_evaluation_workers > 0, (
                "num_evaluation_workers should be greater than 0 when parallel_train_eval is True."
            )
            assert self.global_size == self.num_training_workers + self.num_evaluation_workers, (
                "MPI_size should be equal to num_training_workers + num_evaluation_workers.\n"+
                "MPI_size: {}, num_training_workers: {}, num_evaluation_workers: {}".format(
                    self.global_size, self.num_training_workers, self.num_evaluation_workers
            )
        )
        else:
            assert self.num_training_workers == self.global_size, (
                "num_training_workers should be equal to MPI_size when parallel_train_eval is False.\n"+
                "num_training_workers: {}, MPI_size: {}".format(
                    self.num_training_workers, self.global_size
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
            assert self.num_workers_per_group is not None, (
                "num_workers_per_group should be specified when centralized_training is False (decentralized mode)."
            )
            assert self.global_size % self.num_workers_per_group == 0, (
                "MPI_size should be divisible by num_workers_per_group.\n"+
                "MPI_size: {}, num_workers_per_group: {}".format(
                    self.global_size, self.num_workers_per_group
                )
            )
            if self.parallel_train_eval:
                assert self.num_training_workers % (self.global_size/self.num_workers_per_group)==0, (
                    "num_training_workers should be divisible by number of groups (MPI_size/num_workers_per_group).\n"+
                    "num_training_workers: {}, MPI_size: {}, num_workers_per_group: {}".format(
                        self.num_training_workers, self.global_size, self.num_workers_per_group
                    )
                )
                assert self.num_evaluation_workers % (self.global_size/self.num_workers_per_group)==0, (
                    "num_evaluation_workers should be divisible by number of groups (MPI_size/num_workers_per_group).\n"+
                    "num_evaluation_workers: {}, MPI_size: {}, num_workers_per_group: {}".format(
                        self.num_evaluation_workers, self.global_size, self.num_workers_per_group
                    )
                )
                assert (
                    (self.num_training_workers / (self.global_size/self.num_workers_per_group)) +
                    (self.num_evaluation_workers / (self.global_size/self.num_workers_per_group))
                ) == self.num_workers_per_group , (
                    "num of evaluation workers per group and num of training workers per group should sum up to num_workers_per_group.\n"+
                    "num_training_workers: {}, num_evaluation_workers: {}, num_workers_per_group: {}".format(
                        self.num_training_workers, self.num_evaluation_workers, self.num_workers_per_group
                    )
                )
        if self.centralized_training:
            assert self.num_workers_per_group is None, (
                "num_workers_per_group should be None when centralized_training is True."
            )

        if self.centralized_training == False:
            assert self.num_workers_per_group is not None, (
                "num_workers_per_group should be specified when centralized_training is False."
            )

    def mpi_print_role(self):
        local_output = "Global Rank: {}, Group Index: {}, Group Rank: {}, Role: {}".format(
            self.global_rank, self.group, self.group_rank, self.role
        )

        all_outputs = self.global_comm.gather(local_output, root=0)

        if self.global_rank == 0:
            print("<<<<<<<<<<<<<<<<<< architecture configuration <<<<<<<<<<<<<<<<<<", flush=True)
            for output in all_outputs:
                print(output, flush=True)
            print(">>>>>>>>>>>>>>>>>> architecture configuration >>>>>>>>>>>>>>>>>>", flush=True)
