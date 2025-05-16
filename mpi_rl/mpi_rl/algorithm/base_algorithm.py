# whether to use mpi or not ------------
from mpi_rl.config import get_use_mpi
if get_use_mpi():
    from mpi4py import MPI
    from mpi_rl.mpi_utils.mpi_architecture_config import MPIArchitectureConfig
# --------------------------------------

from mpi_rl import config
from mpi_rl.utils.replay_buffer import Episode, Transaction, ReplayBuffer
from mpi_rl.utils.config_helper import RolloutConfig, ModelConfig
from mpi_rl.utils.metric_group import MetricGroups
from mpi_rl.utils import RemoteObjectProxy
from mpi_rl.utils import get_env_type, flat_double_layer_list, DummyLogger

from typing import Literal, List
from contextlib import contextmanager
from dataclasses import dataclass

import logging
import copy
import os


# typing aliases
LogPath = str

class EnvHolder:
    """
    A class to hold the environment.
    """
    def __init__(self, env):
        self.env = env

@dataclass
class BaseAlgorithmConfig:
    pass

class BaseAlgorithm:
    """
    Base class for all algorithms.
    """
    def __init__(
            self,
            train_env_creator: callable = None,
            train_rollout_config:RolloutConfig = None,
            eval_env_creator: callable = None,
            eval_rollout_config:RolloutConfig = None,
            hard_reset_env:bool = True,
            separate_eval_proc:bool = True,
            replay_buffer:ReplayBuffer = None,
            model_config:ModelConfig = None,
            algorithm_config:BaseAlgorithmConfig = None,
            eval_interval:int = 1,
            mpi_architecture_config:MPIArchitectureConfig = None,
            log_paths:List[LogPath] = ["print"],
            multi_agent:bool = False,
            transaction_process_fn:callable = None
            ):
        self.train_env_creator = train_env_creator
        self.eval_env_creator = eval_env_creator
        self.train_rollout_config = train_rollout_config
        self.eval_rollout_config = eval_rollout_config
        self.hard_reset_env = hard_reset_env
        self.separate_eval_proc = separate_eval_proc
        self.mpi_architecture_config = mpi_architecture_config
        self.model_config = model_config
        self.algorithm_config = algorithm_config
        self.eval_interval = eval_interval
        self.log_paths = log_paths
        self.transaction_process_fn = transaction_process_fn

        self.operating_episode = Episode()
        self.operating_transaction = Transaction()
        self.replay_buffer = replay_buffer

        self.multi_agent = multi_agent

        if self.mpi_architecture_config is not None:
            self.name = f"g{self.mpi_architecture_config.group}_r{self.mpi_architecture_config.group_rank}_{'-'.join(self.mpi_architecture_config.role)}"
        else:
            self.name = "main"

        # initialize logger
        self.logger = self._init_logger()

        # to maintain the rollout config state and env
        self._current_rollout_config = None
        self._env = None
        self._env_creator = None
        self.status = None

        # progress info
        self.progress_info = {
            "current_iteration": 0,
            "num_local_episodes": 0,
            "num_local_step": 0,
            "num_group_episodes": 0,
            "num_group_step": 0,
        }

        # metrics (only in 'main' rank in MPI mode)
        self.training_metrics = MetricGroups()
        self.evaluation_metrics = MetricGroups()

        # initialize the environment
        self.__init_env()

    def _init_logger(self):
        # if log_paths is empty, return DummyLogger
        if not self.log_paths:
            return DummyLogger()

        # build the logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # 设置全局日志级别
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        for path in self.log_paths:
            if path == "print":
                handler = logging.StreamHandler()
            elif isinstance(path, str):
                # check if the path is a valid file path
                dir_path = os.path.dirname(path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                handler = logging.FileHandler(path)
            else:
                raise ValueError("elements in log_paths should be str of path or 'print'.")
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def __init_env(self):
        """
        Initialize the training and evaluation environment.
        After this function, 
            1. self.train_env
            2. self.eval_env
            3. self.env_type
        should be initialized.
        this function will check whether the evaluation env is needed according to the architecture config in MPI mode,
        or check whether the evaluation env creator is provided in non-MPI mode.
        if the evaluation env is not needed, self.eval_env will be None.

        Don't using the initialized env directly, the best practice is,
        using `with self.train_rollout_state():` or `with self.eval_rollout_state()` followed by self._env.
        For example:
        with self.train_rollout_state():
            self._env.env.reset()

        with self.eval_rollout_state():
            self._env.env.reset()
        """
        self.train_env = None
        self.eval_env = None

        def env_creator_remote_maker(env_creator:callable):
            return lambda algo_name: RemoteObjectProxy(env_creator, {"algo_name": algo_name})
        # check if it requested the evaluation env from config
        if self.mpi_architecture_config is not None:       # in mpi mode, check the architecture config
            if "training" in self.mpi_architecture_config.role:  # only init the training env in ranks with training role
                if self.train_env_creator is not None:
                    if self.separate_eval_proc:
                        self.train_env = EnvHolder(RemoteObjectProxy(self.train_env_creator, {"algo_name": self.name}))
                        self.train_env_creator = env_creator_remote_maker(self.train_env_creator)
                    else:
                        self.train_env = EnvHolder(self.train_env_creator(algo_name=self.name))
                else:
                    raise ValueError("train_env_creator should be provided.")

            if "evaluation" in self.mpi_architecture_config.role:  # only init the evaluation env in ranks with evaluation role
                if self.eval_env_creator is not None:
                    if self.separate_eval_proc:
                        self.eval_env = EnvHolder(RemoteObjectProxy(self.eval_env_creator, {"algo_name": self.name}))
                        self.eval_env_creator = env_creator_remote_maker(self.eval_env_creator)
                    else:    
                        self.eval_env = EnvHolder(self.eval_env_creator(algo_name=self.name))
                else:
                    raise ValueError("eval_env_creator should be provided.")
        else: # in non-mpi mode
            if self.train_env_creator is not None:
                if self.separate_eval_proc:
                    self.train_env = EnvHolder(RemoteObjectProxy(self.train_env_creator, {"algo_name": self.name}))
                    self.train_env_creator = env_creator_remote_maker(self.train_env_creator)
                else:
                    self.train_env = EnvHolder(self.train_env_creator(algo_name=self.name))
            if self.eval_env_creator is not None:
                if self.separate_eval_proc:
                    self.eval_env = EnvHolder(RemoteObjectProxy(self.eval_env_creator, {"algo_name": self.name}))
                    self.eval_env_creator = env_creator_remote_maker(self.eval_env_creator)
                else:
                    self.eval_env = EnvHolder(self.eval_env_creator(algo_name=self.name))
            if self.eval_env is None and self.train_env is None:
                raise ValueError("train_env_creator or eval_env_creator should be provided.")

        if self.train_env is not None:
            self.env_type = get_env_type(self.train_env.env)
        else:
            self.env_type = get_env_type(self.eval_env.env)

    def reset_env(self):
        """
        Reset the environment, must be called within the train_state() or eval_state() context.
        """
        if self.hard_reset_env:
            if self.separate_eval_proc:
                self._env.env.close()
            self._env.env = self._env_creator(algo_name=self.name)
        
        return self._env.env.reset()
    
    def rollout(
            self,
            ) -> list[Episode]:
        """
        Rollout the environment to collect data.
        """
        collected_steps = 0
        collected_episodes = 0
        episodes = []
        while True:  # episode loop
            obs, info = self.reset_env()
            if self.env_type in ["pzp", "gym"]:
                while True:  # step loop
                    action = self.compute_action(obs)
                    self.logger.info(f"Act - Step {collected_steps} - {self.status} - Action: {action}")
                    next_obs, reward, termination, truncation, info = self._env.env.step(action)
                    if self.transaction_process_fn is not None:
                        obs, action, reward, next_obs, termination, truncation, info = \
                            self.transaction_process_fn(obs, action, reward, next_obs, termination, truncation, info)
                    collected_steps += 1
                    self.progress_info["num_local_step"] += 1
                    self.operating_transaction.update(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        termination=termination,
                        truncation=truncation,
                        info=info,
                        )
                    if self.multi_agent:
                        # for multi-agent environment, record the agents
                        self.operating_transaction.update(agents=action.keys())
                        # try to get the state, not all multi-agent env has state()
                        try:
                            self.operating_transaction.update(state=self._env.env.state())
                        except Exception as e:
                            pass
                    self.operating_episode.insert_transaction(self.operating_transaction)
                    self.operating_transaction.clear()    # refresh the operation transaction
                    obs = next_obs

                    # early stop if the collected steps reach the target
                    if self._current_rollout_config.complete_episode==False and collected_steps==self._current_rollout_config.num_of_step:
                        break

                    # check if the episode is terminated
                    if self.env_type == "pzp" or (self.multi_agent and self.env_type=="gym"):
                        # for pettingzoo parallel environment and multi-agent gym environment
                        if self._current_rollout_config.pzp_episode_cut_method=="strict_pzp":
                            termi = bool(self._env.env.agents)
                        elif self._current_rollout_config.pzp_episode_cut_method=="any":
                            termi = any(termination.values()) or any(truncation.values())
                        elif self._current_rollout_config.pzp_episode_cut_method=="all":
                            termi = all(termination.values()) or all(truncation.values())
                    else:
                        # for single-agent gym environment
                        termi = termination or truncation
                    if termi:
                        break
            elif self.env_type == "pz_aec":
                raise NotImplementedError("Rollout for pettingzoo AEC environment is not implemented yet.")
            else:
                raise ValueError("env_type should be one of ['pzp', 'gym', 'pz_aec']")

            self.on_end_of_episode()
            
            episodes.append(copy.deepcopy(self.operating_episode))
            self.operating_episode = Episode()   # refresh the operation episode
            collected_episodes += 1
            self.progress_info["num_local_episodes"] += 1

            if self._current_rollout_config.count_by == "step":
                if collected_steps >= self._current_rollout_config.num_of_step:
                    break
            elif self._current_rollout_config.count_by == "episode":
                if collected_episodes == self._current_rollout_config.num_of_episode:
                    break

        # update global progress info
        if self.mpi_architecture_config is not None:
            self.mpi_architecture_config.group_role_comm.allreduce(collected_steps, op=MPI.SUM)
            self.progress_info["num_group_step"] += collected_steps
            self.mpi_architecture_config.group_role_comm.allreduce(collected_episodes, op=MPI.SUM)
            self.progress_info["num_group_episodes"] += collected_episodes
        else:
            # same as local
            self.progress_info["num_group_episodes"] += collected_episodes
            self.progress_info["num_group_step"] += collected_steps

        return episodes
    
    def compute_action(self, obs):
        # Write your own action computation logic here.
        # You may need self.model_config to access models.
        raise NotImplementedError

    def train(self, episodes:list[Episode]):
        # Write your own training logic here.
        # You may need self.model_config to access models.
        #
        # Only be called in 'main' rank (aka. group rank == 0) in MPI mode
        raise NotImplementedError
    
    def evaluate(self, episodes:list[Episode]):
        # Write your own evaluation logic here.
        # Generally, you should calculate the metrics based on collected episodes.
        # And update the evaluation metrics to self.evaluation_metrics.
        #
        # Only be called in 'main' rank (aka. group rank 0) in MPI mode
        #
        # Example of add a fake metric of each episode in each iteration:
        self.evaluation_metrics.add("current_iteration", self.progress_info["current_iteration"],  # add the current iteration as X
                                    episode_rewards=[1 for e in episodes],    # add the fake metrics as Y
                                    # other metrics, can have multiple Ys)
                                    )
    
    def train_iteration(self):
        if self.mpi_architecture_config is not None:
            if self.mpi_architecture_config.group_rank==0:
                self.logger.info("-"*25+f"Start iteration {self.progress_info['current_iteration']}"+"-"*25)
            self.train_iteration_mpi()
        else:
            self.logger.info("-"*25+f"Start iteration {self.progress_info['current_iteration']}"+"-"*25)
            self.train_iteration_non_mpi()
    
    def train_iteration_mpi(self):
        self.mpi_architecture_config.group_comm.Barrier()
        if  "training" in self.mpi_architecture_config.role:
            self.logger.info(f"Start training, begin rollout")
            with self.train_state():
                # each rank rollout and collect episodes
                local_episodes = self.rollout()
                # The Barrier() wait for all ranks to finish the rollout, actually be useful
                # when parallel_train_eval is True, because in train(), the model will be updated.
                # When parallel_train_eval is False, barrier is not needed, but it won't affect the result,
                # because later gather() will block all ranks too.
                self.mpi_architecture_config.group_comm.Barrier()
                # collect episodes from all `training` rank in same group
                episodes = self.mpi_architecture_config.group_role_comm.gather(local_episodes, root=0)
                # episodes in form of [[episodes from rank 0], [episodes from rank 1], ...], so flatten it
                if self.mpi_architecture_config.group_rank == 0:
                    flatted_episodes = flat_double_layer_list(episodes)
                    self.logger.info(f"Training on {len(flatted_episodes)} episodes")
                    self.train(flatted_episodes)
                self.post_train()
        if "evaluation" in self.mpi_architecture_config.role and self.progress_info["current_iteration"]%self.eval_interval==0:
            self.logger.info(f"Start evaluation, begin rollout")
            with self.eval_state():
                # each rank rollout and collect episodes
                local_episodes = self.rollout()
                # The Barrier() wait for all ranks to finish the rollout, actually be useful
                # when parallel_train_eval is True, because in train(), the model will be updated.
                # When parallel_train_eval is False, barrier is not needed, but it won't affect the result,
                # because later gather() will block all ranks too.
                self.mpi_architecture_config.group_comm.Barrier()
                # collect episodes from all `evaluation` rank in same group
                episodes = self.mpi_architecture_config.group_role_comm.gather(local_episodes, root=0)
                # episodes in form of [[episodes from rank 0], [episodes from rank 1], ...], so flatten it
                if self.mpi_architecture_config.group_rank == 0:
                    flatted_episodes = flat_double_layer_list(episodes)
                    self.logger.info(f"Evaluating on {len(flatted_episodes)} episodes")
                    self.evaluate(flatted_episodes)
        self.progress_info["current_iteration"] += 1
        self.mpi_architecture_config.group_comm.Barrier()

    def post_train(self):
        """
        Run after the train(). To synchronize the models.
        """
        pass

    def train_iteration_non_mpi(self):
        if self.train_env is not None:
            with self.train_state():
                episodes = self.rollout()
                self.train(episodes)
                self.post_train()
        if self.eval_env is not None and self.progress_info["current_iteration"]%self.eval_interval==0:
            with self.eval_state():
                episodes = self.rollout()
                self.evaluate(episodes)
    
    def save(self):
        raise NotImplementedError
    
    def reload_from(self):
        raise NotImplementedError
    
    @contextmanager
    def train_state(self):
        """
        Enter the train mode, set self._exploration to True.
        with self.train_state():
            # self._exploration is True in this with block
        """
        # manage the rollout config state
        original_rollout_config = self._current_rollout_config
        self._current_rollout_config = self.train_rollout_config

        # manage the env state
        original_env = self._env
        self._env = self.train_env
        
        # manage the status
        original_status = self.status
        self.status = "train"

        # manager the env creator
        original_env_creator = self._env_creator
        self._env_creator = self.train_env_creator

        try:
            yield
        finally:
            self._current_rollout_config = original_rollout_config
            self._env = original_env
            self.status = original_status
            self._env_creator = original_env_creator

    @contextmanager
    def eval_state(self):
        """
        Enter the eval mode, set self._exploration to False.
        with self.eval_state():
            # self._exploration is False in this with block
        """
        # manage the rollout config state
        original_rollout_config = self._current_rollout_config
        self._current_rollout_config = self.eval_rollout_config

        # manage the env state
        original_env = self._env
        self._env = self.eval_env

        # manage the status
        original_status = self.status
        self.status = "eval"

        # manager the env creator
        original_env_creator = self._env_creator
        self._env_creator = self.eval_env_creator

        try:
            yield
        finally:
            self._current_rollout_config = original_rollout_config
            self._env = original_env
            self.status = original_status
            self._env_creator = original_env_creator

    def on_end_of_episode(self):
        """
        Called at the end of each episode.
        You can implement your own logic here.

        self.operating_episode is the current episode.
        """
        pass
