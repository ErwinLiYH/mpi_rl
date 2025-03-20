# whether to use mpi or not ------------
from mpi_rl import config
__IF_MPI = config.use_mpi

if __IF_MPI:
    from mpi4py import MPI
    from mpi_rl.mpi_utils.mpi_architecture_config import MPIArchitectureConfig
else:
    # typing alias, MPIArchitectureConfig should be None if not using mpi
    MPIArchitectureConfig = None
# --------------------------------------

from mpi_rl.mpi_rl.algorithm.base_algorithm import BaseAlgorithm, BaseAlgorithmConfig
from dataclasses import dataclass

from typing import Any, Callable, Optional

from mpi_rl.utils.replay_buffer import Episode, ReplayBuffer
from mpi_rl.utils.config_helper import RolloutConfig

from mpi_rl.mpi_rl.algorithm.llm.prompt_template import ActTemplate, ReflexTemplate
from mpi_rl.utils import remove_first_line

import json
import os

# Typing Aliases
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import Path_Without_NameT, MessagesT, \
    ContentT, Reasoning_ContentT, Num_Input_TokensT, Num_Output_TokensT, Input_SequenceT, Invoker_outputT

@dataclass
class LLMConfig(BaseAlgorithmConfig):
    system_description: str
    act_prompt:str
    reflex_prompt:str
    act_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    reflex_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    metrics_statistics_fn:Callable[[Episode], Any]
    system_prompt: str = ""
    system_description_in_user: bool = True
    eval_invoke_fn: Optional[Callable[[MessagesT], Invoker_outputT]] = None
    reflex_model_tokens_counter_fn: Callable[[Input_SequenceT],int] = None
    act_model_tokens_counter_fn: Callable[[Input_SequenceT],int] = None
    reflex_max_tokens_input: int = None
    act_max_tokens_input: int = None
    transaction_process_fn: Callable = lambda x: x     # A function to process the transaction before convert to json string,
                                                       # do 1.select the useful data, 2. convert the data type to basic type.
                                                       # Need convert numpy array or pytorch tensor to list to save space
                                                       # because numpy array and pytorch tensor can't using `IntFloatEncoder`,
                                                       # then just `str()` for them, which will waste token count.
                                                       # e.g. str(np.array([1.1111111, 2.0, 3.0]) => "[1.1111111 2.        3.       ]"
                                                       # if using `IntFloatEncoder`, it will be "[1.11, 2, 3]"
    obs_process_fn: Callable = lambda x: x     # A function to process the observation before convert to json string, similar to transaction_process_fn
    round_digits: int = 2
    record_conversation: None|Path_Without_NameT = None
    max_history:int = 50
    reflex_batch:int = 5
    reflex_specific_transaction_process_fn: Optional[Callable] = None

    def __post_init__(self):
        # model_tokens_counter_fn and max_tokens_input should be None together or not None together
        assert (self.act_model_tokens_counter_fn is None and self.act_max_tokens_input is None) or \
                (self.act_model_tokens_counter_fn is not None and self.act_max_tokens_input is not None), \
                "act_model_tokens_counter_fn and act_max_tokens_input should be None together or not None together."
        
        assert (self.reflex_model_tokens_counter_fn is None and self.reflex_max_tokens_input is None) or \
                (self.reflex_model_tokens_counter_fn is not None and self.reflex_max_tokens_input is not None), \
                "reflex_model_tokens_counter_fn and reflex_max_tokens_input should be None together or not None together."
        
        if self.record_conversation is not None:
            os.makedirs(self.record_conversation, exist_ok=True)

class LLM(BaseAlgorithm):
    """
    LLM algorithm is a customized `BaseAlgorithm` for large language model.
    It is designed to use large language model to solve the task in pure Natural Language with chain-of-reflection.
    """
    def __init__(
            self,
            train_env_creator:Callable,
            train_rollout_config:RolloutConfig,
            eval_env_creator: Callable = None,
            eval_rollout_config:RolloutConfig = None,
            replay_buffer:ReplayBuffer = None,
            algorithm_config:LLMConfig = None,
            eval_interval:int = 1,
            mpi_architecture_config:MPIArchitectureConfig = None,
            log_path:str = "print",
            multi_agent:bool = False,
            transaction_process_fn:Callable = None,
            ):
        super().__init__(
            train_env_creator = train_env_creator,
            train_rollout_config = train_rollout_config,
            eval_env_creator = eval_env_creator,
            eval_rollout_config = eval_rollout_config,
            replay_buffer = replay_buffer,
            model_config = None,
            algorithm_config = algorithm_config,
            eval_interval = eval_interval,
            mpi_architecture_config = mpi_architecture_config,
            log_path = log_path,
            multi_agent = multi_agent,
            transaction_process_fn = transaction_process_fn
            )
        
        self.check_config()

        self.policy_conclusion = None

        if self.algorithm_config.system_description_in_user:
            self.act_template = ActTemplate(algorithm_config.system_description, algorithm_config.act_prompt)
            self.reflex_template = ReflexTemplate(algorithm_config.system_description, algorithm_config.reflex_prompt)
            self.system_prompt = self.algorithm_config.system_prompt
        else:
            self.act_template = ActTemplate("", algorithm_config.act_prompt)
            self.reflex_template = ReflexTemplate("", algorithm_config.reflex_prompt)
            if self.algorithm_config.system_prompt != "" and isinstance(self.algorithm_config.system_prompt, str):
                self.system_prompt = self.algorithm_config.system_prompt + "\n\n" + self.algorithm_config.system_description
            else:
                self.system_prompt = self.algorithm_config.system_description

    def check_config(self):
        assert isinstance(self.algorithm_config, LLMConfig), "algorithm_config should be LLMConfig."

    def compute_action(self, obs):
        episode_str = self.episode_to_str(
            self.operating_episode[-self.algorithm_config.max_history:],
            separators=(',', ':')
        ) if len(self.operating_episode) > 0 else 'None'
        
        partial_episode = "下面是按顺序排列的历史步骤:\n(倒数第一行是前一步历史, 倒数第二行是钱两步历史, 以此类推)\n\n"\
                        + episode_str \
                        + "\n\n"
        
        obs_str = json.dumps(self.algorithm_config.obs_process_fn(obs), cls=IntFloatEncoder, 
                             round_digits=self.algorithm_config.round_digits, 
                             separators=(',', ':'), sort_keys=True, ensure_ascii=False, default=str)
        
        obs = "下面是此步的观测:\n\n"\
            + obs_str \
            + "\n\n"

        input_sequence = self.act_template.format(
            obs=obs,
            partial_episode=partial_episode
        )

        # check if need to reduce episode history
        expected_input_token_count = \
            self.algorithm_config.act_model_tokens_counter_fn(input_sequence+self.system_prompt)
        self.logger.debug(f"Expected Input token count: {expected_input_token_count}")
        if expected_input_token_count > self.algorithm_config.act_max_tokens_input:
            self.logger.info(f"Expected Input token count: {expected_input_token_count} \
                             > act_max_tokens_input: {self.algorithm_config.act_max_tokens_input}.")
            to_reduce = expected_input_token_count - self.algorithm_config.act_max_tokens_input
            encoded_episode_str = self.encoder(episode_str)
            # reduce previous replay buffer
            encoded_episode_str = encoded_episode_str[to_reduce+10:]   # plus 10 for safety
            encoded_episode_str = self.encoder.decode(encoded_episode_str)
            encoded_episode_str = remove_first_line(encoded_episode_str)  # remove first uncompleted line
            partial_episode = "下面是按时间顺序排列的历史步骤:\n(倒数第一行是前一步历史, 倒数第二行是前两步历史, 以此类推)\n\n"\
                            + episode_str \
                            + "\n\n"
            input_sequence = self.act_template.format(
                obs=obs,
                partial_episode=partial_episode
            )
        
        self.logger.debug(input_sequence)

        content, llm_action, reasoning_content, input_len, output_len = \
            self.invoke_api(input_sequence)

        # record
        if self.algorithm_config.record_conversation:
            with open(f"{os.path.join(self.algorithm_config.record_conversation, self.name)}_compute_act_{self.status}_history.txt", 'a') as f:
                conversation = self.make_conversation(
                    input_sequence, reasoning_content, content, expected_input_token_count, input_len, output_len
                )
                f.write(conversation)
        
        if len(self.operating_episode) > 0:
            self.operating_episode[-1]["per_step_reflection"] = llm_action["prev_step_reflection"]
        self.operating_transaction.update({"analysis": llm_action["analysis"]})

        return llm_action["action"]

    def train(self, episodes: list[Episode]):
        self.temp_episodes_llm_algo_specific = episodes
        # This algorithm need to store the episodes to replay_buffer in all ranks in MPI mode,
        # so we store the episodes in a temporary variable and store them to replay_buffer in post_train
        # which will run on all ranks.

        for idx, ep in enumerate(episodes):
            self.reflex(ep, idx)

        self.finetune_actor()

    def reflex(self, episode: Episode, idx:int):
        metrics_dict = self.algorithm_config.metrics_statistics_fn(episode)

        episode_metrics = "这是上面这个完整episode的性能指标统计值: \n\n" \
                        + json.dumps(metrics_dict) + "\n\n"
        
        def window_indices(length, n):
            i = 0
            while i + n <= length:
                yield (i, i + n)
                i += n
            if i < length:
                yield (i, length)

        for win_start, win_end in window_indices(len(episode), self.algorithm_config.reflex_batch):

            ep_start = episode[0:win_start]
            ep_focus = episode[win_start:win_end]
            ep_end = episode[win_end:]

            process_fn = self.algorithm_config.reflex_specific_transaction_process_fn \
                if self.algorithm_config.reflex_specific_transaction_process_fn is not None \
                else self.algorithm_config.transaction_process_fn

            episode_start_str = self.episode_to_str(ep_start, process_fn=process_fn, \
                separators=(',', ':')) + "\n" if len(ep_start) > 0 else ''
            episode_focus_str = "<<<\n" + self.episode_to_str(ep_focus, process_fn=process_fn, \
                separators=(',', ':')) + "\n>>>\n"
            episode_end_str = self.episode_to_str(ep_end, process_fn=process_fn, \
                separators=(',', ':')) + "\n" if len(ep_end) > 0 else ''
            episode_str = episode_start_str + episode_focus_str + episode_end_str
            the_episode = "下面是按时间顺序排列的一个episode中的完整步骤, 这一步主要关注用<<<< >>>包裹的片段Z:\n\n"\
                        + episode_str \
                        + "\n\n"
        
            input_sequence = self.reflex_template.format(
                episode=the_episode,
                episode_metrics=episode_metrics
            )

            content, llm_reflex, reasoning_content, input_len, output_len = self.invoke_api(input_sequence, reflex=True)
            
            for idx, ref in enumerate(llm_reflex):
                episode[win_start+idx]["transaction"]["reflex"] = ref

            if self.algorithm_config.record_conversation:
                it = self.progress_info["current_iteration"]
                with open(
                    f"{os.path.join(self.algorithm_config.record_conversation, self.name)}_reflex_iter{it}_ep{idx}_history.txt",
                    'a'
                ) as f:
                    conversation = self.make_conversation(
                        input_sequence, reasoning_content, content, "None", input_len, output_len
                    )
                    f.write(conversation)

    def finetune_actor(self):
        pass

    def post_train(self):
        """
        Broadcast the policy conclusion and collected episodes to all the ranks.
        All ranks store collected episodes to replay_buffer.
        """
        if self.mpi_architecture_config is None:
            for e in self.temp_episodes_llm_algo_specific:
                self.replay_buffer.insert_episode(e)
        else:
            self.temp_episodes_llm_algo_specific = \
                self.mpi_architecture_config.group_comm.broadcast(self.temp_episodes_llm_algo_specific, root=0)
            for e in self.temp_episodes_llm_algo_specific:
                self.replay_buffer.insert_episode(e)
            self.policy_conclusion = \
                self.mpi_architecture_config.group_comm.broadcast(self.policy_conclusion, root=0)

    # ----------------- LLM algorithm specific functions -----------------

    def make_conversation(self, input_sequence, reasoning_content, content, expected_input_token_count, input_len, output_len):
        conversation = "User:\n\n" \
                    + input_sequence + "\n" \
                    + "-"*50 + "\n"
        if reasoning_content is not None:
            conversation = conversation \
                            + "Reasoning:\n\n" \
                            + reasoning_content + "\n" \
                            + "-"*50 + "\n"
        conversation = conversation \
                        + "Assistant:\n\n" \
                        + content + "\n" \
                        + "-"*50 + "\n" \
                        + f"< expected input tokens: {expected_input_token_count}, input tokens: {input_len}, output tokens: {output_len} >\n" \
                        + "<"*25 + ">"*25 + "\n"
        return conversation
    
    def replay_buffer_to_str(self, replay_buffer:ReplayBuffer, process_fn:Callable=None, **kwargs):
        """
        Concert the replay_buffer to a compressed string (to save space) for LLM.
        """
        str_list = []
        episode_index = 0
        for i in range(len(replay_buffer)):
            if process_fn is None:
                transaction = self.algorithm_config.transaction_process_fn(replay_buffer[i]["transaction"])
            else:
                transaction = process_fn(replay_buffer[i]["transaction"])
            index = replay_buffer[i]["episode_index"]
            if index == episode_index+1:
                str_list.append("++++++++++\n")
                episode_index += 1
            json_str = json.dumps(transaction, cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, \
                                  sort_keys=True, ensure_ascii=False, default=str, **kwargs)
            str_list.append(json_str+"\n")
        return "\n".join(str_list)
    
    def episode_to_str(self, episode:Episode, process_fn:Callable=None, **kwargs):
        """
        Concert the episode to a compressed string (to save space) for LLM.
        """
        str_list = []
        for i in range(len(episode)):
            if process_fn is None:
                transaction = self.algorithm_config.transaction_process_fn(episode[i]["transaction"])
            else:
                transaction = process_fn(episode[i]["transaction"])
            json_str = json.dumps(transaction, cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, \
                                  sort_keys=True, ensure_ascii=False, default=str, **kwargs)
            str_list.append(json_str)
        return "\n".join(str_list)

    def invoke_api(self, input_sequence:str, reflex=False):
        def inner():
            messages = [{"role": "system", "content": f"{self.system_prompt}"}] \
                if self.algorithm_config.system_description_in_user == False else []
            messages.append({"role": "user", "content": input_sequence})
            if reflex:
                return self.algorithm_config.reflex_invoke_fn(messages)
            else:
                return self.algorithm_config.act_invoke_fn(messages)
        return inner()


class IntFloatEncoder(json.JSONEncoder):
    """
    Used in json.dumps to encode integer `float` to `int` (4.0 -> 4)
    and `float` to a fixed number of digits (1.234 -> 1.23).
    The goal is to save input tokens to LLM.
    """
    def __init__(self, round_digits=2, **kwargs):
        super().__init__(**kwargs)
        self.round_digits = round_digits

    def encode(self, obj):
        def _format(o):
            if isinstance(o, float):
                if o.is_integer():
                    return int(o)
                else:
                    return round(o, self.round_digits)
            elif isinstance(o, dict):
                return {k: _format(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [_format(e) for e in o]
            else:
                return str(o)

        formatted_obj = _format(obj)
        return super().encode(formatted_obj)