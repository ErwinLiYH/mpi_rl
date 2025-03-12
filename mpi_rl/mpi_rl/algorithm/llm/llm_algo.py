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

from mpi_rl.utils import dedent
from functools import partial
import json
import os

from tenacity import retry, stop_after_attempt

# Typing Aliases
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import Markdown_Path_Without_NameT, MessagesT, \
    ContentT, Reasoning_ContentT, Num_Input_TokensT, Num_Output_TokensT, Input_SequenceT, Invoker_outputT

@dataclass
class LLMConfig(BaseAlgorithmConfig):
    system_description: str
    action_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    reflection_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    metrics_statistics_fn:Callable[[Episode], Any]
    system_description_in_user: bool = False
    eval_invoke_fn: Optional[Callable[[MessagesT], Invoker_outputT]] = None
    model_tokens_counter_fn: Callable[[Input_SequenceT],int] = None
    max_tokens_input: int = None
    transaction_process_fn: Callable = lambda x: x     # A function to process the transaction before convert to json string,
                                                       # do 1.select the useful data, 2. convert the data type to basic type.
                                                       # Need convert numpy array or pytorch tensor to list to save space
                                                       # because numpy array and pytorch tensor can't using `IntFloatEncoder`,
                                                       # then just `str()` for them, which will waste token count.
                                                       # e.g. str(np.array([1.1111111, 2.0, 3.0]) => "[1.1111111 2.        3.       ]"
                                                       # if using `IntFloatEncoder`, it will be "[1.11, 2, 3]"
    obs_process_fn: Callable = lambda x: x     # A function to process the observation before convert to json string, similar to transaction_process_fn
    round_digits: int = 2
    gen_markdown: None|Markdown_Path_Without_NameT = None
    act_user_template: str = "Please generate action according to the current episode history and previous policy conclusion."
    act_assistant_template: str = ""
    eval_user_template: str = "Please generate action according to the current episode history and previous policy conclusion."
    eval_assistant_template: str = ""
    reflection_user_template: str = "These are some full episode history and the policy previously generated, please generate reflection on them. Think about whether the episode is good enough, whether there are spaces for improvement, and how to act later. Think about how improve the current policy (if exit) and generate a new policy conclusion. If previous policy is None, just generate a new one. Don't need generate action json structure here."
    reflection_assistant_template: str = ""

    def __post_init__(self):
        # model_tokens_counter_fn and max_tokens_input should be None together or not None together
        assert (self.model_tokens_counter_fn is None and self.max_tokens_input is None) or \
                (self.model_tokens_counter_fn is not None and self.max_tokens_input is not None), \
                "model_tokens_counter_fn and max_tokens_input should be None together or not None together."
        
        if self.gen_markdown is not None:
            os.makedirs(self.gen_markdown, exist_ok=True)

class LLM(BaseAlgorithm):
    """
    LLM algorithm is a customized `BaseAlgorithm` for large language model.
    It is designed to use large language model to solve the task in pure Natural Language with chain-of-reflection.
    """
    def __init__(
            self,
            train_env_creator: Callable,
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

    def check_config(self):
        assert isinstance(self.algorithm_config, LLMConfig), "algorithm_config should be LLMConfig."

    def compute_action(self, obs):
        compressed_replay_buffer_str = self.replay_buffer_to_str(self.replay_buffer, separators=(',', ':')) \
            if len(self.replay_buffer) > 0 else 'None'
        compressed_episode_str = self.episode_to_str(self.operating_episode, separators=(',', ':')) \
            if len(self.operating_episode) > 0 else 'None'
        obs_str = json.dumps(self.algorithm_config.obs_process_fn(obs), cls=IntFloatEncoder, 
                             round_digits=self.algorithm_config.round_digits, 
                             separators=(',', ':'), sort_keys=True, ensure_ascii=False, default=str)

        input_sequence = dedent(f"""\
            **Previous Episodes History (separated by '++++++++++'):**
            {compressed_replay_buffer_str}

            **Current Episode History:**
            {compressed_episode_str}
            
            **Current Observation:**
            {obs_str}

            **Previous Policy:**
            {self.policy_conclusion if self.policy_conclusion is not None else 'None'}

            {self.algorithm_config.act_prompt}""", char=' '*12, plus_char='')

        # check if need to reduce previous replay buffer
        expected_input_token_count = \
            self.algorithm_config.model_tokens_counter_fn(input_sequence+self.algorithm_config.system_prompt)
        self.logger.debug(f"Expected Input token count: {expected_input_token_count}")
        if expected_input_token_count > self.algorithm_config.max_tokens_input:
            self.logger.info(f"Expected Input token count: {expected_input_token_count} \
                             > max_tokens_input: {self.algorithm_config.max_tokens_input}.")
            to_reduce = expected_input_token_count - self.algorithm_config.max_tokens_input
            encoded_replay_buffer_str = self.encoder(compressed_replay_buffer_str)
            # reduce previous replay buffer
            encoded_replay_buffer_str = encoded_replay_buffer_str[to_reduce+10:]   # plus 10 for safety
            compressed_replay_buffer_str = self.encoder.decode(encoded_replay_buffer_str)
        
        self.logger.debug(input_sequence)

        content, llm_action, reasoning_content, input_len, output_len = \
            self.invoke_api(input_sequence)

        # record in markdown
        if self.algorithm_config.gen_markdown is not None:
            reasoning_template = "<details>\n    <summary>Thinking:\
                </summary>\n   {reasoning_content}\n</details>"

            record_str = dedent(f"""\
                **User:**

                <details>
                    <summary>Previous Episodes History (separated by '++++++++++'):</summary>
                    {self.replay_buffer_to_str(self.replay_buffer) if len(self.replay_buffer) > 0 else 'None'}
                </details>
                
                <details>
                    <summary>Current Episode History:</summary>
                    {self.episode_to_str(self.operating_episode) if len(self.operating_episode) > 0 else 'None'}
                </details>

                Current Observation:
                {obs_str}

                <details>
                    <summary>Previous Policy</summary>
                    {self.policy_conclusion if self.policy_conclusion is not None else 'None'}
                </details>

                {self.algorithm_config.act_prompt}
                
                **Model:**
                {
                    reasoning_template.format(reasoning_content=reasoning_content)
                    if reasoning_content is not None else ''
                }

                <details>
                    <summary>Content:</summary>
                    {content}
                </details>

                *Steps: {len(self.operating_episode)+1}, Token usage <Expected input: {expected_input_token_count}; Input: {input_len}; Output: {output_len}>*

                ---\n""", char=' '*16)

            with open(f"{os.path.join(self.algorithm_config.gen_markdown, self.name)}_ca_{self.status}_history.md", 'a') as f:
                f.write(record_str)
        
        # if len(self.operating_episode) > 0:
        #     self.operating_episode[-1]["reflection"] = llm_action["prev_step_reflection"]

        self.operating_transaction.update({"prev_step_reflection": llm_action["prev_step_reflection"]})
        self.operating_transaction.update({"analysis": llm_action["analysis"]})

        return llm_action["action"]

    def train(self, episodes: Episode):
        self.temp_episodes_llm_algo_specific = episodes
        # This algorithm need to store the episodes to replay_buffer in all ranks in MPI mode,
        # so we store the episodes in a temporary variable and store them to replay_buffer in post_train_mpi
        # which will run on all ranks.

        metrics = [self.algorithm_config.metrics_statistics_fn(episodes[i]) for i in range(len(episodes))]

        template_compressed = dedent("""\
        **Episode {n}:**
        {episode}
        **Metrics for episode {n}**
        {metrics}""", char=' '*8)

        input_sequence = dedent(f"""\
        {("+"*10+os.linesep).join([
            template_compressed.format(
                n = i+1,
                episode = self.episode_to_str(episodes[i], separators=(',', ':')),
                metrics = json.dumps(metrics[i], 
                                     cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits,
                                     separators=(',', ':'), sort_keys=True, ensure_ascii=False, default=str)
            ) for i in range(len(episodes))]
            )}

        **Previous Policy:**
        {self.policy_conclusion if self.policy_conclusion is not None else 'None'}

        {self.algorithm_config.reflection_prompt}""", char=' '*8, plus_char='')

        # We assume input sequence in reflection step never exceed max_tokens_input
        # so we don't check token count here.
        
        self.logger.debug(input_sequence)

        content, _, reasoning_content, input_len, output_len = \
            self.invoke_api(input_sequence, no_action=True)

        # record in markdown
        if self.algorithm_config.gen_markdown is not None:
            reasoning_template = "<details>\n    <summary>Thinking:\
                </summary>\n   {reasoning_content}\n</details>"

            markdown_template = dedent("""\
            <details>
                <summary>**Episode {n}:**</summary>

                {episode}
            </details>

            **Metrics for episode {n}**
                {metrics}\n""",char=' '*12)

            record_str = dedent(f"""\
            **User:**

                {(os.linesep+"+"*10+os.linesep*2).join([
                    markdown_template.format(
                        n = i+1,
                        episode = self.episode_to_str(episodes[i]),
                        metrics = json.dumps(metrics[i], 
                                            cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, 
                                            sort_keys=True, ensure_ascii=False, default=str)
                    ) for i in range(len(episodes))
                ])}

            **Previous Policy:**
            {self.policy_conclusion if self.policy_conclusion is not None else 'None'}

            {self.algorithm_config.reflection_prompt}

            **Model:**
            {
                reasoning_template.format(reasoning_content=reasoning_content)
                if reasoning_content is not None else ' '
            }

            <details>
                    <summary>Content:</summary>
                    {content}
            </details>

            *Token usage <Input: {input_len} ; Output: {output_len}>*

            ---\n""", char=' '*12)

            with open(f"{os.path.join(self.algorithm_config.gen_markdown, self.name)}_rf_{self.status}_history.md", 'a') as f:
                f.write(record_str)

        self.policy_conclusion = content

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
    
    def replay_buffer_to_str(self, replay_buffer:ReplayBuffer, **kwargs):
        """
        Concert the replay_buffer to a compressed string (to save space) for LLM.
        """
        str_list = []
        episode_index = 0
        for i in range(len(replay_buffer)):
            transaction = self.algorithm_config.transaction_process_fn(replay_buffer[i]["transaction"])
            index = replay_buffer[i]["episode_index"]
            if index == episode_index+1:
                str_list.append("++++++++++\n")
                episode_index += 1
            json_str = json.dumps(transaction, cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, \
                                  sort_keys=True, ensure_ascii=False, default=str, **kwargs)
            str_list.append(json_str+"\n")
        return "\n".join(str_list)
    
    def episode_to_str(self, episode:Episode, **kwargs):
        """
        Concert the episode to a compressed string (to save space) for LLM.
        """
        str_list = []
        for i in range(len(episode)):
            transaction = self.algorithm_config.transaction_process_fn(episode[i]["transaction"])
            json_str = json.dumps(transaction, cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, \
                                  sort_keys=True, ensure_ascii=False, default=str, **kwargs)
            str_list.append(json_str+"\n")
        return "\n".join(str_list)

    def invoke_api(self, input_sequence:str, no_action=False):
        def inner():
            messages = [
                {
                    "role": "system", "content": f"{self.algorithm_config.system_prompt}"
                },
                {
                    "role": "user",
                    "content": input_sequence
                }
            ]
            if no_action:
                return self.algorithm_config.reflection_invoke_fn(messages)
            else:
                return self.algorithm_config.action_invoke_fn(messages)
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