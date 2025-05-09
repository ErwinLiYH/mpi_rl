# whether to use mpi or not ------------
from mpi_rl.config import get_use_mpi
if get_use_mpi():
    from mpi_rl.mpi_utils.mpi_architecture_config import MPIArchitectureConfig
# --------------------------------------
from .llm_algo import LLM, LLMConfig, IntFloatEncoder
from typing import Callable, List, Dict, Any
from dataclasses import dataclass

from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils.replay_buffer import ReplayBuffer, Episode
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import BaseInvoker
from mpi_rl.mpi_rl.algorithm.llm.prompt_template import ActTemplateMA, ReflexTemplate
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import make_conversation

from concurrent.futures import ThreadPoolExecutor
import threading

import json
import os


lock = threading.Lock()

@dataclass
class LLM_MAConfig(LLMConfig):
    act_invoke_fn: Dict[Any, BaseInvoker]|BaseInvoker
    reflex_invoke_fn: Dict[Any, BaseInvoker]|BaseInvoker
    teach_invoke_fn: Dict[Any, BaseInvoker]|BaseInvoker
    act_multi_threading: bool = False

class LLM_MA(LLM):
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
            log_paths:List[str] = ["print"],
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
            log_paths = log_paths,
            multi_agent = True,
            transaction_process_fn = transaction_process_fn
            )
        
        self.check_config()
        
        if self.algorithm_config.system_description_in_user:
            self.act_template = ActTemplateMA(algorithm_config.system_description, algorithm_config.act_prompt)
            self.reflex_template = ReflexTemplate(algorithm_config.system_description, algorithm_config.reflex_prompt)
            self.system_prompt = self.algorithm_config.system_prompt
        else:
            self.act_template = ActTemplateMA("", algorithm_config.act_prompt)
            self.reflex_template = ReflexTemplate("", algorithm_config.reflex_prompt)
            if self.algorithm_config.system_prompt != "" and isinstance(self.algorithm_config.system_prompt, str):
                self.system_prompt = self.algorithm_config.system_prompt + "\n\n" + self.algorithm_config.system_description
            else:
                self.system_prompt = self.algorithm_config.system_description

    def compute_action(self, obs):
        # get all agents
        agents = self._env.agents

        # try to get the state
        try:
            state = self._env.state()
        except Exception as e:
            state = None

        episode_str = self.episode_to_str(
            self.operating_episode[-self.algorithm_config.max_history:],
            separators=(',', ':')
        ) if len(self.operating_episode) > 0 else 'None'

        def process_agent(agent, invoker):
            obs_str = json.dumps(self.algorithm_config.obs_process_fn(obs[agent]), cls=IntFloatEncoder, 
                                round_digits=self.algorithm_config.round_digits, 
                                separators=(',', ':'), sort_keys=True, ensure_ascii=False, default=str)

            input_sequence = self.act_template.format(
                obs=obs_str,
                partial_episode=episode_str,
                step=str(self.progress_info["num_local_step"]),
                agent=agent
            )
            
            self.logger.debug(input_sequence)

            message = self.make_message(input_sequence)

            content, llm_action, reasoning_content, input_len, output_len = invoker(message)

            # record
            if self.algorithm_config.record_conversation:
                filename = f"{os.path.join(self.algorithm_config.record_conversation, self.name)}_compute_act_{self.status}_{agent}.txt"
                with open(filename, 'a') as f:
                    conversation = make_conversation(
                        input_sequence, reasoning_content, content, "None", input_len, output_len
                    )
                    f.write(conversation)
            
            with lock:
                self.operating_transaction["prev_step_reflection"][agent] = llm_action["prev_step_reflection"]
                self.operating_transaction["analysis"][agent] = llm_action["analysis"]
                self.operating_transaction["content"][agent] = content
                self.operating_transaction["reasoning_content"][agent] = reasoning_content
                self.operating_transaction["input_sequence"] = input_sequence

            return agent,llm_action

        if self.algorithm_config.act_multi_threading:
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                futures = []
                for agent in agents:
                    if isinstance(self.algorithm_config.act_invoke_fn, dict):
                        invoker = self.algorithm_config.act_invoke_fn[agent]
                    else:
                        invoker = self.algorithm_config.act_invoke_fn
                    futures.append(executor.submit(process_agent, agent, invoker))
                llm_actions = [future.result() for future in futures]
                
        else:
            llm_actions = []
            for agent in agents:
                if isinstance(self.algorithm_config.act_invoke_fn, dict):
                    invoker = self.algorithm_config.act_invoke_fn[agent]
                else:
                    invoker = self.algorithm_config.act_invoke_fn
                llm_actions.append(process_agent(agent, invoker))

        return {agent: act["action"]+self.algorithm_config.action_offset for agent, act in llm_actions}
