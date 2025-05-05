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

from typing import Any, Callable, Optional, List

from mpi_rl.utils.replay_buffer import Episode, ReplayBuffer, Transaction
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils import chunk

from mpi_rl.mpi_rl.algorithm.llm.prompt_template import ActTemplate, ReflexTemplate, TeachTemplate
from mpi_rl.utils import remove_first_line

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

# Typing Aliases
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import Path_Without_NameT, MessagesT, \
    Input_SequenceT, Invoker_outputT, Kto_Rollout_StatisticsT, BaseInvoker

@dataclass
class LLMConfig(BaseAlgorithmConfig):
    system_description: str
    act_prompt:str
    reflex_prompt:str
    teach_prompt:str
    act_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    reflex_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    teach_invoke_fn: Callable[[MessagesT], Invoker_outputT]|List[Callable[[MessagesT], Invoker_outputT]]
    metrics_statistics_fn:Callable[[Episode], Any]
    system_prompt: str = ""
    system_description_in_user: bool = True
    teach_model_tokens_counter_fn: Optional[Callable[[Input_SequenceT],int]] = None
    reflex_model_tokens_counter_fn: Optional[Callable[[Input_SequenceT],int]] = None
    act_model_tokens_counter_fn: Optional[Callable[[Input_SequenceT],int]] = None
    reflex_max_tokens_input: int = None
    act_max_tokens_input: int = None
    teach_max_tokens_input: int = None
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
    fine_tune_data_path: Optional[Path_Without_NameT] = None
    action_offset:float = 0
    teach_method:str = "instruction" # "instruction" or "kto"
    kto_rollout_stop_fn: Callable[[Kto_Rollout_StatisticsT], bool] = lambda x: x["total_times"] >= 20
    kto_human_refine:bool = False
    kto_finish_check_fn: Callable[[Kto_Rollout_StatisticsT], bool] = lambda x: x["good_times"] > 0
    kto_specified_transaction_process_fn: Optional[Callable] = lambda x: x

    def __post_init__(self):
        # model_tokens_counter_fn and max_tokens_input should be None together or not None together
        assert (self.act_model_tokens_counter_fn is None and self.act_max_tokens_input is None) or \
                (self.act_model_tokens_counter_fn is not None and self.act_max_tokens_input is not None), \
                "act_model_tokens_counter_fn and act_max_tokens_input should be None together or not None together."
        
        assert (self.reflex_model_tokens_counter_fn is None and self.reflex_max_tokens_input is None) or \
                (self.reflex_model_tokens_counter_fn is not None and self.reflex_max_tokens_input is not None), \
                "reflex_model_tokens_counter_fn and reflex_max_tokens_input should be None together or not None together."
        
        assert (self.teach_model_tokens_counter_fn is None and self.teach_max_tokens_input is None) or \
                (self.teach_model_tokens_counter_fn is not None and self.teach_max_tokens_input is not None), \
                "teach_model_tokens_counter_fn and teach_max_tokens_input should be None together or not None together."
        
        if self.record_conversation is not None:
            os.makedirs(self.record_conversation, exist_ok=True)

        if self.fine_tune_data_path is not None:
            os.makedirs(self.fine_tune_data_path, exist_ok=True)

        if self.teach_method == "kto":
            assert self.kto_rollout_stop_fn is not None, "kto_rollout_stop_fn should be set when teach_method is kto."

        if self.teach_method == "instruction":
            assert self.kto_rollout_stop_fn is None, "kto_rollout_stop_fn should not be set when teach_method is instruction."

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
            log_paths:List[str] = ["print"],
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
            log_paths = log_paths,
            multi_agent = multi_agent,
            transaction_process_fn = transaction_process_fn
            )
        
        self.check_config()

        self.policy_conclusion = None

        if self.algorithm_config.system_description_in_user:
            self.act_template = ActTemplate(algorithm_config.system_description, algorithm_config.act_prompt)
            self.reflex_template = ReflexTemplate(algorithm_config.system_description, algorithm_config.reflex_prompt)
            self.teach_template = TeachTemplate(algorithm_config.system_description, algorithm_config.teach_prompt)
            self.system_prompt = self.algorithm_config.system_prompt
        else:
            self.act_template = ActTemplate("", algorithm_config.act_prompt)
            self.reflex_template = ReflexTemplate("", algorithm_config.reflex_prompt)
            self.teach_template = TeachTemplate("", algorithm_config.teach_prompt)
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
        
        partial_episode = "下面是按顺序排列的历史步骤:\n(如果为None则代表这是初始步,当前无历史记录)\n\n"\
                        + episode_str \
                        + "\n\n"
        
        obs_str = json.dumps(self.algorithm_config.obs_process_fn(obs), cls=IntFloatEncoder, 
                             round_digits=self.algorithm_config.round_digits, 
                             separators=(',', ':'), sort_keys=True, ensure_ascii=False, default=str)
        
        obs = f"下面是 step {self.progress_info['num_local_step']} 的观测:\n\n"\
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
        
        self.operating_transaction.update({"prev_step_reflection": llm_action["prev_step_reflection"]})
        self.operating_transaction.update({"analysis": llm_action["analysis"]})
        self.operating_transaction.update({"content": content})
        self.operating_transaction.update({"reasoning_content": reasoning_content})
        self.operating_transaction.update({"input_sequence": input_sequence})

        return llm_action["action"] + self.algorithm_config.action_offset

    def train(self, episodes: list[Episode]):
        self.temp_episodes_llm_algo_specific = episodes
        # This algorithm need to store the episodes to replay_buffer in all ranks in MPI mode,
        # so we store the episodes in a temporary variable and store them to replay_buffer in post_train
        # which will run on all ranks.
        import pickle
        iteration = self.progress_info["current_iteration"]
        with open(os.path.join(self.algorithm_config.record_conversation, f"episodes_{iteration}_ori.pkl"),'wb') as f:
            pickle.dump(episodes, f)

        fine_tune_data_path_list = []
        for idx, ep in enumerate(episodes):
            self.reflex(ep, idx)
        with open(os.path.join(self.algorithm_config.record_conversation, f"episodes_{iteration}_reflex.pkl"),'wb') as f:
            pickle.dump(episodes, f)
        for idx, ep in enumerate(episodes):
            self.teach(ep, idx)
            fine_tune_data_path_list.append(self.build_finetune_instruction(ep, idx))

        # TODO: move this to post train
        # save episodes for test
        with open(os.path.join(self.algorithm_config.record_conversation, f"episodes_{iteration}_reflex_teach.pkl"),'wb') as f:
            pickle.dump(episodes, f)

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

            self.logger.info(f"Reflex - Episode {idx} - Step {win_start} to {win_end}")
            
            for i, ref in enumerate(llm_reflex):
                episode[win_start+i]["transaction"]["reflex"] = ref

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

    def teach(self, episode:Episode, idx:int):
        """
        construct instruction sft data, by:
            for all bad steps in the episode:
                1. extend the original reasoning by a adversative clause (`turning` in code), according to the reflection
                2. generate new action according to the reflection
                3. construct the instruction data: instruction=original input, output=original reasoning + turning + new action
            for all good steps in the episode:
                1. construct the instruction data: instruction=original input, output=original reasoning + original action analysis
        """
        for i, t in enumerate(episode):
            original_action_analysis = {
                "action": t["transaction"]["action"]-self.algorithm_config.action_offset,
                "analysis": t["transaction"]["analysis"],
                "prev_step_reflection": t["transaction"]["prev_step_reflection"],
            }
            if t["transaction"]["reflex"]['reflection']["good"] == False:

                # partial episode history
                left_index = i - self.algorithm_config.max_history
                partial_episode = episode[0:i+1] if left_index < 0 else episode[left_index:i+1]
                episode_str = self.episode_to_str(
                    partial_episode,
                    separators=(',', ':')
                ) if len(self.operating_episode) > 0 else 'None'
                episode_str = "下面是按时间顺序排列的历史步骤:\n(如果为None则代表这是初始步,当前无历史记录)\n\n"\
                            + episode_str + "\n" \
                            + "\n"
                
                # reasoning content
                reasoning = t["transaction"]["reasoning_content"]
                reasoning_str = f"下面是 step {t['index_in_episode']} 中actor的思考过程：\n\n" \
                            + reasoning + "\n" \
                            + "\n"

                # original action analysis
                original_action_analysis_str = json.dumps(original_action_analysis, cls=IntFloatEncoder, \
                    round_digits=self.algorithm_config.round_digits, sort_keys=True, ensure_ascii=False, default=str)
                original_action_analysis_str = f"下面是 step {t['index_in_episode']} 中actor生成的原始动作：\n\n" \
                                            + original_action_analysis_str + "\n" \
                                            + "\n"
                
                # reflection
                reflex = t["transaction"]["reflex"]
                reflex_str = json.dumps(reflex, cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, \
                    sort_keys=True, ensure_ascii=False, default=str)
                reflex_str = f"下面是reflector对 step {t['index_in_episode']} 的反思：\n\n" \
                        + reflex_str + "\n" \
                        + "\n"

                # invoke API
                input_sequence = self.teach_template.format(
                    partial_episode=episode_str,
                    reasoning_content=reasoning_str,
                    original_action_analysis=original_action_analysis_str,
                    reflection=reflex_str
                )
                content, llm_teach, reasoning_content, input_len, output_len = self.invoke_api(input_sequence, teach=True)

                self.logger.info(f"Teach - Step {t['index_in_episode']}")

                turning = [j for j in llm_teach if j["language"] == "text"][0]["code"]
                new_action = [j for j in llm_teach if j["language"] == "json"][0]["text"]

                episode[i]["transaction"]["turning"] = turning
                episode[i]["transaction"]["new_action"] = new_action

                # record conversation
                if self.algorithm_config.record_conversation:
                    it = self.progress_info["current_iteration"]
                    with open(
                        f"{os.path.join(self.algorithm_config.record_conversation, self.name)}_teach_iter{it}_ep{idx}_history.txt",
                        'a'
                    ) as f:
                        conversation = self.make_conversation(
                            input_sequence, reasoning_content, content, "None", input_len, output_len
                        )
                        f.write(conversation)
    
    def build_finetune_instruction(self, episode: Episode, idx:int):
        """generate instruction finetune data for `self.teach()`"""
        jsonl_list = []
        for t in episode:
            if t["transaction"]["reflex"]['reflection']["good"] == False:
                turning = t["transaction"]["turning"]
                new_action = t["transaction"]["new_action"]
                data = {
                    "instruction":t["transaction"]["input_sequence"] ,
                    "input": "",
                    "output": f"<think>\n{t['transaction']['reasoning_content']}\n{turning}\n</think>\n{new_action}\n",
                }
            else:
                original_action_analysis = {
                    "prev_step_reflection": t["transaction"]["prev_step_reflection"],
                    "analysis": t["transaction"]["analysis"],
                    "action": t["transaction"]["action"]-self.algorithm_config.action_offset,
                }
                original_action_analysis = json.dumps(original_action_analysis, sort_keys=False, ensure_ascii=False, default=str)
                                                      
                data = {
                    "instruction":t["transaction"]["input_sequence"] ,
                    "input": "",
                    "output": f"<think>\n{t['transaction']['reasoning_content']}\n</think>\n```json\n{original_action_analysis}\n```\n",
                }
            data["step"] = t["index_in_episode"]
            data["reflex_good"] = t["transaction"]["reflex"]['reflection']["good"]
            jsonl_list.append(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str))

        # save jsonl data to file
        jsonl_str = "\n".join(jsonl_list)
        it = self.progress_info["current_iteration"]
        file_name = f"{os.path.join(self.algorithm_config.fine_tune_data_path, self.name)}_finetune_iter{it}_{idx}.jsonl"
        with open(file_name, 'w') as f:
            f.write(jsonl_str)
        return file_name
    
    def kto_iteration(self, episode: Episode, ep_idx:int, kto_it_idx:int):
        """"rollout kto data"""
        # make folder to record conversation
        it = self.progress_info["current_iteration"] # train iteration
        if self.algorithm_config.record_conversation:
            os.makedirs(os.path.join(self.algorithm_config.record_conversation, f"{self.name}_iter{it}", f"kto_iter{kto_it_idx}"), exist_ok=True)
        # human refine data
        if self.algorithm_config.kto_human_refine:
            self.kto_human_refine(episode, ep_idx)
        # rollout kto data
        final_data = []
        unfinished_transactions = []
        # TODO check good here
        for trans in episode:
            kto_finished = trans["transaction"]["reflex"]["reflection"].get("kto_finished", False)
            reflex = trans["transaction"]["reflex"]
            human_good = reflex["reflection"].get("human_good", None)
            reflex_good = reflex["reflection"].get("good", None)
            input_sequence = trans["transaction"]["input_sequence"]
            transaction = trans["transaction"]
            if human_good is not None:
                good = human_good
            else:
                if reflex_good is not None:
                    good = reflex_good
                else:
                    raise ValueError("human_good and reflex_good are both None.")
            if good and (kto_finished==False):
                reflex["reflection"]["kto_finished"] = True
                reflex["reflection"]["kto_remark"] = "Kto finished"
                final_data.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": input_sequence
                        },
                        {
                            "from": "gpt",
                            "value": f"<think>\n{transaction['reasoning_content']}\n</think>\n{transaction['content']}\n"
                        }
                    ],
                    "step": trans["index_in_episode"],
                    "episode_index": ep_idx,
                    "iteration": self.progress_info["current_iteration"],
                    "kto_tag": True
                })
            elif not good and (kto_finished==False):
                unfinished_transactions.append(trans)
            else:
                pass
        if len(unfinished_transactions) == 0:
            self.logger.info(f"All transactions in episode {ep_idx} are finished.")
            return None
        else:
            self.logger.info(f"Number of unfinished transactions in episode {ep_idx}: {len(unfinished_transactions)}")
            for unfinished_transaction in unfinished_transactions:
                self.logger.info(f"Unfinished transaction: {unfinished_transaction['index_in_episode']}")
        if isinstance(self.algorithm_config.teach_invoke_fn, list) and len(self.algorithm_config.teach_invoke_fn)>1:  # multi-thread mode
            n_thread = len(self.algorithm_config.teach_invoke_fn)
            self.logger.info(f"Got {n_thread} invokers, using multi-thread mode for KTO.")
            # separate the unfinished transactions into n_thread parts
            unfinished_transactions_lst = chunk(unfinished_transactions, n_thread)
            # invoke the transactions in parallel
            with ThreadPoolExecutor(max_workers=n_thread) as executor:
                futures = [
                    executor.submit(
                        self.kto_episode, 
                        unfinished_transactions_lst[i],
                        self.algorithm_config.teach_invoke_fn[i],
                        ep_idx,
                        i,
                        kto_it_idx
                    ) for i in range(n_thread)
                ]
                res = [future.result() for future in futures]
                for r in res:
                    final_data.extend(r)
        else: # single-thread mode
            self.logger.info(f"Got 1 invoker, using single-thread mode for KTO.")
            if isinstance(self.algorithm_config.teach_invoke_fn, list):
                teach_invoke_fn = self.algorithm_config.teach_invoke_fn[0]
            else:
                teach_invoke_fn = self.algorithm_config.teach_invoke_fn
            final_data = self.kto_episode(unfinished_transactions, teach_invoke_fn, ep_idx, kto_it_idx)
        # log final situation in this Kto iteration
        rollout_total = 0
        finished_num = 0
        for trans in unfinished_transactions:
            self.logger.info(f"Episode {ep_idx} - Step {trans['index_in_episode']} - {trans['transaction']['reflex']['reflection']['kto_remark']}")
            rollout_total += 1
            if trans["transaction"]["reflex"]["reflection"].get("kto_finished", False):
                finished_num += 1
        self.logger.info(f"Episode {ep_idx} - Kto iteration {kto_it_idx} - Finished {finished_num}/{rollout_total} transactions.")
        # save json data to file
        data_dir = os.path.join(self.algorithm_config.fine_tune_data_path, f"{self.name}_iter{it}")
        os.makedirs(data_dir, exist_ok=True) # guarantee the directory exists
        file_name = os.path.join(data_dir, f"kto_finetune_iter{kto_it_idx}_ep{ep_idx}.json")
        with open(file_name, 'w') as f:
            json.dump(final_data, f, indent=4, sort_keys=False, ensure_ascii=False, default=str)
        return file_name, finished_num, rollout_total
    
    def kto_human_refine(self, episode: Episode, ep_idx:int):
        it = self.progress_info["current_iteration"] # train iteration
        file_name = f"{os.path.join(self.algorithm_config.fine_tune_data_path, self.name)}_kto_human_refine_iter{it}_{ep_idx}.txt"

        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                f.write(s)
            str_list = []
            for i in range(len(episode)):
                reflection = episode[i]["transaction"]["reflex"]["reflection"]
                transaction = self.algorithm_config.kto_specified_transaction_process_fn(episode[i]["transaction"])
                json_str = json.dumps(transaction, cls=IntFloatEncoder, round_digits=self.algorithm_config.round_digits, \
                                    sort_keys=True, ensure_ascii=False, default=str)
                human_suggested_action = reflection.get("human_suggested_action", None)
                if human_suggested_action is not None:
                    json_str = f"Step {episode[i]['index_in_episode']}: {json_str}{{{human_suggested_action}}}"
                else:
                    original_action = episode[i]["transaction"]["action"]-self.algorithm_config.action_offset
                    json_str = f"Step {episode[i]['index_in_episode']}: {json_str}{{{original_action}}}"
                str_list.append(json_str)
            s = "\n".join(str_list)
            # save s to file

            print(f"Please edit the file {file_name} to add human suggested action.")
            input("Edit the file and press Enter to continue...")

        # read the file and update the transaction
        with open(file_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            match = re.search(r'Step (\d+).*?\{(\d+)\}', line)
            if match:
                step = int(match.group(1))
                human_suggested_action = int(match.group(2))
                original_action = episode[step]["transaction"]["action"]-self.algorithm_config.action_offset
                human_good = human_suggested_action==original_action
                episode[step]["transaction"]["reflex"]["reflection"]["human_suggested_action"] = human_suggested_action
                episode[step]["transaction"]["reflex"]["reflection"]["human_good"] = human_good
                self.logger.info(f"Step {step} - {original_action} - Human suggested action: {human_suggested_action}, Good: {human_good}")
    
    def kto_episode(self, transactions: list[Transaction], invoker:BaseInvoker, ep_idx:int, thread_idx:int=0, kto_it_idx:int=0):
        final_data = []
        for trans in transactions:
            data = self.kto_transaction(trans["transaction"], invoker, trans["index_in_episode"], ep_idx, thread_idx, kto_it_idx)
            final_data.extend(data)
        return final_data

    def kto_transaction(self, transaction:Transaction, invoker:BaseInvoker, trans_idx:int, ep_idx:int, thread_idx:int=0, kto_it_idx:int=0):
        kto_finished = transaction["reflex"]["reflection"].get("kto_finished", False)
        if kto_finished:
            return []
        data = []
        input_sequence = transaction["input_sequence"]
        reflex = transaction["reflex"]
        human_good = reflex["reflection"].get("human_good", None)
        reflex_good = reflex["reflection"].get("good", None)
        if human_good is not None:
            good = human_good
        else:
            if reflex_good is not None:
                good = reflex_good
            else:
                raise ValueError("human_good and reflex_good are both None.")
        if good:
            reflex["reflection"]["kto_finished"] = True
            reflex["reflection"]["kto_remark"] = "Kto finished"
            data.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": input_sequence
                    },
                    {
                        "from": "gpt",
                        "value": f"<think>\n{transaction['reasoning_content']}\n</think>\n{transaction['content']}\n"
                    }
                ],
                "step": trans_idx,
                "episode_index": ep_idx,
                "iteration": self.progress_info["current_iteration"],
                "kto_tag": True
            })
        else:
            rollout_statistics = {"total_times":0, "good_times":0, "bad_times":0}
            kto_actions = []
            data.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": input_sequence
                    },
                    {
                        "from": "gpt",
                        "value": f"<think>\n{transaction['reasoning_content']}\n</think>\n{transaction['content']}\n"
                    }
                ],
                "step": trans_idx,
                "episode_index": ep_idx,
                "iteration": self.progress_info["current_iteration"],
                "kto_tag": False
            })
            while True:
                reflex_suggested_action = reflex["reflection"].get("suggested_action", None)
                human_suggested_action = reflex["reflection"].get("human_suggested_action", None)
                if reflex_suggested_action is None and human_suggested_action is None:
                    raise ValueError("reflex_suggested_action and human_suggested_action are both None.")
                target_action = human_suggested_action if human_suggested_action is not None else reflex_suggested_action
                content, llm_teach, reasoning_content, input_len, output_len = invoker(self.build_messages(input_sequence))
                rollout_statistics["total_times"] += 1
                if llm_teach["action"] == target_action: # good action
                    kto_tag = True
                    remark = "kto rollout good"
                    rollout_statistics["good_times"] += 1
                else:
                    kto_tag = False
                    remark = f"kto rollout bad, expected action: {target_action}, got: {llm_teach['action']}"
                    rollout_statistics["bad_times"] += 1
                if thread_idx is not None:
                    self.logger.info(f"KTO - Thread {thread_idx} - Step {trans_idx} - Rollout <total: {rollout_statistics['total_times']} good: {rollout_statistics['good_times']} bad: {rollout_statistics['bad_times']}> - Expect: {target_action} - Got: {llm_teach['action']}")
                else:
                    self.logger.info(f"KTO - Step {trans_idx} - Rollout <total: {rollout_statistics['total_times']} good: {rollout_statistics['good_times']} bad: {rollout_statistics['bad_times']}> - Expect: {target_action} - Got: {llm_teach['action']}")
                kto_actions.append(llm_teach["action"])
                # build data
                data.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": input_sequence
                        },
                        {
                            "from": "gpt",
                            "value": f"<think>\n{reasoning_content}\n</think>\n{content}\n"
                        }
                    ],
                    "step": trans_idx,
                    "episode_index": ep_idx,
                    "iteration": self.progress_info["current_iteration"],
                    "kto_target_action": llm_teach["action"],
                    "kto_tag": kto_tag
                })
                # record conversation
                if self.algorithm_config.record_conversation:
                    it = self.progress_info["current_iteration"]
                    file_name = os.path.join(
                        self.algorithm_config.record_conversation,
                        f"{self.name}_iter{it}", f"kto_iter{kto_it_idx}",
                        f"kto_teach_ep{ep_idx}_step{trans_idx}_history.txt",
                        )
                    with open(file_name,'a') as f:
                        conversation = self.make_conversation(
                            (input_sequence if rollout_statistics["total_times"]==1 else None), reasoning_content,
                            content, "None", input_len, output_len, remark
                        )
                        f.write(conversation)
                # stop rollout
                if self.algorithm_config.kto_rollout_stop_fn(rollout_statistics):
                    if self.algorithm_config.kto_finish_check_fn(rollout_statistics):
                        reflex["reflection"]["kto_finished"] = True
                        reflex["reflection"]["kto_remark"] = "Kto finished"
                    else:
                        reflex["reflection"]["kto_finished"] = False
                        reflex["reflection"]["kto_remark"] = f"Kto not finished, expected action: {target_action}, rollout statistics: {rollout_statistics}"
                    if "kto_actions" in reflex["reflection"]:
                        reflex["reflection"]["kto_actions"].append(kto_actions)
                    else:
                        reflex["reflection"]["kto_actions"] = [kto_actions]
                    break
        return data

    def finetune_actor(self, episodes: list[Episode]):
        # make data for finetune
        finetune_conversation = []
        for ep in episodes:
            finetune_conversation.extend(self.build_finetune_conversation(ep))
        
        # begin finetune
        ## stop current model

        ## finetune

        ## start finetuned model

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

    def make_conversation(self, input_sequence, reasoning_content, content, expected_input_token_count, input_len, output_len, remark=None):
        conversation = ""
        if input_sequence is not None:
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
                        + (f"< remark: {remark} >\n" if remark!=None else "") \
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
            json_str = f"Step {replay_buffer[i]['index_in_episode']}: {json_str}"
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
            json_str = f"Step {episode[i]['index_in_episode']}: {json_str}"
            str_list.append(json_str)
        return "\n".join(str_list)

    def invoke_api(self, input_sequence:str, reflex=False, teach=False):
        def inner():
            messages = [{"role": "system", "content": f"{self.system_prompt}"}] \
                if self.algorithm_config.system_description_in_user == False else []
            messages.append({"role": "user", "content": input_sequence})
            if reflex:
                return self.algorithm_config.reflex_invoke_fn(messages)
            elif teach:
                return self.algorithm_config.teach_invoke_fn(messages)
            else:
                return self.algorithm_config.act_invoke_fn(messages)
        return inner()
    
    def build_messages(self, input_sequence:str):
        """
        Build the message for the LLM API.
        """
        messages = [{"role": "system", "content": f"{self.system_prompt}"}] \
            if self.algorithm_config.system_description_in_user == False else []
        messages.append({"role": "user", "content": input_sequence})
        return messages


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