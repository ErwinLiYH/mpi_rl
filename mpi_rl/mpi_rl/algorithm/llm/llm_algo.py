# whether to use mpi or not ------------
from mpi_rl.config import get_use_mpi
if get_use_mpi():
    from mpi_rl.mpi_utils.mpi_architecture_config import MPIArchitectureConfig
# --------------------------------------
from mpi_rl.mpi_rl.algorithm.base_algorithm import BaseAlgorithm, BaseAlgorithmConfig
from dataclasses import dataclass, field

from typing import Any, Callable, Optional, List

from mpi_rl.utils.replay_buffer import Episode, ReplayBuffer, Transaction
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils import chunk

from mpi_rl.mpi_rl.algorithm.llm.prompt_template import ActTemplate, ReflexTemplate
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import make_conversation, LLMManager

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

# Typing Aliases
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import Path_Without_NameT, MessagesT, \
    Input_SequenceT, Invoker_outputT, Kto_Rollout_StatisticsT, BaseInvoker, JsonableT

@dataclass
class EpisodeKTO:
    ep_idx: int
    agent: str
    episode: Episode
    promising_num_list: List[int] = field(default_factory=list)
    finished_num_list: List[int] = field(default_factory=list)
    total_rollout_list: List[int] = field(default_factory=list)
    kto_stop: bool = False
    num_kto_iter: int = 0

@dataclass
class LLMConfig(BaseAlgorithmConfig):
    system_description: str
    act_prompt:str
    reflex_prompt:str
    teach_prompt:str
    act_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    reflex_invoke_fn: Callable[[MessagesT], Invoker_outputT]
    teach_invoke_fn: Callable[[MessagesT], Invoker_outputT]|List[Callable[[MessagesT], Invoker_outputT]]
    metrics_statistics_fn:Callable[[Episode], JsonableT]
    system_prompt: str = ""
    system_description_in_user: bool = True
    transaction_process_fn: Callable[[Transaction], JsonableT] = lambda x: x     # A function to process the transaction before convert to json string,
                                                       # do 1.select the useful data, 2. convert the data type to basic type.
                                                       # Need convert numpy array or pytorch tensor to list to save space
                                                       # because numpy array and pytorch tensor can't using `IntFloatEncoder`,
                                                       # then just `str()` for them, which will waste token count.
                                                       # e.g. str(np.array([1.1111111, 2.0, 3.0]) => "[1.1111111 2.        3.       ]"
                                                       # if using `IntFloatEncoder`, it will be "[1.11, 2, 3]"
    obs_process_fn: Callable[[Any],JsonableT] = lambda x: x     # A function to process the observation before convert to json string, similar to transaction_process_fn
    round_digits: int = 2
    record_conversation: None|Path_Without_NameT = None
    max_history:int = 50
    reflex_batch:int = 5
    reflex_specific_transaction_process_fn: Optional[Callable[[Transaction], JsonableT]] = None
    fine_tune_data_path: Optional[Path_Without_NameT] = None
    action_offset:float = 0
    node_configs: Optional[dict] = None # can't use train() if None
    run_name: Optional[str] = None
    initial_train_config_path: Optional[str] = None
    initial_merge_config_path: Optional[str] = None
    kto_rollout_stop_fn: Callable[[Kto_Rollout_StatisticsT], bool] = lambda x: x["total_times"] >= 20
    kto_human_refine:bool = False
    kto_finish_check_fn: Callable[[Kto_Rollout_StatisticsT], bool] = lambda x: x["good_times"] > 0
    kto_specified_transaction_process_fn: Callable[[Transaction], JsonableT] = lambda x: x
    kto_max_iteration:int = 5
    kto_early_stop_check_iteration:int = 2
    skip_reflex: bool = False

    def __post_init__(self):
        if self.record_conversation is not None:
            os.makedirs(self.record_conversation, exist_ok=True)

        if self.fine_tune_data_path is not None:
            os.makedirs(self.fine_tune_data_path, exist_ok=True)

        if self.run_name is None:
            # init using datetime
            from datetime import datetime
            self.run_name = f"lrl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.initial_train_config_path is None:
            self.warning("initial_train_config_path is None, will raise error when fine-tune.")

        if self.initial_merge_config_path is None:
            self.warning("initial_merge_config_path is None, will raise error when fine-tune.")

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
            multi_agent = False,
            transaction_process_fn = transaction_process_fn
            )
        
        self.check_config()

        self.init_llm_manager()

        self.init_llm()

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

        if self.mpi_architecture_config is not None:
            if self.mpi_architecture_config.global_rank == 0:
                self.logger.info(f"Run name: {self.algorithm_config.run_name}")
        else:
            self.logger.info(f"Run name: {self.algorithm_config.run_name}")

    def check_config(self):
        assert isinstance(self.algorithm_config, LLMConfig), "algorithm_config should be LLMConfig."

    def init_llm_manager(self):
        if self.algorithm_config.node_configs is not None:
            self.llm_manager = LLMManager(self.algorithm_config.node_configs)
            self.current_model_path = self.algorithm_config.node_configs[0]["processes"][0]["model_path"]
        else:
            self.llm_manager = None
            self.logger.warning("No node_configs found, can't init LLMmanager")

    def init_llm(self):
        # initialize LLMs
        # LLMs will be closed in in self.fintune_actor()
        def init():
            if self.llm_manager is None:
                self.logger.warning("No LLMManager found, can't manager LLMs automatically.")
            else:
                self.logger.info("Initializing LLMs...")
                self.llm_manager.start_nodes()
                self.llm_manager.wait_route()
        if self.mpi_architecture_config is not None:
            if self.mpi_architecture_config.global_rank == 0:
                init()
        else:
            init()

    def compute_action(self, obs):
        episode_str = self.episode_to_str(
            self.operating_episode[-self.algorithm_config.max_history:],
            separators=(',', ':')
        ) if len(self.operating_episode) > 0 else 'None'
        
        obs_str = json.dumps(self.algorithm_config.obs_process_fn(obs), cls=IntFloatEncoder, 
                             round_digits=self.algorithm_config.round_digits, 
                             separators=(',', ':'), sort_keys=True, ensure_ascii=False, default=str)

        input_sequence = self.act_template.format(
            obs=obs_str,
            partial_episode=episode_str,
            step=str(self.progress_info["num_local_step"])
        )
        
        self.logger.debug(input_sequence)

        content, llm_action, reasoning_content, input_len, output_len = \
            self.invoke_api(input_sequence)

        # record
        if self.algorithm_config.record_conversation:
            with open(f"{os.path.join(self.algorithm_config.record_conversation, self.name)}_compute_act_{self.status}_history.txt", 'a') as f:
                conversation = make_conversation(
                    input_sequence, reasoning_content, content, "None", input_len, output_len
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

        if not self.algorithm_config.skip_reflex:
            for idx, ep in enumerate(episodes):
                self.reflex(ep, idx)
        else:
            # if skip reflex, we add empty reflex field to the transaction
            self.logger.info("Skip reflex, add empty reflex field to the transaction")
            for ep in episodes:
                for trans in ep:
                    trans["transaction"]["reflex"] = {"reflection":{}}
        with open(os.path.join(self.algorithm_config.record_conversation, f"episodes_{iteration}_reflex.pkl"),'wb') as f:
            pickle.dump(episodes, f)
        
        # human refine data
        for idx, ep in enumerate(episodes):
            if self.algorithm_config.kto_human_refine:
                self.kto_human_refine(ep, idx)
        # teach
        episode_wrappers = []
        for idx, ep in enumerate(episodes):
            episode_wrappers.append(
                EpisodeKTO(
                    ep_idx=idx,
                    agent=None,
                    episode=ep
                )
            )
        self.teach(episode_wrappers)
        
        with open(os.path.join(self.algorithm_config.record_conversation, f"episode_wrappers_{iteration}_teach.pkl"),'wb') as f:
            pickle.dump(episode_wrappers, f)

    def evaluate(self, episodes: list[Episode]):
        for ep in episodes:
            metrics_dict = self.algorithm_config.metrics_statistics_fn(ep)
            episode_metrics = json.dumps(metrics_dict)
            self.logger.info(f"Episode metrics: {episode_metrics}")

    def reflex(self, episode: Episode, ep_idx, agent=None):
        metrics_dict = self.algorithm_config.metrics_statistics_fn(episode)

        episode_metrics = json.dumps(metrics_dict)
        
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
        
            input_sequence = self.reflex_template.format(
                episode=episode_str,
                episode_metrics=episode_metrics
            )

            content, llm_reflex, reasoning_content, input_len, output_len = self.invoke_api(input_sequence, reflex=True)

            idx = f"{ep_idx}_{agent}" if agent is not None else ep_idx
            self.logger.info(f"Reflex - Episode {idx} - Step {win_start} to {win_end}")
            
            for i, ref in enumerate(llm_reflex):
                episode[win_start+i]["transaction"]["reflex"] = ref

            if self.algorithm_config.record_conversation:
                it = self.progress_info["current_iteration"]
                with open(
                    f"{os.path.join(self.algorithm_config.record_conversation, self.name)}_reflex_iter{it}_ep{idx}_history.txt",
                    'a'
                ) as f:
                    conversation = make_conversation(
                        input_sequence, reasoning_content, content, "None", input_len, output_len
                    )
                    f.write(conversation)

    def teach(self, episode_wrappers: List[EpisodeKTO]):
        finetune_files = []
        def kto_iter(episode_wrapper:EpisodeKTO, episode_wrapper_idx:int)->bool:  # true means no new data on this episode
            if episode_wrapper.kto_stop == False:
                ep_idx_str = f"{episode_wrapper.ep_idx}_{episode_wrapper.agent}" \
                    if episode_wrapper.agent is not None else episode_wrapper.ep_idx
                # begin to rollout kto data
                self.logger.info(f"Begin KTO iteration rollout {episode_wrapper.num_kto_iter} for episode {ep_idx_str} ...")
                filename, finished_num, promising_num, rollout_total = self.kto_iteration(
                    episode_wrapper.episode,
                    self.algorithm_config.teach_invoke_fn,
                    episode_wrapper.ep_idx,
                    self.progress_info["current_iteration"],
                    agent=episode_wrapper.agent
                )

                # check if all transactions are finished
                if rollout_total == 0:  # no finetune data
                    assert filename is None
                    self.logger.info(f"KTO stop - Reason: all transactions in episode {ep_idx_str} are finished.")
                    episode_wrapper.kto_stop = True
                    return True
                
                finetune_files.append(filename)  # TODO: remember to clear
                episode_wrapper.num_kto_iter += 1
                episode_wrapper.promising_num_list.append(promising_num)
                episode_wrapper.finished_num_list.append(finished_num)
                episode_wrapper.total_rollout_list.append(rollout_total)

                # check if reach max iteration
                if episode_wrapper.num_kto_iter == self.algorithm_config.kto_max_iteration:
                    self.logger.info(f"KTO stop - Reason: max iteration {self.algorithm_config.kto_max_iteration} reached in Episode {ep_idx_str}.")
                    episode_wrapper.kto_stop = True
                
                # early stop check
                min_len = self.algorithm_config.kto_early_stop_check_iteration
                if len(episode_wrapper.promising_num_list) >= min_len and\
                    len(episode_wrapper.finished_num_list) >= min_len and\
                    sum(episode_wrapper.promising_num_list[-min_len:]) == 0 and \
                    sum(episode_wrapper.finished_num_list[-min_len:]) == 0:
                    self.logger.info(f"KTO stop - Reason: no promising transactions in the last {self.algorithm_config.kto_early_stop_check_iteration} iterations in Episode {ep_idx_str}.")
                    episode_wrapper.kto_stop = True
                
                return False
            else:
                return True

        n_kto_iter = 0
        while True:
            self.logger.info(f"Begin KTO iteration {n_kto_iter} in Training iteration {self.progress_info['current_iteration']} ...")
            no_new_data_to_finetune = []
            for episode_wrapper_idx, episode_wrapper in enumerate(episode_wrappers):
                no_new_data_to_finetune.append(kto_iter(episode_wrapper, episode_wrapper_idx))

            if all(no_new_data_to_finetune): # all episodes no new data to finetune
                self.logger.info("All episodes no new data to finetune.")
                break
            else:
                self.logger.info(f"begin to fine-tune LLMs on data {','.join(finetune_files)}")

                ## close the LLMs
                self.logger.info("Closing LLMs...")
                self.llm_manager.stop_nodes()

                ## fine-tune the LLMs
                self.logger.info(f"Begin to fine-tune LLMs with {','.join(finetune_files)}...")
                new_model_path = self.finetune_actor(finetune_files, n_kto_iter)

                # clean up the fine-tune data
                finetune_files = []

                ## update the model path
                self.logger.info(f"Update the model path to {new_model_path}...")
                self.llm_manager.change_model_path(new_model_path)

                ## open the LLMs for next kto iteration / or next episode / next train iteration
                self.logger.info("Opening LLMs...")
                self.llm_manager.start_nodes()
                self.llm_manager.wait_route()

    def kto_iteration(self, episode: Episode, invokers:List[BaseInvoker]|BaseInvoker, ep_idx, kto_it_idx:int, agent=None):
        """"rollout kto data"""
        # make folder to record conversation
        ep_idx_str = f"{ep_idx}_{agent}" if agent is not None else ep_idx
        it = self.progress_info["current_iteration"] # train iteration
        if self.algorithm_config.record_conversation:
            os.makedirs(os.path.join(self.algorithm_config.record_conversation, f"{self.name}_iter{it}", f"kto_iter{kto_it_idx}"), exist_ok=True)
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
                    "episode_index": ep_idx_str,
                    "iteration": self.progress_info["current_iteration"],
                    "kto_tag": True
                })
            elif not good and (kto_finished==False):
                unfinished_transactions.append(trans)
            else:
                pass
        if len(unfinished_transactions) == 0:
            self.logger.info(f"All transactions in episode {ep_idx_str} are finished.")
            return None, 0, 0, 0
        else:
            self.logger.info(f"Number of unfinished transactions in episode {ep_idx_str}: {len(unfinished_transactions)}")
            for unfinished_transaction in unfinished_transactions:
                self.logger.info(f"Unfinished transaction: {unfinished_transaction['index_in_episode']}")
        if isinstance(invokers, list) and len(invokers)>1:  # multi-thread mode
            n_thread = len(invokers)
            self.logger.info(f"Got {n_thread} invokers, using multi-thread mode for KTO.")
            # separate the unfinished transactions into n_thread parts
            unfinished_transactions_lst = chunk(unfinished_transactions, n_thread)
            # invoke the transactions in parallel
            with ThreadPoolExecutor(max_workers=n_thread) as executor:
                futures = [
                    executor.submit(
                        self.kto_episode, 
                        unfinished_transactions_lst[i],
                        invokers[i],
                        ep_idx,
                        i,
                        kto_it_idx,
                        agent
                    ) for i in range(n_thread)
                ]
                res = [future.result() for future in futures]
                for r in res:
                    final_data.extend(r)
        else: # single-thread mode
            self.logger.info(f"Got 1 invoker, using single-thread mode for KTO.")
            if isinstance(invokers, list):
                teach_invoke_fn = invokers[0]
            else:
                teach_invoke_fn = invokers
            final_data = self.kto_episode(unfinished_transactions, teach_invoke_fn, ep_idx, kto_it_idx, agent)
        # log final situation in this Kto iteration
        rollout_total = 0
        finished_num = 0
        promising_num = 0
        for trans in unfinished_transactions:
            self.logger.info(f"Episode {ep_idx_str} - Step {trans['index_in_episode']} - {trans['transaction']['reflex']['reflection']['kto_remark']}")
            rollout_total += 1
            if trans["transaction"]["reflex"]["reflection"].get("kto_finished", False):
                finished_num += 1
            if trans["transaction"]["reflex"]["reflection"].get("kto_promising", False):
                promising_num += 1
        self.logger.info(f"Episode {ep_idx_str} - Kto iteration {kto_it_idx} - Finished {finished_num}/{rollout_total} transactions, {promising_num}/{rollout_total} promising transactions.")
        # save json data to file
        data_dir = os.path.join(self.algorithm_config.fine_tune_data_path, f"{self.name}_iter{it}")
        os.makedirs(data_dir, exist_ok=True) # guarantee the directory exists
        file_name = os.path.join(data_dir, f"kto_finetune_iter{kto_it_idx}_ep{ep_idx_str}.json")
        with open(file_name, 'w') as f:
            json.dump(final_data, f, indent=4, sort_keys=False, ensure_ascii=False, default=str)
        return file_name, finished_num, promising_num, rollout_total
    
    def kto_human_refine(self, episode: Episode, ep_idx, agent=None):
        ep_idx = f"{ep_idx}_{agent}" if agent is not None else ep_idx
        it = self.progress_info["current_iteration"] # train iteration
        file_name = f"{os.path.join(self.algorithm_config.fine_tune_data_path, self.name)}_kto_human_refine_iter{it}_{ep_idx}.txt"

        if not os.path.exists(file_name):
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
            with open(file_name, 'w') as f:
                f.write(s)

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
    
    def kto_episode(self, transactions: list[Transaction], invoker:BaseInvoker, ep_idx, thread_idx:int=0, kto_it_idx:int=0, agent=None):
        final_data = []
        for trans in transactions:
            data = self.kto_transaction(trans["transaction"], invoker, trans["index_in_episode"], ep_idx, thread_idx, kto_it_idx, agent)
            final_data.extend(data)
        return final_data

    def kto_transaction(self, transaction:Transaction, invoker:BaseInvoker, trans_idx:int, ep_idx, thread_idx:int=0, kto_it_idx:int=0, agent=None):
        ep_idx = f"{ep_idx}_{agent}" if agent is not None else ep_idx
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
                        conversation = make_conversation(
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

                    if rollout_statistics["good_times"] > 0:
                        reflex["reflection"]["kto_promising"] = True
                    else:
                        reflex["reflection"]["kto_promising"] = False
                    break
        return data

    def finetune_actor(self, file_names:List[str], kto_it_idx:int):
        if self.llm_manager is None:
            self.logger.warning("No LLMManager found, can't manager LLMs automatically.")
            return None
        else:
            # use the first node to finetune
            finetune_client = self.llm_manager.nodes_config[0]["client"]

            # upload the fine-tune data
            self.logger.info("Uploading fine-tune data...")
            save_paths = []
            for file_name in file_names:
                filename = os.path.basename(file_name)
                save_path = f"data/{self.algorithm_config.run_name}_it{self.progress_info['current_iteration']}/{filename}"
                save_paths.append(save_path)
                self.logger.info(f"Uploading {file_name} to {save_path}...")
                finetune_client.upload_data_file(file_name, save_path, timeout=1800)

            # upload example configs
            train_config_path = f"examples/train_lora/{self.algorithm_config.run_name}_train_config.yaml"
            merge_config_path = f"examples/merge_lora/{self.algorithm_config.run_name}_merge_config.yaml"

            if self.progress_info["current_iteration"] == 0:
                self.logger.info("Uploading example configs...")
                self.logger.info(f"Uploading {self.algorithm_config.initial_merge_config_path} to {train_config_path}")
                finetune_client.upload_data_file(
                    self.algorithm_config.initial_train_config_path,
                    train_config_path,
                    timeout=1800
                )
                self.logger.info(f"Uploading {self.algorithm_config.initial_merge_config_path} to {merge_config_path}")
                finetune_client.upload_data_file(
                    self.algorithm_config.initial_merge_config_path,
                    merge_config_path,
                    timeout=1800
                )

            # update configs
            ## update dataset.json
            dataset_config = {
                f"kto_finetune_{i}": {
                    "file_name": p.lstrip("data/"),
                    "formatting": "sharegpt",
                    "columns": {
                    "messages": "conversations",
                    "kto_tag": "kto_tag"
                    }
                } for i, p in enumerate(save_paths)
            }
            self.logger.info(f"Updating dataset_info.json")
            finetune_client.update_json_config(
                "data/dataset_info.json",
                dataset_config
            )
            if len(dataset_config) > 1:
                dataset_str = ",".join(dataset_config.keys())
            else:
                dataset_str = list(dataset_config.keys())[0]

            ## update train config
            output_dir = f"saves/{self.algorithm_config.run_name}_train_it{self.progress_info['current_iteration']}_kto_it{kto_it_idx}"
            train_config = {
                "model_name_or_path": self.current_model_path,
                "dataset": dataset_str,
                "output_dir": output_dir,
                "run_name": self.algorithm_config.run_name,
            }
            finetune_client.update_yaml_config(
                train_config_path,
                train_config
            )

            ## update merge config
            export_dir = f"output/{self.algorithm_config.run_name}_train_it{self.progress_info['current_iteration']}_kto_it{kto_it_idx}"
            merge_config = {
                "model_name_or_path": self.current_model_path,
                "adapter_name_or_path": output_dir,
                "export_dir": export_dir,
            }
            finetune_client.update_yaml_config(
                merge_config_path,
                merge_config
            )

            # get the configs
            train_config_check = finetune_client.get_config(train_config_path)
            self.logger.info(f"Train config: {train_config_check}")
            merge_config_check = finetune_client.get_config(merge_config_path)
            self.logger.info(f"Merge config: {merge_config_check}")

            # fine-tune the LLMs
            self.logger.info("Fine-tuning LLMs...")
            output_file = os.path.join(
                self.algorithm_config.record_conversation,
                f"{self.name}_iter{self.progress_info['current_iteration']}",
                f"finetune_log_train_it{self.progress_info['current_iteration']}_kto_it{kto_it_idx}.log"
            )
            finetune_client.run_command(
                f"FORCE_TORCHRUN=1 llamafactory-cli train {train_config_path} && llamafactory-cli export {merge_config_path}",
                process_id = f"finetune_{self.algorithm_config.run_name}_it{self.progress_info['current_iteration']}",
                output_file = output_file,
            )

            return export_dir

    # ----------------- LLM algorithm specific functions -----------------
    
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