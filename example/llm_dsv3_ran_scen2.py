from mpi_rl import config
config.use_mpi = False

from .my_secret_keys import volcano_api_key, bailian_api_key # use your own API keys

from mpi_rl.mpi_rl.algorithm.llm.llm_algo import LLM, LLMConfig
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import HuggingfaceTokenCounter, GeneralRequestInvoker, llm_action_filter, DoubleInvoker
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils.replay_buffer import ReplayBuffer, Episode, Transaction
from mpi_rl.utils import Pipeline

from textwrap import dedent
from copy import deepcopy
from pathlib import Path
import os

from ransim_env.envs import scenario_two
from .prj_utils import path_allocator


exp_folder = path_allocator("exps/exp", "")
exp_folder = Path(exp_folder).resolve().as_posix()
print("Experiment result path:", exp_folder)

def check_action(llm_action):
    assert llm_action["action"] in range(6, 41)
    return llm_action

def substract_6(llm_action):
    llm_action["action"] = llm_action["action"]-6
    return llm_action

def get_policy(llm_action):
    print(llm_action)
    return llm_action["policy"]

llm_action_filter_check = Pipeline(llm_action_filter, check_action, substract_6)
llm_reflection_filter_check = Pipeline(llm_action_filter, get_policy)

volcano_v3_reflection_api_invoker = GeneralRequestInvoker(
    full_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    API_key = volcano_api_key,
    model_name="ep-20250212231007-4dm2m", # use your own endpoint
    content_filter_fn=llm_reflection_filter_check
)

volcano_distill_qwen_reflection_api_invoker = GeneralRequestInvoker(
    full_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    API_key = volcano_api_key,
    model_name="ep-20250218021435-88j95", # use your own endpoint
    content_filter_fn=llm_reflection_filter_check
)

bailian_v3_api_reflection_invoker = GeneralRequestInvoker(
    full_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    API_key = bailian_api_key,
    model_name="deepseek-v3",
    content_filter_fn=llm_reflection_filter_check
)

volcano_v3_action_api_invoker = GeneralRequestInvoker(
    full_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    API_key = volcano_api_key,
    model_name="ep-20250212231007-4dm2m", # use your own endpoint
    content_filter_fn = llm_action_filter_check
)

volcano_distill_qwen_action_api_invoker = GeneralRequestInvoker(
    full_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    API_key = volcano_api_key,
    model_name="ep-20250218021435-88j95", # use your own endpoint
    content_filter_fn = llm_action_filter_check
)

bailian_v3_api_action_invoker = GeneralRequestInvoker(
    full_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    API_key = bailian_api_key,
    model_name="deepseek-v3",
    content_filter_fn = llm_action_filter_check
)

double_guarantee_action_invoker = DoubleInvoker(volcano_v3_action_api_invoker, bailian_v3_api_action_invoker)
double_guarantee_reflection_invoker = DoubleInvoker(volcano_v3_reflection_api_invoker, bailian_v3_api_reflection_invoker)

huggingface_counter = HuggingfaceTokenCounter(model_name="deepseek-ai/DeepSeek-V3", bias=5)

system_prompt = dedent("""
    你作为RAN网络切片的智能无线电资源控制器,需要通过动态调整资源分配来优化网络性能。

    观测空间为长度为5的向量,写作O,含义为：
    [当前资源分配, 当前频谱效率, 当前流量大小, 当前排队包变化量, 当前丢包量]
                       
    目标包括三个:
    1.最大化频谱效率, 频谱效率最好能在4到6之间
    2.最小化资源调整频率
    3.最小化SLA违约次数
    三个目标的优先级"最小化SLA违约次数">"最大化频谱效率">"最小化资源调整频率"
                       
    单步性能指标是长度为3的向量,写作M,含义为:
    [频谱效率, 是否违反SLA, 资源数是否改变]
    "是否违反SLA"和"资源数是否改变"用0与1表示,0代表没有违反SLA或者资源数没有改变,反之则违反了SLA或者资源数改变了。
    刚开始时只提供一些观测,没有完整的历史信息。
    性能指标与目标一一对应,即M[0]对应"最大化频谱效率",M[1]对应"最小化SLA违约次数",M[2]对应"最小化资源调整频率"。

    动作空间为6到40的整数,动作即为资源分配数,写作A。
                       
    角标"_t"代表当前时间步,"_{t+1}"代表下一个时间步,"_{t-1}"代表上一个时间步。
                       
    历史信息有多行,每一行为时间步t的完整信息,写作:
    [O_t, A_t, M_{t+1}, 对O_t的分析(A_t的理由), 对A_{t-1}的反思(即上一行历史信息的分反思)]
                       
    每一完整Episode会有一个Episode级的性能指标,用于评估整个Episode的性能。
    Episode级性能指标包括: 平均频谱效率, 总SLA违约次数, 总资源调整次数

    一些调整经验:
    1.这三个目标是相互冲突的,频谱效率过高会导致SLA违约。为了不SLA违约,过于谨慎地过多分配资源的话频谱效率会很低。尽量减少资源调整次数的话,即在不合适的资源数上保持,会导致频谱效率低或者SLA违约。
    2.在历史经验不足的情况下可以大胆探索,不要担心违反目标,大幅度增减(比如增减10左右)资源来探索环境。
    3.在频谱效率不错且没有违反SLA的情况下,小幅度流量变化可以不变化资源以最小化资源调整次数
    4.如果到资源调整到最低了还频谱效率很低,那也就保持最低。
    5.频谱效率与流量和分配的资源有关,频谱效率和"流量"成正相关,与"当前资源分配"成负相关。
    6.如果频谱效率很低,应该大幅度减少资源来提升频谱效率。

    每次生成动作时要返回一个json给我,json结构为:
    {
    "prev_step_reflection": "反思之前一系列的动作做得好还是坏,原因是什么,反思在类似上一步的情况中,如何做到更好,让各个性能指标更佳",
    "analysis": "为什么这样分配",
    "action": int,
    }
                       
    每次反思生成策略时要返回一个json给我,json结构为:
    {
    "analysis": "分析所给的Episode历史及性能指标,分析在这个Episode中,哪些动作做得好,哪些做得不好,如何做到更好,让各个性能指标更佳",
    "policy": "详细地描述改进后的新策略,比如在什么情况下选择什么动作,如何选择动作.把你学到的东西总结为一个策略,让你的策略在未来的Episode中表现更好"                
    }
    """)

def metrics_statistics(ep:Episode):
    num_of_reconf = sum([t["transaction"]["info"]["changed_RB"] for t in ep])
    num_of_violation = sum([t["transaction"]["info"]["SLA_violation"] for t in ep])
    avg_SE = sum([t["transaction"]["info"]["SE"] for t in ep])/len(ep)
    return {"num of reconfiguration": num_of_reconf, "num of SLA violation": num_of_violation, "average SE": avg_SE}

def process_transaction(transaction: Transaction):
    obs = list(transaction["obs"][0:5])
    action = transaction["action"]+6
    metrics_SE = transaction["next_obs"][1]
    metrics_SLA = transaction["next_obs"][5]
    metrics_RB = transaction["next_obs"][6]
    # analysis = transaction.get("analysis", "None")
    # reflection = transaction.get("prev_step_reflection", "None")
    # return [obs, action, [metrics_SE, metrics_SLA, metrics_RB], analysis, reflection]
    return [obs, action, [metrics_SE, metrics_SLA, metrics_RB]]

def process_obs(obs):
    return list(obs[0:5])

llm_config = LLMConfig(
    system_prompt = system_prompt,
    action_invoke_fn = double_guarantee_action_invoker,
    reflection_invoke_fn = double_guarantee_reflection_invoker,
    metrics_statistics_fn = metrics_statistics,
    model_tokens_counter_fn = huggingface_counter,
    max_tokens_input = 64000, # 64k tokens
    gen_markdown = os.path.join(exp_folder, "markdowns"),
    transaction_process_fn = process_transaction,
    obs_process_fn = process_obs
)

episode_len = 300

def train_env_scen2_creator():
    return scenario_two.ScenarioTwo(
        f"train", episode_len,
        unchange_terminate = 0,
        root_path=os.path.join(exp_folder, "train_logs"),
        no_termi=True,
        plot = False,
        num_obs_history=1,
        warmup_times = 75
    )

def eval_env_scen2_creator():
    return scenario_two.ScenarioTwo(
        f"eval", episode_len,
        unchange_terminate = 0,
        root_path=os.path.join(exp_folder, "eval_logs"),
        no_termi=True,
        plot = True,
        num_obs_history=1,
        warmup_times = 75
    )

train_rollout_config = RolloutConfig(
    count_by = "episode",
    explore = True,          # ignored in LLM algorithm
    num_of_episode = 1,
)
eval_rollout_config = deepcopy(train_rollout_config)

replay = ReplayBuffer(30000)

llm_algo = LLM(
    train_env_creator = train_env_scen2_creator,
    train_rollout_config = train_rollout_config,
    eval_env_creator = eval_env_scen2_creator,
    eval_rollout_config = eval_rollout_config,
    replay_buffer = replay,
    algorithm_config = llm_config,
    multi_agent = False
)

llm_algo.logger.setLevel("INFO")

for _ in range(10):
    llm_algo.train_iteration()