from mpi_rl import config
config.use_mpi = False

from my_secret_keys import volcano_api_key, bailian_api_key, ds_api_key # use your own API keys
from my_secret_keys import volcano_r1_name

from mpi_rl.mpi_rl.algorithm.llm.llm_algo import LLM, LLMConfig
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import HuggingfaceTokenCounter, GeneralRequestInvoker, json_filter, RoundRobinInvoker
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils.replay_buffer import ReplayBuffer, Episode, Transaction
from mpi_rl.utils import Pipeline

from copy import deepcopy
from pathlib import Path
import os

from ransim_env.envs import scenario_two
from prj_utils import path_allocator


exp_folder = path_allocator("exps/exp", "")
exp_folder = Path(exp_folder).resolve().as_posix()
print("Experiment result path:", exp_folder)

def check_action(llm_action):
    assert llm_action["action"] in range(6, 41)
    return llm_action

def substract_6(llm_action):
    llm_action["action"] = llm_action["action"]-6
    return llm_action

llm_action_filter_check = Pipeline(json_filter, check_action, substract_6)

ds_invoker_1 = GeneralRequestInvoker( # ds official
    full_url="https://api.deepseek.com/v1/chat/completions",
    model_name="deepseek-reasoner",
    API_key=ds_api_key,
    content_filter_fn=json_filter
)

ds_invoker_2 = GeneralRequestInvoker( # volcano
    full_url="https://ark.cn-beijing.volces.com/api/v3/",
    model_name=volcano_r1_name,
    API_key=volcano_api_key,
    content_filter_fn=json_filter
)

ds_invoker_3 = GeneralRequestInvoker( # bailian
    full_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    model_name="deepseek-r1",
    API_key=bailian_api_key,
    content_filter_fn=json_filter
)

reflex_invoker = RoundRobinInvoker(ds_invoker_1, ds_invoker_2, ds_invoker_3, weights=[4, 2, 1], try_round=2, sleep_btw_round=300)

act_invoker = GeneralRequestInvoker(
    full_url="localhost:9000/v1/chat/completions",
    model_name="Qwen7B",
    content_filter_fn=llm_action_filter_check
)

qwen_counter = HuggingfaceTokenCounter(model_name="Qwen/Qwen2.5-7B", bias=5)
deepseek_counter = HuggingfaceTokenCounter(model_name="deepseek-ai/DeepSeek-R1", bias=5)

system_desc = """\
# 网络切片资源控制任务

你作为RAN网络切片的智能控制器,通过动态调整资源分配(6-40的整数)优化以下三个目标：
1.最小化SLA违约次数(权重最高)
2.保持较高的频谱效率在, 理想为4-6之间(权重中等)
3.减少资源调整次数(权重最低)

## 动作空间

动作空间为6到40的整数, 代表为切片分配的资源数量, 数值越高代表分配的资源越多。

## 观测与指标规范

### 观测空间(O_t)

[当前资源数, 当前频谱效率, 当前流量大小, 排队包变化量, 丢包量]

### 性能指标(M_t)

[频谱效率值, SLA违约标记(0/1), 资源变更标记(0/1)]

性能指标与目标一一对应,即M[0]对应"最大化频谱效率",M[1]对应"最小化SLA违约次数",M[2]对应"最小化资源调整频率"。
1代表SLA违约或者资源变更了。

## 历史记录格式

每行历史包含：
[O_t, A_t, M_t, 即时分析, 上步反思]
注: _t表示当前步, _{t±n}表示时间偏移。

## 专家经验库

1. 难点:
   - 这三个目标是相互冲突的,频谱效率过高会导致SLA违约。为了不SLA违约,过于谨慎地过多分配资源的话频谱效率会很低。尽量减少资源调整次数的话,即在不合适的资源数上保持,会导致频谱效率低或者SLA违约。所以要总结经验, 以学会最合适的资源分配策略。

2. 资源分配法则:
   - 当频谱效率<4时: 大幅减少资源(Δ≥5)
   - 频谱效率与流量和分配的资源有关,频谱效率和"流量"成正相关,与"当前资源分配"成负相关。

3. 探索策略
   - 在历史经验不足的情况下可以大胆探索,不要担心违反目标,大幅度增减(比如增减10左右)资源来探索环境。

4. 异常处理:
   - 资源已达下限仍低效：保持最低资源数。
   
"""

act_prompt = """\
根据以上信息自, 生成这一步的动作, 严格按照以下格式分析当前情况并生成动作:

当前[是/不是]第一步

{% if not first_step %}
## 上一步反思

上一步的情况为:
- O_{t-1}: [值列表]
- A_{t-1}: [整数]
- M_{t-1}: [值列表] 
分析要点：
1. 根据M_t判断是否达成各目标
2. 提出下一步改进方向

(分析时, 主语用"这一步", 不要用"上一步")
{% endif %}

## 当前观测分析

显示当前O_t: [值列表]
分析逻辑链：
1. 流量变化趋势
2. 频谱效率状态
3. SLA风险预测
4. 资源调整必要性论证

## 最终结果

生成严格遵循此结构的JSON:
{
"prev_step_retrospection": [对上一步反思的总结], 
"analysis": [对当前观测的分析],
"action": [这一步决定采取的动作, 6-40的整数]
}"""

reflex_prompt = """\
结合上面完整的episode信息, 逐条分析其中用<<<< >>>包裹的片段Z:

先声明片段Z中一共有多少条历史

然后对每一条历史执行:

1. 声明这是Z中的第几条历史记录
2. 重复这条历史记录的O, A和M
3. 分析这条历史记录中的动作好坏, 有什么优点和缺陷
4. 分析在这一步的O下, 是否需要换另一个动作来达到更好的效果
5. 给出应该更换的动作

然后输出最终的json格式化结果, 要严格按照下面格式, 严格按照Z中历史记录的顺序生成:

[
  {
   "observation": [重复这一步的观测O]
   "original_action": [重复这一步的动作A],
   "retrospection": {
     "advantage": [决策优点(若无可省略)],
     "flaw": [主要缺陷分析(若无可省略)],
     "suggested_action": [改进后的动作]
   }
  },
  ...
]

retrospection.advantage和retrospection.flaw要尽可能第详细
"""

def metrics_statistics(ep:Episode):
    num_of_reconf = sum([t["transaction"]["info"]["changed_RB"] for t in ep])
    num_of_violation = sum([t["transaction"]["info"]["SLA_violation"] for t in ep])
    avg_SE = sum([t["transaction"]["info"]["SE"] for t in ep])/len(ep)
    return {"num of total reconfiguration": num_of_reconf, "num of total SLA violation": num_of_violation, "average SE": avg_SE}

def process_transaction(transaction: Transaction):
    obs = list(transaction["obs"][0:5])
    action = transaction["action"]+6
    metrics_SE = transaction["next_obs"][1]
    metrics_SLA = transaction["next_obs"][5]
    metrics_RB = transaction["next_obs"][6]
    analysis = transaction.get("analysis", "None")
    reflection = transaction.get("prev_step_retrospection", "None")
    return [obs, action, [metrics_SE, metrics_SLA, metrics_RB], analysis, reflection]

def process_obs(obs):
    return list(obs[0:5])

llm_config = LLMConfig(
    system_description=system_desc,
    act_prompt=act_prompt,
    reflex_prompt=reflex_prompt,
    act_invoke_fn=act_invoker,
    reflex_invoke_fn=reflex_invoker,
    metrics_statistics_fn=metrics_statistics,
    reflex_model_tokens_counter_fn=deepseek_counter,
    act_model_tokens_counter_fn=qwen_counter,
    reflex_max_tokens_input=64000, # 64k tokens
    act_max_tokens_input=64000, # 64k tokens
    transaction_process_fn = process_transaction,
    obs_process_fn = process_obs,
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

for _ in range(2):
    llm_algo.train_iteration()