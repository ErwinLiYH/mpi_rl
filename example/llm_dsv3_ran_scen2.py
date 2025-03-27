from mpi_rl import config
config.use_mpi = False

from my_secret_keys import volcano_api_key, bailian_api_key, ds_api_key, openrouter_api_key # use your own API keys
from my_secret_keys import volcano_r1_name

from mpi_rl.mpi_rl.algorithm.llm.llm_algo import LLM, LLMConfig
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import HuggingfaceTokenCounter, GeneralRequestInvoker, RoundRobinInvoker, json_filter, code_filter, json_filter2
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils.replay_buffer import ReplayBuffer, Episode, Transaction
from mpi_rl.utils import Pipeline, advanced_retry

from copy import deepcopy
from pathlib import Path
import os

from ransim_env.envs import scenario_two
from prj_utils import path_allocator

from Kkit import init_logger, PathJoin


exp_folder = path_allocator("exps/exp", "")
exp_folder = Path(exp_folder).resolve().as_posix()
print("Experiment result path:", exp_folder)

prj_path_join = PathJoin(exp_folder)

def check_action(llm_action):
    assert llm_action["action"] in range(6, 41)
    return llm_action

def substract_6(llm_action):
    llm_action["action"] = llm_action["action"]-6
    return llm_action

reflex_batch = 5

def check_reflex_length(reflex_action):
    assert len(reflex_action) == reflex_batch
    return reflex_action

llm_action_filter_check = Pipeline(code_filter, json_filter2, check_action, substract_6)
llm_reflex_filter_check = Pipeline(code_filter, json_filter2, check_reflex_length)

ds_invoker_1 = GeneralRequestInvoker( # ds official
    full_url="https://api.deepseek.com/v1/chat/completions",
    model_name="deepseek-reasoner",
    API_key=ds_api_key,
    content_filter_fn=llm_reflex_filter_check
)

ds_invoker_2 = GeneralRequestInvoker( # volcano
    full_url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    model_name=volcano_r1_name,
    API_key=volcano_api_key,
    content_filter_fn=llm_reflex_filter_check
)

ds_invoker_3 = GeneralRequestInvoker( # bailian
    full_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    model_name="deepseek-r1",
    API_key=bailian_api_key,
    content_filter_fn=llm_reflex_filter_check
)

roundrobin_logger = init_logger("roundrobin", prj_path_join("roundrobin.log"))

reflex_invoker = RoundRobinInvoker(ds_invoker_1, ds_invoker_2, ds_invoker_3, weights=[2, 1, 1], try_round=10, sleep_btw_round=300, logger=roundrobin_logger)

act_invoker = GeneralRequestInvoker(
    full_url="http://localhost:9000/v1/chat/completions",
    model_name="Qwen14B",
    content_filter_fn=llm_action_filter_check
)

qwq32b_invoker_1 = GeneralRequestInvoker(
    full_url="https://openrouter.ai/api/v1/chat/completions",
    model_name="qwen/qwq-32b:free",
    API_key=openrouter_api_key,
    reasoning_key="reasoning",
    content_filter_fn=llm_action_filter_check,
    top_k=40,
    top_p=0.95,
    temperature=0.6,
)

qwq32b_invoker_2 = GeneralRequestInvoker(
    full_url="http://localhost:9000/v1/chat/completions",
    model_name="Qwen7B",
    content_filter_fn=llm_action_filter_check,
    top_k=40,
    top_p=0.95,
    temperature=0.6,
)

qwen_counter = HuggingfaceTokenCounter(model_name="Qwen/Qwen2.5-7B", bias=5)
deepseek_counter = HuggingfaceTokenCounter(model_name="deepseek-ai/DeepSeek-R1", bias=5)

system_desc = """\
# 网络切片资源控制任务

这是一个网络切片资源控制的任务，将有3个Agent合作优化资源分配，一个是actor，负责接收当前观测和历史记录，生成动作，即分配的资源；一个是reflector，接收完整的episode记录，结合完整episode反思每一步的动作优劣，并给出更好的动作；最后一个是teacher，这个Agent结合reflector对每一步的反思，改写actor的思考过程，引导actor生成reflector修改过的动作，以优化资源控制。

在RAN网络切片的资源分配任务中,Actor需要通过动态调整资源分配(6-40的整数)优化以下三个目标：
1.最小化SLA违约次数(权重最高)
2.保持较高的频谱效率在, 理想为5-6之间(权重中等)
3.减少资源调整次数(权重最低)

## 动作空间

动作空间即为分配资源数的合法空间,动作空间为6到40的整数,代表为切片分配的资源数量,数值越高代表分配的资源越多。

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
注: _t表示当前步, _{{t±n}}表示时间偏移。

## 专家经验库

1. 难点:
   - 这三个目标是相互冲突的,频谱效率过高会导致SLA违约。为了不SLA违约,过于谨慎地过多分配资源的话频谱效率会很低。尽量减少资源调整次数的话,即在不合适的资源数上保持,会导致频谱效率低或者SLA违约。所以要总结经验, 以学会最合适的资源分配策略。
2. 资源分配经验:
   - 当频谱效率小于4时: 大幅减少资源
   - 频谱效率与流量和分配的资源有关,频谱效率和"流量"成正相关,与"当前资源分配"成负相关。
   - 频谱效率和(流量/资源数)的比值成正比。
   - 当流量增大时,资源分配要增大。当流量减小时,资源分配要减小。
   - 频谱效率过低时,为了提高频谱效率,资源分配要减小来提升频谱效率。当频谱效率过高时,资源分配要增大以减小频谱效率。
   - 要结合流量大小和频谱效率来调整资源分配。
   - 资源分配要符合动作空间的范围,不要超出范围。
3. 探索策略
   - 在历史经验不足的情况下可以大胆探索,不要担心违反目标,大幅度增减资源来探索环境。
4. 异常处理:
   - 如果资源已达下限6了,仍然频谱效率低：保持最低资源数,6。
    - 如果资源已达上限40了,仍然SLA违约：保持最高资源数,40。
"""

act_prompt = """\
你是Actor, 根据以上信息, 生成这一步的动作, 严格按照以下格式分析当前情况并生成动作:

当前[是/不是]第一步
(如果历史步骤为None, 则为第一步)

{{% if not first_step %}}
## 上一步反思

上一步的情况为:
- O_{{t-1}}: [值列表]
- A_{{t-1}}: [整数]
- M_{{t-1}}: [值列表] 
分析要点：
1. 根据M_t判断是否达成各目标
2. 为什么无法达成目标
3. 提出下一步改进方向

{{% endif %}}

## 当前观测分析

显示当前O_t: [值列表]

需要分析的点：
1. 结合历史记录,流量变化趋势是什么?
2. 频谱效率状态
3. SLA风险预测
4. 相对上一步是否应该改变资源数?增大还是减小,或者保持不变?

## 最终结果

生成严格遵循此结构的JSON,并且动作要严格按照动作空间:
```json
{{
"prev_step_reflection": [对上一步反思的总结], 
"analysis": [对当前观测的分析],
"action": [这一步决定采取的动作, 6-40的整数,不要超过这个区间]
}}
```"""

reflex_prompt = """\
你是reflector, 结合上面完整的episode信息, 逐条分析其中用<<<< >>>包裹的片段Z:

先声明片段Z中一共有多少条历史

然后对每一条历史执行:

1. 声明这是Z中的第几条历史记录
2. 重复这条历史记录的O, A和M
3. 结合完整Episode的信息,分析完整Episode的流量变化趋势,流量是否在某一区间波动,在某一步大幅度减小或者增大
4. 分析在这一步的情况下流量变化趋势, 频谱效率状态, SLA风险预测, 资源调整必要性
5. 分析这条历史记录中的动作好坏, 有什么优点和缺陷
6. 分析在这一步的O下, 是否需要换另一个动作来达到更好的效果
7. 分析这一步相对上一步是否要改变资源分配,增大还是减小,或者保持不变
8. 给出应该更换的动作

PS: 对每条历史反思时,都要结合整个Episode的信息,比如分析结合前后的流量变化趋势,分析是否可以让前后的资源分配都保持一致并且不会SLA违约且频谱效率高
    每行历史为[O_t, A_t, M_t],没有即时分析和上步反思,只有基础的历史记录

然后输出最终的json格式化结果, 要严格按照下面格式, 严格按照Z中历史记录的顺序生成:

```json
[
  {{
   "observation": [重复这一步的观测O]
   "original_action": [重复这一步的动作A],
   "reflection": {{
     "good": [是否是足够好的动作,布尔值,True/False],
     "analysis": [分析在这一步的情况下流量变化趋势, 频谱效率状态, SLA风险预测, 资源调整必要性],
     "advantage": [决策优点(若无可省略)],
     "flaw": [主要缺陷分析(若无可省略)],
     "suggested_action": [改进后的动作]
   }}
  }},
  ...
]
```

reflection.advantage和reflection.flaw要尽可能地详细
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
    reflection = transaction.get("pre_step_reflection", "None")
    return [obs, action, [metrics_SE, metrics_SLA, metrics_RB], analysis, reflection]

def process_transaction_reflex(transaction: Transaction):
    obs = list(transaction["obs"][0:5])
    action = transaction["action"]+6
    metrics_SE = transaction["next_obs"][1]
    metrics_SLA = transaction["next_obs"][5]
    metrics_RB = transaction["next_obs"][6]
    return [obs, action, [metrics_SE, metrics_SLA, metrics_RB]]

def process_obs(obs):
    return list(obs[0:5])

llm_config = LLMConfig(
    system_description=system_desc,
    act_prompt=act_prompt,
    reflex_prompt=reflex_prompt,
    act_invoke_fn=lambda x: advanced_retry(qwq32b_invoker_2, 5, 3, messages=x),
    reflex_invoke_fn=reflex_invoker,
    metrics_statistics_fn=metrics_statistics,
    reflex_model_tokens_counter_fn=deepseek_counter,
    act_model_tokens_counter_fn=qwen_counter,
    reflex_max_tokens_input=64000, # 64k tokens
    act_max_tokens_input=64000, # 64k tokens
    transaction_process_fn = process_transaction,
    obs_process_fn = process_obs,
    record_conversation = exp_folder,
    reflex_specific_transaction_process_fn=process_transaction_reflex,
    reflex_batch=reflex_batch
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