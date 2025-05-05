from mpi_rl import config
config.use_mpi = False

from my_secret_keys import volcano_api_key, bailian_api_key, ds_api_key, openrouter_api_key # use your own API keys
from my_secret_keys import volcano_r1_name

from mpi_rl.mpi_rl.algorithm.llm.llm_algo import LLM, LLMConfig
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import (
        HuggingfaceTokenCounter, 
        GeneralRequestInvoker, 
        RoundRobinInvoker, 
        code_filter_multi, 
        code_filter, 
        json_filter2
    )
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils.replay_buffer import ReplayBuffer, Episode, Transaction
from mpi_rl.utils import Pipeline, advanced_retry

from copy import deepcopy
from pathlib import Path
import os
import sys
import pickle

from ransim_env.envs import scenario_two
from prj_utils import path_allocator

from Kkit import init_logger, PathJoin
from Kkit.port_control import PortController


if len(sys.argv) == 2:
    exp_folder = sys.argv[1]
elif len(sys.argv) == 1:
    exp_folder = path_allocator("exps/exp", "")
    exp_folder = Path(exp_folder).resolve().as_posix()
print("Experiment result path:", exp_folder)

prj_path_join = PathJoin(exp_folder)

def check_action(llm_action):
    assert llm_action["action"] in range(6, 41), f"action should be in range(6, 41)\n{llm_action}"
    return llm_action

reflex_batch = 2

def check_reflex_length(reflex_action):
    assert len(reflex_action) == reflex_batch, f"reflex_action should have {reflex_batch} elements\n{reflex_action}"
    return reflex_action

def teach_check(teach_action):
    assert len(teach_action) == 2, f"teach_action should have 2 elements\n{teach_action}"
    # these assertions are from the teach_prompt
    assert teach_action[0]["language"] == "text", f"teach_action[0] should be text\n{teach_action[0]}"
    assert teach_action[1]["language"] == "json", f"teach_action[1] should be json\n{teach_action[1]}"
    return teach_action

llm_action_filter_check = Pipeline(code_filter, json_filter2, check_action)
llm_reflex_filter_check = Pipeline(code_filter, json_filter2, check_reflex_length)
llm_teach_filter_check = Pipeline(code_filter_multi, teach_check)
llm_teach2_filter_check = Pipeline(code_filter, json_filter2, check_action)

ds_invoker_1 = GeneralRequestInvoker( # ds official
    full_url="https://api.deepseek.com/v1/chat/completions",
    model_name="deepseek-reasoner",
    API_key=ds_api_key,
    content_filter_fn=llm_reflex_filter_check
)
ds_invoker_1_retry = lambda x: advanced_retry(ds_invoker_1, 5, 60, messages=x)

ds_invoker_2 = GeneralRequestInvoker( # volcano
    full_url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    model_name=volcano_r1_name,
    API_key=volcano_api_key,
    content_filter_fn=llm_reflex_filter_check
)
ds_invoker_2_retry = lambda x: advanced_retry(ds_invoker_2, 5, 60, messages=x)

ds_invoker_3 = GeneralRequestInvoker( # bailian
    full_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    model_name="deepseek-r1",
    API_key=bailian_api_key,
    content_filter_fn=llm_reflex_filter_check
)
ds_invoker_3_retry = lambda x: advanced_retry(ds_invoker_3, 5, 60, messages=x)

roundrobin_logger = init_logger("roundrobin", prj_path_join("roundrobin.log"))

reflex_invoker = RoundRobinInvoker(ds_invoker_1_retry, ds_invoker_2_retry, weights=[1, 1], try_round=10, sleep_btw_round=300, logger=roundrobin_logger)

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

# Init local invokers
original_model_name = "Qwen3-8B"
trained_model_name = "QwQ_1"
original_port_controller = PortController(name=original_model_name)
# trained_port_controller = PortController(name=trained_model_name)
original_ports = original_port_controller.allocate_port(1)
# trained_port = trained_port_controller.allocate_port()

act_port = original_ports[0]
reflex_port = original_ports[0]
teach_port = original_ports

print(f"original model port: {original_ports}")
# print(f"trained model port: {trained_port}")

print(f"act_port: {act_port}")
print(f"reflex_port: {reflex_port}")
print(f"teach_port: {teach_port}")

qwq32b_invoker_act = GeneralRequestInvoker(
    full_url=f"http://localhost:{act_port}/v1/chat/completions",
    model_name=original_model_name,
    content_filter_fn=llm_action_filter_check,
    top_k=40,
    top_p=0.95,
    temperature=0.6,
)

qwq32b_invoker_teach = GeneralRequestInvoker(
    full_url=f"http://localhost:{teach_port}/v1/chat/completions",
    model_name="QwQ_ori",
    content_filter_fn=llm_teach_filter_check,
    top_k=40,
    top_p=0.95,
    temperature=0.6,
)

qwq32b_invoker_reflex = GeneralRequestInvoker(
    full_url=f"http://localhost:{reflex_port}/v1/chat/completions",
    model_name="QwQ_ori",
    content_filter_fn=llm_reflex_filter_check,
    top_k=40,
    top_p=0.95,
    temperature=0.6,
)

def make_retry_lambda(invoker):
    return lambda x: advanced_retry(invoker, 10, 3, messages=x)

qwq32b_invokers_teach2 = [
    GeneralRequestInvoker(
        full_url=f"http://localhost:{port}/v1/chat/completions",
        model_name=original_model_name,
        content_filter_fn=llm_teach2_filter_check,
        top_k=40,
        top_p=0.95,
        temperature=0.6,
    ) for port in teach_port
]
qwq32b_invokers_teach2_retry = [make_retry_lambda(invoker) for invoker in qwq32b_invokers_teach2]

qwen_counter = HuggingfaceTokenCounter(model_name="Qwen/Qwen2.5-7B", bias=5)
deepseek_counter = HuggingfaceTokenCounter(model_name="deepseek-ai/DeepSeek-R1", bias=5)

system_desc = """\
# 网络切片资源控制任务

这是一个网络切片资源控制的任务，将有3个Agent合作优化资源分配，一个是actor，负责接收当前观测和历史记录，生成动作，即分配的资源；一个是reflector，接收完整的episode记录，结合完整episode反思每一步的动作优劣，并给出更好的动作；最后一个是teacher，这个Agent结合reflector对每一步的反思，改写actor的思考过程，引导actor生成reflector修改过的动作，以优化资源控制。

在RAN网络切片的资源分配任务中,Actor需要通过动态调整资源分配(6-40的整数)优化以下三个目标：
1.最小化SLA违约次数(权重最高)
2.保持较高的频谱效率在, 理想为5-6.5之间(权重中等)
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
[O_t, A_{{t+1}}, M_{{t+1}}, 即时分析, 上步反思]
注: _t表示当前步, _{{t±n}}表示时间偏移。

这里的马尔科夫过程为，Actor获取O_t，输出下一步的动作A_{{t+1}}，然后环境运行A_{{t+1}}并统计出{{t+1}}步中的Metrics M_{{t+1}}。下一步的观测O_{{t+1}}会包含动作A_{{t+1}}，和部分M{{t+1}}的值。O{{t+1}}里的当前资源数就是A_{{t+1}}，当前频谱效率是M_{{t+1}}[0]。

## 供参考的经验库

1. 难点:
   - 这三个目标是相互冲突的,频谱效率过高会导致SLA违约。为了不SLA违约,过于谨慎地过多分配资源的话频谱效率会很低。尽量减少资源调整次数的话,即在不合适的资源数上保持,会导致频谱效率低或者SLA违约。所以要总结经验, 以学会最合适的资源分配策略。
2. 资源分配经验:
   - 频谱效率等于 5.55*(流量/资源数)。
   - 资源分配要符合动作空间的范围,不要超出范围。
   - 尽量少调整资源，减少资源调整次数，但是要保证频谱效率和SLA违约次数都在合理范围内。
   - 不必在意频谱效率是否在理想范围的正中间，而且略微超出范围也可以接受。
   - 在资源分配不变的情况下，排队包变化量较大时（一般为±4以上），代表下一步流量可能会有较大变化，并且与当前资源数不匹配，可以考虑增减资源数。
   - 排队包变化量也不完全代表流量变化，在流量不变的情况下，增减资源数也会导致排队包变化量的明显改变。
   - 用排队包变化量来判断流量变化时，一定要注意是否是由于资源数的变化导致的排队包变化量的改变，如果此步资源数调整了，排队包变化量的改变可能是由于资源数的变化导致的，而不是流量的变化。
   - 用排队包变化量来判断流量变化时一定还要结合流量变化来一起看
3. 探索策略
   - 在历史经验不足的情况下可以大胆探索,不要担心违反目标,大幅度增减资源来探索环境。
4. 异常处理:
   - 如果资源已达下限6了,仍然频谱效率低：保持最低资源数,6。
   - 如果资源已达上限40了,仍然SLA违约：保持最高资源数,40。
   - 系统有时会因为干扰而SLA违约，发现SLA违约时要根据以往的经验来判断是否是资源不够导致的，还是系统本身的干扰导致的。如果是干扰导致的SLA违约,则不需要调整资源数。
"""

act_prompt = """\
你是Actor, 根据以上信息, 生成这一步的动作, 严格按照以下格式分析当前情况并生成动作:

当前[是/不是]第一步
(如果历史步骤为None, 则为第一步)

{{% if not first_step %}}
## 上一步反思

上一步的情况为:
- O_{{t-1}}: [值列表]
- A_{{t}}: [整数]
- M_{{t}}: [值列表] 
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
"action": [这一步决定采取的动作，A{{t+1}}, 6-40的整数,不要超过这个区间，int类型]
}}
```"""

reflex_prompt = """\
你是reflector, 结合上面完整的episode信息, 逐条分析其中用<<<< >>>包裹的片段Z:

先声明片段Z中一共有多少条历史

然后对每一条历史执行:

- 声明这是Z中的第几条历史记录
- 重复这条历史记录的O, A和M
- 如果这一步的M都达到了目标，则不用分析后续的内容了
- 结合完整Episode的信息，结合前5步左右的流量，判断这一步是和前5步一样流量小幅度波动还是有明显增高过减少
- 分析这一步的频谱效率状态, SLA风险预测, 资源调整必要性
- 分析这条历史记录中的动作好坏, 有什么优点和缺陷
- 分析这个动作的优缺点，综合优缺点做后续分析
- 分析这一步原来分配的资源偏高，偏低还是恰好，如偏高或者偏低，结合历史记录以及计算结果，给出应该更换的动作

PS: 对每条历史反思时,都要结合整个Episode的信息,比如分析结合前后的流量变化趋势,分析是否可以让前后的资源分配都保持一致并且不会SLA违约且频谱效率高
    每行历史为[O_t, A_{{t+1}}, M_{{t+1}}],没有即时分析和上步反思,只有基础的历史记录
    反思时如果动作不好，可以参考这个episode内的相似情况来选择更好的动作

然后输出最终的json格式化结果, 要严格按照下面格式, 严格按照Z中历史记录的顺序生成:

```json
[
  {{
   "observation": [重复这一步的观测O]
   "original_action": [重复这一步的动作A],
   "original_metrics": [重复这一步的指标M],
   "reflection": {{
     "analysis": [分析这一步资源偏高还是偏低，还是恰好],
     "good": [资源分配是否恰好,布尔值,True/False]
     "advantage": [决策优点（若分析资源分配不好可省略为None）],
     "flaw": [主要缺陷分析（若分析资源分配恰好省略为None）],
     "suggested_action": [改进后的动作，int类型],
     "reason": [改进动作的原因，为了避免SLA违约次数，还是为了提高频谱效率，还是为了减少资源调整次数],
   }}
  }},
  ...
]
```
资源分配恰好不是表示是否达成了目标,而是表示这个动作是否足够好。比如流量很低，资源就算给到最低了，频谱效率还是低于要求的最低值，虽然没有达成目标，但这一步的动作是足够好的，good为True。

如果这一步的M都达成了目标，reflection中的good为True，其他字段都写字符串None
"""

teach_prompt = """\
你是teacher，你要结合reflector对这一步的反思和actor生成的原思考内容与动作结论，接着原来的思考内容生成一段转折，如“但是...所以应该...”，再生成新的动作。新的动作和转折不许提到Reflector，只是结合Reflector的反思来生成。也不许提到Actor和Teacher。也不要提到后续步骤的明确状态，要假装不知道后续的明确历史，站在那一步的角度上思考。

这段转折和新的动作要参照以下格式：

```text
对原思考过程续写的转折，只保留续写的转折，不要把原思考过程的内容在此重复
```

```json
{
"prev_step_reflection": [对上一步反思的总结，不是对原动作的反思，照抄原action里的prev_step_reflection就行了],
"analysis": [对当前观测的分析],
"action": [这一步决定采取的动作, 6-40的整数,不要超过这个区间，int类型]
}
```
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
    reflection = transaction.get("prev_step_reflection", "None")
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

def kto_rollout_stop(rollout_statistics):
    if rollout_statistics["total_times"] >= 20:
        return True
    return False

def kto_finish_check(rollout_statistics):
    if rollout_statistics["good_times"]/rollout_statistics["total_times"] >= 0.6:
        return True
    return False

def kto_filter(transaction: Transaction):
    metrics_SE = transaction["next_obs"][1]
    metrics_SLA = transaction["next_obs"][5]
    metrics_RB = transaction["next_obs"][6]
    metrics_is_good = (metrics_SE > 5 and metrics_SE < 6.5) and metrics_SLA == 0 and metrics_RB == 0
    reflex_is_good = transaction["reflex"]["reflection"]["good"]
    return metrics_is_good and reflex_is_good

llm_config = LLMConfig(
    system_description=system_desc,
    act_prompt=act_prompt,
    reflex_prompt=reflex_prompt,
    teach_prompt=teach_prompt,
    act_invoke_fn=lambda x: advanced_retry(qwq32b_invoker_act, 10, 3, messages=x),
    reflex_invoke_fn=reflex_invoker,
    teach_invoke_fn=qwq32b_invokers_teach2_retry,
    metrics_statistics_fn=metrics_statistics,
    reflex_model_tokens_counter_fn=deepseek_counter,
    act_model_tokens_counter_fn=qwen_counter,
    reflex_max_tokens_input=64000, # 64k tokens
    act_max_tokens_input=64000, # 64k tokens
    transaction_process_fn = process_transaction,
    obs_process_fn = process_obs,
    record_conversation = exp_folder,
    reflex_specific_transaction_process_fn=process_transaction_reflex,
    reflex_batch=reflex_batch,
    fine_tune_data_path=prj_path_join("fine_tune_data"),
    action_offset=-6,
    teach_method="kto",
    kto_rollout_stop_fn=kto_rollout_stop,
    kto_finish_check_fn=kto_finish_check,
    kto_human_refine=True,
    kto_specified_transaction_process_fn=process_transaction_reflex,
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
    multi_agent = False,
    log_paths = ["print", prj_path_join("llm_algo.log")],
)

llm_algo.logger.setLevel("INFO")

# train
# for _ in range(1):
#     llm_algo.train_iteration()

# eval
for _ in range(1):
    with llm_algo.train_state():
        eps = llm_algo.rollout()
        for ep in eps:
            llm_algo.logger.info(llm_algo.algorithm_config.metrics_statistics_fn(ep))

# kto rollout
# ori = prj_path_join("episodes_0_reflex_teach2_iter3.pkl")
# out = prj_path_join("episodes_0_reflex_teach2_iter4.pkl")
# it = 4
# with open(ori, "rb") as f:
#     eps = pickle.load(f)
#     print(f"loaded episodes in {ori}")

# for ep in eps:
#     llm_algo.kto_iteration(ep, 0, it)

# with open(out, "wb") as f:
#     pickle.dump(eps, f)
#     print(f"saved episodes in {out}")