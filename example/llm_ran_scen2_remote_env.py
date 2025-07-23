from my_secret_keys import volcano_api_key, bailian_api_key, ds_api_key, openrouter_api_key # use your own API keys
from my_secret_keys import volcano_r1_name

# set random seed for reproducibility
from ransim import config as ransim_config

from mpi_rl.mpi_rl.algorithm.llm.llm_algo import LLM, LLMConfig
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import (
        GeneralRequestInvoker, 
        RoundRobinInvoker, 
        code_filter_multi, 
        code_filter, 
        json_filter2
    )
from mpi_rl.utils.config_helper import RolloutConfig
from mpi_rl.utils.replay_buffer import ReplayBuffer, Episode, Transaction
from mpi_rl.utils import Pipeline, advanced_retry
from mpi_rl.mpi_utils.mpi_architecture_config import MPIArchitectureConfig
from mpi_rl.mpi_utils import allocate_list_resource_by_num

from copy import deepcopy
from pathlib import Path
import os
import sys

from remote_env_client import RLEnvClient
from prj_utils import path_allocator

from Kkit import init_logger, PathJoin
from Kkit.port_control import PortController

from sshtunnel import SSHTunnelForwarder
import paramiko


mpi_config = MPIArchitectureConfig(
    num_training_workers=1,
    num_evaluation_workers=11,
    parallel_train_eval=True,
    centralized_training=True,
    num_workers_per_group=None
)
mpi_config.mpi_print_role()


exp_folder = None
if mpi_config.global_rank == 0:
    # Set up SSH tunnel if needed ----------
    tunnel = SSHTunnelForwarder(
        ('login.isca.ex.ac.uk', 22),
        ssh_username='yl1118',
        ssh_private_key=paramiko.Ed25519Key.from_private_key_file('/root/.ssh/id_ed25519'),
        allow_agent=False,
        remote_bind_addresses=[
            ('10.122.0.86', 5003),
            ],
        local_bind_addresses=[
            ('localhost', 5000),
            ]
    )
    tunnel.start()
    # --------------------------------------
    if len(sys.argv) == 2:
        exp_folder = sys.argv[1]
    elif len(sys.argv) == 1:
        exp_folder = path_allocator("exps/exp", "")
        exp_folder = Path(exp_folder).resolve().as_posix()
exp_folder = mpi_config.global_comm.bcast(exp_folder, root=0)
print("Experiment result path:", exp_folder)

# exp_folder = "exps/exp57"

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
original_model_name = "qwq32_b"
nodes_config_model_path = "output/lrl_20250723_065852_train_it0_kto_it5" # "/lustre/home/yl1118/LLaMA-Factory/output/lrl_20250522_005527_train_it0_kto_it0" "Qwen/QwQ-32B" "/lustre/home/yl1118/LLaMA-Factory/output/lrl_20250522_005527_train_it0_kto_it1"  "Qwen/QwQ-32B" "Qwen/Qwen3-8B"
print(f"Model Path: {nodes_config_model_path}")
nodes_config_model_name = original_model_name
qwen3=True
qwen3_think=True
original_ports = [9001, 9002, 9004, 9005, 9007, 9008]
n_proc_each_port = 2

act_ports = allocate_list_resource_by_num(original_ports*n_proc_each_port, 1, mpi_config.global_comm)
# *n because each port can be used by n processes

act_port = act_ports[0]
reflex_port = original_ports[0]
teach_port = original_ports*n_proc_each_port

if mpi_config.global_rank == 0:
    print(f"original model port: {original_ports}")

prefix = f"g{mpi_config.group}_r{mpi_config.group_rank}"
print(f"{prefix} act_port: {act_port}")
print(f"{prefix} reflex_port: {reflex_port}")
print(f"{prefix} teach_port: {teach_port}")

top_p = 0.95 # 0.8
top_k = 20
temp = 0.6 # 0.7

qwq32b_invoker_act = GeneralRequestInvoker(
    full_url=f"http://localhost:{act_port}/v1/chat/completions",
    model_name=original_model_name,
    content_filter_fn=llm_action_filter_check,
    top_k=top_k,
    top_p=top_p,
    temperature=temp,
)

qwq32b_invoker_teach = GeneralRequestInvoker(
    full_url=f"http://localhost:{teach_port}/v1/chat/completions",
    model_name="QwQ_ori",
    content_filter_fn=llm_teach_filter_check,
    top_k=top_k,
    top_p=top_p,
    temperature=temp,
)

qwq32b_invoker_reflex = GeneralRequestInvoker(
    full_url=f"http://localhost:{reflex_port}/v1/chat/completions",
    model_name="QwQ_ori",
    content_filter_fn=llm_reflex_filter_check,
    top_k=top_k,
    top_p=top_p,
    temperature=temp,
)

def make_retry_lambda(invoker):
    return lambda x: advanced_retry(invoker, 10, 3, messages=x)

qwq32b_invokers_teach2 = [
    GeneralRequestInvoker(
        full_url=f"http://localhost:{port}/v1/chat/completions",
        model_name=original_model_name,
        content_filter_fn=llm_teach2_filter_check,
        top_k=top_k,
        top_p=top_p,
        temperature=temp,
    ) for port in teach_port
]
qwq32b_invokers_teach2_retry = [make_retry_lambda(invoker) for invoker in qwq32b_invokers_teach2]

nodes_config = [
    {
        "processes":[
            {
                "gpu":0,
                "model_path": nodes_config_model_path,
                "port": 9001,
                "model_name": nodes_config_model_name,
                "log_file": "/lustre/home/yl1118/vllmlogs/vllm9001",
            },
            {
                "gpu":1,
                "model_path": nodes_config_model_path,
                "port": 9002,
                "model_name": nodes_config_model_name,
                "log_file": "/lustre/home/yl1118/vllmlogs/vllm9002",
            }
        ],
        "server_url": "http://localhost:9000",
        "up_command": "CUDA_VISIBLE_DEVICES={gpu} vllm serve {model_path} --max-model-len 32768 --gpu-memory-utilization 0.98 --host 0.0.0.0 --port {port} --served-model-name {model_name} --enable-reasoning --reasoning-parser deepseek_r1 > {log_file}.out 2> {log_file}.err",
        "wait_routes": ["http://localhost:9001/v1/models", "http://localhost:9002/v1/models"],
    },
    {
        "processes":[
            {
                "gpu":0,
                "model_path": nodes_config_model_path,
                "port": 9004,
                "model_name": nodes_config_model_name,
                "log_file": "/lustre/home/yl1118/vllmlogs/vllm9004",
            },
            {
                "gpu":1,
                "model_path": nodes_config_model_path,
                "port": 9005,
                "model_name": nodes_config_model_name,
                "log_file": "/lustre/home/yl1118/vllmlogs/vllm9005",
            }
        ],
        "server_url": "http://localhost:9003",
        "up_command": "CUDA_VISIBLE_DEVICES={gpu} vllm serve {model_path} --max-model-len 32768 --gpu-memory-utilization 0.98 --host 0.0.0.0 --port {port} --served-model-name {model_name} --enable-reasoning --reasoning-parser deepseek_r1 > {log_file}.out 2> {log_file}.err",
        "wait_routes": ["http://localhost:9004/v1/models", "http://localhost:9005/v1/models"],
    },
    {
        "processes":[
            {
                "gpu":0,
                "model_path": nodes_config_model_path,
                "port": 9007,
                "model_name": nodes_config_model_name,
                "log_file": "/lustre/home/yl1118/vllmlogs/vllm9007",
            },
            {
                "gpu":1,
                "model_path": nodes_config_model_path,
                "port": 9008,
                "model_name": nodes_config_model_name,
                "log_file": "/lustre/home/yl1118/vllmlogs/vllm9008",
            }
        ],
        "server_url": "http://localhost:9006",
        "up_command": "CUDA_VISIBLE_DEVICES={gpu} vllm serve {model_path} --max-model-len 32768 --gpu-memory-utilization 0.98 --host 0.0.0.0 --port {port} --served-model-name {model_name} --enable-reasoning --reasoning-parser deepseek_r1 > {log_file}.out 2> {log_file}.err",
        "wait_routes": ["http://localhost:9007/v1/models", "http://localhost:9008/v1/models"],
    }
]

master_host = "10.122.1.239"

finetune_nodes_config = [
    {
        "processes":[
            {
                "master_addr": master_host,
                "master_port": 29500,
                "config_path": "init later",
            }
        ],
        "server_url": "http://localhost:9000",
        "up_command": "FORCE_TORCHRUN=1 NNODES={num_n} NODE_RANK={n_idx} MASTER_ADDR={master_addr} MASTER_PORT={master_port} llamafactory-cli train {config_path}"
    },
    {
        "processes":[
            {
                "master_addr": master_host,
                "master_port": 29500,
                "config_path": "init later",
            }
        ],
        "server_url": "http://localhost:9003",
        "up_command": "FORCE_TORCHRUN=1 NNODES={num_n} NODE_RANK={n_idx} MASTER_ADDR={master_addr} MASTER_PORT={master_port} llamafactory-cli train {config_path}"
    },
    {
        "processes":[
            {
                "master_addr": master_host,
                "master_port": 29500,
                "config_path": "init later",
            }
        ],
        "server_url": "http://localhost:9006",
        "up_command": "FORCE_TORCHRUN=1 NNODES={num_n} NODE_RANK={n_idx} MASTER_ADDR={master_addr} MASTER_PORT={master_port} llamafactory-cli train {config_path}"
    }
]

system_desc = """\
# 网络切片资源控制任务

这是一个网络切片资源控制的任务，将有2个Agent合作优化资源分配，一个是actor，负责接收当前观测和历史记录，生成动作，即分配的资源；一个是reflector，接收完整的episode记录，结合完整episode反思每一步的动作优劣，并给出更好的动作；

在RAN网络切片的资源分配任务中,Actor需要通过动态调整资源分配(6-40的整数)优化以下三个目标：
1.最小化SLA违约次数
2.保持较高的频谱效率, 频谱效率的要求分两种情况：1.流量大于9时理想为5-6.5之间；2.流量小于9时理想为4.5-5.2
3.减少资源调整次数

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
   - 这三个目标是相互冲突的,频谱效率过高（即资源数太少）会导致SLA违约。为了不SLA违约,过于谨慎地过多分配资源的话频谱效率会很低。尽量减少资源调整次数的话,即在不合适的资源数上保持,会导致频谱效率低或者SLA违约。所以要总结经验, 以学会最合适的资源分配策略。
2. 资源分配经验:
   - 频谱效率的近似值约等于 5.55*(流量/资源数)，所以频谱效率不足时应该减少资源数，频谱效率过高时应该增加资源数。
   - 发生SLA时，由于大量丢包，上面公式计算频谱效率会大幅度偏低而不准确，此时应该增加资源数。
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
   - 如果计算出来需要的资源数小于6,则直接选择6；如果计算出来需要的资源数大于40,则直接选择40。

要相信你的计算结果，相信我给的公式和经验是绝对正确的！！！
历史记录里的反思，分析和动作可能是错的！！！
历史记录里的反思与分析和我给的公式与经验冲突时，以我给的公式与经验为准！！！
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

决策思路：
1. 计算估算在当前流量下，让频谱效率在5-6.5之间需要的资源数的理论范围，算出明确的理论范围，算出明确的理论范围，算出明确的理论范围
2. 结合历史记录,流量变化趋势是什么?
3. 结合计算结果，判断当前资源数是否合适，并做出下一步的动作决策。如果现实无法满足理论范围，则直接选择最接近的动作。比如算出来的理论范围都在6以下，则直接选择6；如果算出来的理论范围都在40以上，则直接选择40。

## 最终结果

生成严格遵循此结构的JSON code block,并且动作要严格按照动作空间:
```json
{{
"prev_step_reflection": [对上一步反思的总结], 
"analysis": [对当前观测的分析],
"action": [这一步决定采取的动作，A{{t+1}}, 6-40的整数,不要超过这个区间，int类型]
}}
```"""

if qwen3:
    if qwen3_think:
        act_prompt += "\n/think\n"
    else:
        act_prompt += "\n/no_think\n"

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

然后输出最终的json格式化结果, json要严格按照下面格式并生成在json code block里, 严格按照Z中历史记录的顺序生成:

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

这段转折和新的动作要参照以下格式生成两个code blocks：

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
    actor_rollout_num=1,
    reflex_prompt=reflex_prompt,
    teach_prompt=teach_prompt,
    act_invoke_fn=lambda x: advanced_retry(qwq32b_invoker_act, 10, 3, messages=x),
    reflex_invoke_fn=reflex_invoker,
    teach_invoke_fn=qwq32b_invokers_teach2_retry,
    metrics_statistics_fn=metrics_statistics,
    transaction_process_fn = process_transaction,
    obs_process_fn = process_obs,
    record_conversation = exp_folder,
    reflex_specific_transaction_process_fn=process_transaction_reflex,
    reflex_batch=reflex_batch,
    fine_tune_data_path=prj_path_join("fine_tune_data"),
    inference_node_configs=nodes_config,
    finetune_node_configs=finetune_nodes_config,
    initial_train_config_path=os.path.join(os.path.dirname(__file__), "example_train.yaml"),
    initial_merge_config_path=os.path.join(os.path.dirname(__file__), "example_merge.yaml"),
    action_offset=-6,
    max_history=10,
    kto_rollout_stop_fn=kto_rollout_stop,
    kto_finish_check_fn=kto_finish_check,
    kto_human_refine=True,
    kto_specified_transaction_process_fn=process_transaction_reflex,
    kto_max_iteration=10,
    kto_early_stop_check_iteration=10,
    skip_reflex=True,
)

episode_len = 300

warmup_times = 50

def env_scen2_creator(algo_name):
    env = RLEnvClient("EnvType1", server_url="http://localhost:5000",)
    env.connect()
    return env

train_rollout_config = RolloutConfig(
    count_by = "episode",
    explore = True,          # ignored in LLM algorithm
    num_of_episode = 1,
)
eval_rollout_config = deepcopy(train_rollout_config)

replay = ReplayBuffer(30000)

llm_algo = LLM(
    train_env_creator = env_scen2_creator,
    train_rollout_config = train_rollout_config,
    eval_env_creator = env_scen2_creator,
    eval_rollout_config = eval_rollout_config,
    env_type = "gym",
    replay_buffer = replay,
    algorithm_config = llm_config,
    log_paths = ["print", prj_path_join("llm_algo.log")],
    mpi_architecture_config=mpi_config,
    hard_reset_env=True,
)

llm_algo.logger.setLevel("INFO")

# train
for _ in range(1):
    llm_algo.train_iteration()

# import pickle
# from mpi_rl.mpi_rl.algorithm.llm.llm_algo import EpisodeKTO

# if llm_algo.mpi_architecture_config.global_rank == 0:
#     with open("/root/mpi_rl/exps/exp57/episodes_0_reflex.pkl", "rb") as f:
#         eps = pickle.load(f)

#     for idx, ep in enumerate(eps):
#         if llm_algo.algorithm_config.kto_human_refine:
#             llm_algo.kto_human_refine(ep, idx)

#     wrappers = []
#     for i, ep in enumerate(eps):
        
#         for trans in ep:
#             # if trans["transaction"]["reflex"]["reflection"]["kto_finished"] exits, remove this key
#             trans["transaction"]["reflex"]["reflection"].pop("kto_finished", None)
#             # if trans["transaction"]["reflex"]["reflection"]["kto_promising"] exits, remove this key
#             trans["transaction"]["reflex"]["reflection"].pop("kto_promising", None)
#             # if trans["transaction"]["reflex"]["reflection"]["kto_remark"] exits, remove this key
#             trans["transaction"]["reflex"]["reflection"].pop("kto_remark", None)
#         wrappers.append(
#             EpisodeKTO(
#                 ep_idx=i,
#                 agent=None,
#                 episode=ep
#             )
#         )
    
#     llm_algo.teach(wrappers)

# if llm_algo.mpi_architecture_config.global_rank == 0:
#     for w in wrappers:
#         llm_algo.kto_human_refine(w.episode, w.ep_idx, None)
#     llm_algo.teach(
#         wrappers,
#         [
#             "/root/mpi_rl/exps/exp32/fine_tune_data/g0_r0_training-main_iter0/kto_finetune_iter0_ep0.json",
#             "/root/mpi_rl/exps/exp32/fine_tune_data/g0_r0_training-main_iter0/kto_finetune_iter0_ep1.json"
#         ],
#         0
#     )

    # with open("/root/mpi_rl/exps/exp57/episodes_wrappers.pkl", "wb") as f:
    #     pickle.dump(wrappers, f)