from mpi_rl.mpi_rl.algorithm.llm.llm_utils import GeneralRequestInvoker
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import RoundRobinInvoker

from my_secret_keys import volcano_api_key, silicon_api_key, bailian_api_key # use your own API keys


volcano_r1_api_invoker = GeneralRequestInvoker(
    full_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    API_key = volcano_api_key,
    model_name="ep-20250211224011-vkzsm"  # use your own endpoint
)

silicon_r1_api_invoker = GeneralRequestInvoker(
    full_url = "https://api.siliconflow.cn/v1/chat/completions",
    API_key = silicon_api_key,
    model_name="Pro/deepseek-ai/DeepSeek-R1"
)

bailian_r1_api_invoker = GeneralRequestInvoker(
    full_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    API_key = "break this for test",
    model_name="deepseek-r1"
)

round_robin_invoker = RoundRobinInvoker(volcano_r1_api_invoker, silicon_r1_api_invoker, bailian_r1_api_invoker)

output = round_robin_invoker([{"role": "user", "content": "如果是版本号的话, 3.11和3.9哪个大"}])

output = round_robin_invoker([{"role": "user", "content": "如果是版本号的话, 3.11和3.9哪个大"}])

output = round_robin_invoker([{"role": "user", "content": "如果是版本号的话, 3.11和3.9哪个大"}])

print(output)

print(round_robin_invoker.success_invoke_times)

print(round_robin_invoker.fail_invoke_times)

print(round_robin_invoker.last_exceptions)