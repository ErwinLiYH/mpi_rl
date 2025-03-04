from mpi_rl import config
config.use_mpi = False

from my_secret_keys import volcano_api_key, silicon_api_key, bailian_api_key # use your own API keys

from mpi_rl.mpi_rl.algorithm.llm.llm_utils import HuggingfaceTokenCounter, GeneralRequestInvoker
import time

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
    API_key = bailian_api_key,
    model_name="deepseek-r1"
)

input_sequence = "如果是版本号的话, 3.11和3.9哪个大"

huggingface_token_counter = HuggingfaceTokenCounter(
    model_name = "deepseek-ai/DeepSeek-R1"
)

excepted_input_len = huggingface_token_counter(input_sequence)

a = time.time()
content, filtered_content, reasoning_content, input_len, output_len = silicon_r1_api_invoker([{"role": "user", "content": input_sequence}])
b = time.time()

print("Silicon Flow")
# print("硅基流动")
print("calculated_input_len:", excepted_input_len)
print("input_len:", input_len)
print("calculated_output_len:", huggingface_token_counter(content+reasoning_content))
print("output_len:", output_len)
print("speed:", output_len/(b-a), "tokens/s")

print("\n")

a = time.time()
content, filtered_content, reasoning_content, input_len, output_len = volcano_r1_api_invoker([{"role": "user", "content": input_sequence}])
b = time.time()

print("Volcano Engine")
# print("火山引擎")
print("calculated_input_len:", excepted_input_len)
print("input_len:", input_len)
print("calculated_output_len:", huggingface_token_counter(content+reasoning_content))
print("output_len:", output_len)
print("speed:", output_len/(b-a), "tokens/s")

print("\n")

a = time.time()
content, filtered_content, reasoning_content, input_len, output_len = bailian_r1_api_invoker([{"role": "user", "content": input_sequence}])
b = time.time()

print("Ali Bailian")
# print("阿里云百炼")
print("calculated_input_len:", excepted_input_len)
print("input_len:", input_len)
print("calculated_output_len:", huggingface_token_counter(content+reasoning_content))
print("output_len:", output_len)
print("speed:", output_len/(b-a), "tokens/s")