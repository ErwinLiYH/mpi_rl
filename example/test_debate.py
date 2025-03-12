from mpi_rl.mpi_rl.algorithm.llm.llm_action_compute import MultiAgentDebate
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import GeneralRequestInvoker
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import json_filter
from mpi_rl.utils import Pipeline
from pprint import pprint
from my_secret_keys import silicon_api_key


system_prompt = """\
you are a helpful assistant, when answering questions, output final result in json format like:
{"answer": "the answer"}
the answer should contain analysis and final result, the final result in json format should be minimal, and only contain the answer content. for example, if the answer is "the answer is 42", the final result should be {"answer": "42"}
"""

question = "What is the result of 4+23*6+24-24*12"

invoker = GeneralRequestInvoker(
    content_filter_fn=json_filter,
    full_url = "https://api.siliconflow.cn/v1/chat/completions",
    API_key=silicon_api_key,
    model_name="Qwen/Qwen2-1.5B-Instruct",
    max_output_tokens=1024,
    )

debate = MultiAgentDebate(invoker, n_agents=5, max_rounds=10, filter_fn=lambda x: x["answer"])
content ,filtered_content, res, input_tokens, output_tokens = debate([{"role": "system", "content": system_prompt}, {"role": "user", "content": question}])

pprint(debate.actions_each_round)
print("content:", content)