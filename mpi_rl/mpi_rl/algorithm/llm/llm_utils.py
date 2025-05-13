import requests
import re
from typing import Callable, Any, Optional, List, TypeVar
import time
import json5
from mpi_rl.utils import SpeedCalculator
from mpi_rl.config import get_use_mpi
from llf_server.llamafactory_server_call import LLamaFactoryClient
import multiprocessing

# ------------------ Typing Alias ------------------

Path_Without_NameT = str
MessagesT = list[dict[str, str]]
ContentT = str
Filtered_ContentT = Any
Reasoning_ContentT = str
Num_Input_TokensT = int
Num_Output_TokensT = int
Input_SequenceT = str
Invoker_outputT = tuple[ContentT, Filtered_ContentT, Reasoning_ContentT, Num_Input_TokensT, Num_Output_TokensT]
Kto_Rollout_StatisticsT = dict[str, Any] # {"total_times":0, "good_times":0, "bad_times":0}
JsonableT = TypeVar("JsonableT")    # can be json serialized by json.dumps

# ------------------ Model invoking ------------------

class RateLimitError(Exception):
    pass

class BaseInvoker:
    def __init__(self):
        pass
    def __call__(self, messages:MessagesT) -> Invoker_outputT:
        pass

class GeneralRequestInvoker(BaseInvoker):
    """
    General invoker for almost providers using request.
    """
    def __init__(
            self,
            full_url:str,
            model_name:str,
            API_key:str = None,
            top_p:int = 0.7,
            temperature:float = 0.6,
            top_k:int = 50,
            frequency_penalty: float = 0,
            max_tokens: int = 8192,
            content_filter_fn: Callable[[ContentT], Any] = lambda x: None,
            rate_limit_code: int = 429,
            timeout = 1800,
            reasoning_key: str = "reasoning_content"
            ):
        self.full_url = full_url
        self.payload = {
            "model": model_name,
            "stream": False,
            "max_tokens": max_tokens,
            "stop": ["null"],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": 1,
            "response_format": {"type": "text"},
        }
        if API_key is not None:
            self.headers = {
                "Authorization": f"Bearer {API_key}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Content-Type": "application/json"
            }
        self.content_filter_fn = content_filter_fn
        self.rate_limit_code = rate_limit_code
        self.timeout = timeout
        self.reasoning_key = reasoning_key

    def update_payload(self, **kwargs):
        self.payload.update(kwargs)

    def __call__(self, messages:MessagesT, avg_token_speed_calculator:SpeedCalculator=None) -> Invoker_outputT:
        self.payload["messages"] = messages
        start_time = time.time()
        response = requests.request("POST", self.full_url, json=self.payload, headers=self.headers, timeout=self.timeout)
        request_time = time.time() - start_time

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            reasoning_content = response.json()["choices"][0]["message"].get(self.reasoning_key, None)
            num_input_tokens = response.json()["usage"]["prompt_tokens"]
            num_output_tokens = response.json()["usage"]["completion_tokens"]
            filtered_content = self.content_filter_fn(content)
            if avg_token_speed_calculator is not None:
                avg_token_speed_calculator.add_data(num_output_tokens, request_time)
            return content, filtered_content, reasoning_content, num_input_tokens, num_output_tokens
        elif response.status_code == self.rate_limit_code:
            raise RateLimitError(f"Rate limit error: {response.text}")
        else:
            raise ValueError(f"API error: {response.text}")

        
class RoundRobinInvoker(BaseInvoker):
    """
    Round robin invoker with weights, invoke invokers in order according to their weights.
    """
    def __init__(self, *invokers: BaseInvoker, weights: Optional[List[int]] = None, try_round: int = 1, sleep_btw_round: int = 0, logger=None):
        self.logger = logger
        self.original_invokers = invokers
        if weights is None:
            weights = [1] * len(invokers)
        else:
            if len(weights) != len(invokers):
                raise ValueError("Length of weights must match the number of invokers")
            if any(w <= 0 for w in weights):
                raise ValueError("Weights must be positive integers")
        self.weights = weights
        self.invoker_indices = []
        for i, weight in enumerate(weights):
            self.invoker_indices.extend([i] * weight)
        self.index = 0
        self.try_round = try_round
        self.sleep_btw_round = sleep_btw_round
        num_original = len(self.original_invokers)
        self.success_invoke_times = {f"invoker_{i}": 0 for i in range(num_original)}
        self.fail_invoke_times = {f"invoker_{i}": 0 for i in range(num_original)}
        self.last_exceptions = {f"invoker_{i}": None for i in range(num_original)}

    def log_info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def __call__(self, messages: MessagesT) -> Invoker_outputT:
        self.log_info("Calling RoundRobinInvoker...")
        last_exception = None
        for r in range(self.try_round):
            for _ in range(len(self.invoker_indices)):
                current_index = self.index
                original_i = self.invoker_indices[current_index]
                invoker = self.original_invokers[original_i]
                self.index = (current_index + 1) % len(self.invoker_indices)
                try:
                    res = invoker(messages)
                    self.log_info(f"Invoker {original_i} success")
                    self.success_invoke_times[f"invoker_{original_i}"] += 1
                    return res
                except Exception as e:
                    self.log_info(f"Invoker {original_i} failed: {e}")
                    self.fail_invoke_times[f"invoker_{original_i}"] += 1
                    last_exception = e
                    self.last_exceptions[f"invoker_{original_i}"] = e
            if r < self.try_round - 1:
                # sleep except the last round
                time.sleep(self.sleep_btw_round)
        if last_exception is not None:
            self.log_info("All invokers failed")
            self.log_info(f"Last exception: {last_exception}")
            self.log_info(f"Success invoke times: {self.success_invoke_times}")
            self.log_info(f"Fail invoke times: {self.fail_invoke_times}")
            raise last_exception
        else:
            raise RuntimeError("No invokers available")


# ------------------ Token counter ------------------

class BaseTokenCounter:
    def __init__(self, model_name:str, bias:int):
        pass

    def __call__(self, input_sequence:str) -> int:
        pass
    
class HuggingfaceTokenCounter(BaseTokenCounter):
    def __init__(
            self,
            model_name:str,
            bias:int = 4
            ):
        self.model_name = model_name
        self.bias = bias
        from transformers import AutoTokenizer
        self.encoder = AutoTokenizer.from_pretrained(self.model_name)

    def __call__(
            self,
            input_sequence:str
            ) -> int:
        return len(self.encoder.encode(input_sequence))+self.bias
    
class OpenaiTokenCounter(BaseTokenCounter):
    def __init__(
            self,
            model_name:str,
            bias:int = 0
            ):
        self.model_name = model_name
        self.bias = bias
        import tiktoken
        self.encoder = tiktoken.encoding_for_model(self.model_name)

    def __call__(
            self,
            input_sequence:str
            ) -> int:
        return len(self.encoder.encode(input_sequence))+self.bias
    
class DummyTokenCounter(BaseTokenCounter):
    def __init__(
            self,
            model_name:str,
            bias:int = 0
            ):
        pass

    def __call__(
            self,
            input_sequence:str
            ) -> int:
        return 0
    

# ------------------ Content filter ------------------

def json_filter(
        content:ContentT
        ):
    matches = re.findall(r'\{.*\}', content, re.DOTALL)
    valid_json_list = []
    for match in matches:
        valid_json_list.append(json5.loads(match))
    if len(valid_json_list) > 0:
        return valid_json_list[-1]
    else:
        raise ValueError(f"No valid json found in response: {content}")
    
def code_filter(
        content:ContentT
        ):
    matches = re.findall(r'```.*?```', content, re.DOTALL)
    valid_code_list = []
    for match in matches:
        valid_code_list.append(match)
    if len(valid_code_list) > 0:
        return valid_code_list[-1]
    else:
        raise ValueError(f"No valid code block found in response: {content}")
    
def code_filter_multi(
        content:ContentT
        ):
    pattern = r'```([^\n`]*)\n([\s\S]*?)\n```'
    matches = re.finditer(pattern, content)
    result = []
    for match in matches:
        lang = match.group(1).strip() or None
        code = match.group(2)
        full = match.group(0)
        result.append({
            'language': lang,
            'code': code,
            'text': full,
        })
    if len(result) > 0:
        return result
    else:
        raise ValueError(f"No valid code block found in response: {content}")
    
def json_filter2(
        content:ContentT
        ):
    content_r = content.replace("```json", "").replace("```", "")
    data = json5.loads(content_r)
    return data

# ------------------ Others ------------------

def make_conversation(input_sequence, reasoning_content, content, expected_input_token_count, input_len, output_len, remark=None):
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

class LLMManager:
    def __init__(self, nodes_config: List[dict]):
        self.nodes_config = nodes_config
        for node in nodes_config:
            server_url = node["server_url"]
            client = LLamaFactoryClient(server_url)
            node["client"] = client
        self.processes = []

    def start_nodes(self, output_file: Optional[str] = None, process_id: Optional[str] = None):
        def run(client, command, output_file, print_output, process_id):
            pid = None
            try:
                while True:
                    try:
                        process_id, pid, resp = client.run_command_bg(command)
                        break
                    except Exception as e:
                        time.sleep(1)
                client.watch_bg_command(
                    resp,
                    output_file=output_file,
                    print_output=print_output,
                    process_id=process_id,
                )
            except Exception as e:
                print(f"debug Error in run command {command}: {e}")
                if pid:
                    print(f"debug Error {e} in run command {command}, killing process {pid}")
                    client.kill_process(pid)
        
        for i, node in enumerate(self.nodes_config):
            for j, process in enumerate(node["processes"]):
                command = node["up_command"].format(**process)
                self.processes.append(
                    multiprocessing.Process(
                        target=run,
                        kwargs={
                            "client": node["client"],
                            "command": command,
                            "output_file": output_file.format(n_idx=i, p_idx=j) if output_file else None,
                            "print_output": False,
                            "process_id": process_id.format(n_idx=i, p_idx=j) if process_id else None,
                        },
                        daemon=True,  # Will terminate automatically when the main process terminates
                    )
                )
        for p in self.processes:
            p.start()

    def stop_nodes(self):
        for p in self.processes:
            p.terminate()
            p.join()
        self.processes = []

    def wait_route(self):
        for node in self.nodes_config:
            for route in node["wait_routes"]:
                while True:
                    try:
                        response = requests.get(route)
                        if response.status_code == 200:
                            break
                    except requests.exceptions.RequestException as e:
                        time.sleep(1)

    def change_model_path(self, new_model_path: str, model_path_key: str = "model_path"):
        for node in self.nodes_config:
            for process in node["processes"]:
                process[model_path_key] = new_model_path
                