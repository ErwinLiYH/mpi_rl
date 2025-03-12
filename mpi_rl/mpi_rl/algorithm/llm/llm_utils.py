import requests
import re
from typing import Callable, Any
import time

# ------------------ Typing Alias ------------------

Markdown_Path_Without_NameT = str
MessagesT = list[dict[str, str]]
ContentT = str
Filtered_ContentT = Any
Reasoning_ContentT = str
Num_Input_TokensT = int
Num_Output_TokensT = int
Input_SequenceT = str
Invoker_outputT = tuple[ContentT, Filtered_ContentT, Reasoning_ContentT, Num_Input_TokensT, Num_Output_TokensT]

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
            rate_limit_sleep: int = 60,
            wait = False    # wait rate_limit_sleep when retry
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
        self.rate_limit_sleep = rate_limit_sleep
        self.wait = wait

    def update_payload(self, **kwargs):
        self.payload.update(kwargs)

    def __call__(self, messages:MessagesT) -> Invoker_outputT:
        self.payload["messages"] = messages
        response = requests.request("POST", self.full_url, json=self.payload, headers=self.headers, timeout=1800)

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            reasoning_content = response.json()["choices"][0]["message"].get("reasoning_content", None)
            num_input_tokens = response.json()["usage"]["prompt_tokens"]
            num_output_tokens = response.json()["usage"]["completion_tokens"]
            filtered_content = self.content_filter_fn(content)
            return content, filtered_content, reasoning_content, num_input_tokens, num_output_tokens
        elif response.status_code == self.rate_limit_code:
            raise RateLimitError(f"Rate limit error: {response.text}")
        else:
            raise ValueError(f"API error: {response.text}")

        
class RoundRobinInvoker(BaseInvoker):
    """
    Round robin invoker, invoke invokers in order.
    """
    def __init__(self, *invokers:BaseInvoker, try_round:int = 1, sleep_btw_round:int = 0):
        self.invokers = invokers
        self.index = 0
        self.try_round = try_round
        self.sleep_btw_round = sleep_btw_round
        self.success_invoke_times = {f"invoker_{i}": 0 for i in range(len(self.invokers))}
        self.fail_invoke_times = {f"invoker_{i}": 0 for i in range(len(self.invokers))}
        self.last_exceptions = {f"invoker_{i}": None for i in range(len(self.invokers))}

    def __call__(self, messages:MessagesT) -> Invoker_outputT:
        last_exception = None
        for r in range(self.try_round):
            for _ in range(len(self.invokers)):
                current_index = self.index
                invoker = self.invokers[current_index]
                self.index = (current_index + 1) % len(self.invokers)
                try:
                    res = invoker(messages)
                    self.success_invoke_times[f"invoker_{current_index}"] += 1
                    return res
                except Exception as e:
                    self.fail_invoke_times[f"invoker_{current_index}"] += 1
                    last_exception = e
                    self.last_exceptions[f"invoker_{current_index}"] = e
            if r < self.try_round - 1:
                # sleep except the last round
                time.sleep(self.sleep_btw_round)
        if last_exception is not None:
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
    from jsonc_parser.parser import JsoncParser
    matches = re.findall(r'\{.*\}', content, re.DOTALL)
    valid_json_list = []
    for match in matches:
        valid_json_list.append(JsoncParser.parse_str(match))
    if len(valid_json_list) > 0:
        return valid_json_list[0]
    else:
        raise ValueError(f"No valid json found in response: {content}")