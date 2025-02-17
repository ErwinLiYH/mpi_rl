import requests
import re
from tenacity import retry, stop_after_attempt, RetryError
from typing import Callable, Any

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
            API_key:str,
            full_url:str,
            model_name:str,
            max_retry:int = 3,
            top_p:int = 0.7,
            temperature:float = 0.7,
            top_k:int = 50,
            frequency_penalty: float = 0.5,
            max_output_tokens: int = 8192,
            content_filter_fn: Callable[[ContentT], Any] = lambda x: None
            ):
        self.full_url = full_url
        self.payload = {
            "model": model_name,
            "stream": False,
            "max_tokens": max_output_tokens,
            "stop": ["null"],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": 1,
            "response_format": {"type": "text"},
        }
        self.headers = {
            "Authorization": f"Bearer {API_key}",
            "Content-Type": "application/json"
        }
        self.max_retry = max_retry
        self.content_filter_fn = content_filter_fn

    def update_payload(self, **kwargs):
        self.payload.update(kwargs)

    def __call__(self, messages:MessagesT) -> Invoker_outputT:
        @retry(stop=stop_after_attempt(self.max_retry))
        def inner():
            self.payload["messages"] = messages
            response = requests.request("POST", self.full_url, json=self.payload, headers=self.headers, timeout=1800)

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                reasoning_content = response.json()["choices"][0]["message"].get("reasoning_content", None)
                num_input_tokens = response.json()["usage"]["prompt_tokens"]
                num_output_tokens = response.json()["usage"]["completion_tokens"]
                filtered_content = self.content_filter_fn(content)
                return content, filtered_content, reasoning_content, num_input_tokens, num_output_tokens
            else:
                raise ValueError(f"API error: {response.text}")
        
        try:
            return inner()
        except RetryError as e:
            raise e.last_attempt.exception()

class DoubleInvoker(BaseInvoker):
    """
    Main invoker and vice invoker
    """
    def __init__(self, main_invoker:BaseInvoker, vice_invoker:BaseInvoker):
        self.main_invoker = main_invoker
        self.vice_invoker = vice_invoker

    def __call__(self, messages:MessagesT) -> Invoker_outputT:
        try:
            return self.main_invoker(messages)
        except:
            return self.vice_invoker(messages)


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

def llm_action_filter(
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