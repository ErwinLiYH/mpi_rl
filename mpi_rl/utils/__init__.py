import gymnasium as gym
from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
import multiprocessing
from functools import reduce
import time
from typing import Callable, Iterable, Type, Union, Any, TypeVar


def get_env_type(env):
    if isinstance(env, gym.Env):
        return "gym"
    elif isinstance(env, ParallelEnv):
        return "pzp"
    elif isinstance(env, AECEnv):
        return "pz_aec"
    else:
        raise ValueError("Unknown environment type: {}".format(type(env)))
    
def flat_double_layer_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def same_structure_recursive(dict1, dict2, max_depth, current_depth=0):
    if current_depth >= max_depth:
        return True
    
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not same_structure_recursive(dict1[key], dict2[key], max_depth, current_depth + 1):
                return False
        elif isinstance(dict1[key], dict) or isinstance(dict2[key], dict):
            return False
    
    return True

class DummyLogger:
    def __init__(self):
        pass

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

def dedent(input_seq:str, char, line_sep="\n", plus_char="    "):
    lines = input_seq.split("\n")
    for i in range(len(lines)):
        if lines[i].startswith(char):
            lines[i] = lines[i][len(char):]
        elif lines[i]=="":
            pass
        else:
            lines[i] = plus_char+lines[i]
    return line_sep.join(lines)

class RemoteObjectProxy:
    """
    RemoteObjectProxy is a proxy class for remote object.
    """
    class IsCallable:
        def __init__(self):
            pass

    def __init__(self, remote_obj_creator):
        self.parent_conn, child_conn = multiprocessing.Pipe()

        self.process = multiprocessing.Process(
            target=self._run_remote_object,
            args=(remote_obj_creator, child_conn),
            daemon=True
        )
        self.process.start()
        self.parent_conn.recv()  # wait for the remote object to be created

    def _run_remote_object(self, remote_obj_creator, conn):
        remote_obj = remote_obj_creator()
        conn.send("ready")
        while True:
            try:
                cmd = conn.recv()
                if cmd == "close":
                    break

                func_name, not_test, args, kwargs = cmd
                try:
                    attr = getattr(remote_obj, func_name)
                    if not callable(attr):
                        result = getattr(remote_obj, func_name)
                        conn.send(result)
                    else:
                        if not_test:
                            result = getattr(remote_obj, func_name)(*args, **kwargs)
                            conn.send(result)
                        else:
                            is_callable = self.IsCallable()
                            conn.send(is_callable)
                except Exception as e:
                    conn.send(e)
            except (EOFError, ConnectionResetError):
                break

    def __getattr__(self, name):
        def dynamic_call(*args, **kwargs):
            self.parent_conn.send((name, True, args, kwargs))
            
            result = self.parent_conn.recv()
            if isinstance(result, Exception):
                raise result
            return result

        self.parent_conn.send((name, False, (), {}))
        res = self.parent_conn.recv()
        if isinstance(res, Exception):
            raise res
        if isinstance(res, self.IsCallable):
            return dynamic_call
        else:
            return res

    def close(self):
        if self.process.is_alive():
            self.parent_conn.send("close")
            self.process.join()
        self.parent_conn.close()

def Pipeline(*funcs):
    def pipeline_two(f, g):
        return lambda x: g(f(x))
    return reduce(pipeline_two, funcs, lambda x: x)

def remove_first_line(s: str) -> str:
    lines = s.splitlines(keepends=True)
    return ''.join(lines[1:]) if len(lines) > 1 else ''

def advanced_retry(
    fn: Callable,
    max_attempts: int,
    retry_interval: float,
    retry_exceptions: Union[Type[Exception], Iterable[Type[Exception]]] = (Exception,),
    *args: Any,
    **kwargs: Any
) -> Any:
    if isinstance(retry_exceptions, type):
        retry_exceptions = (retry_exceptions,)
    elif not isinstance(retry_exceptions, tuple):
        retry_exceptions = tuple(retry_exceptions)

    last_exception: Union[Exception, None] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except retry_exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                time.sleep(retry_interval)
        except Exception as e:
            raise e

    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("Unknown error")
    
class SpeedCalculator:
    def __init__(self):
        self.total_quantity = 0.0
        self.total_time = 0.0

    def add_data(self, quantity: float, time: float) -> None:
        self.total_quantity += quantity
        self.total_time += time

    @property
    def average_speed(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_quantity / self.total_time

    def __str__(self) -> str:
        return f"{self.average_speed:.2f}"
    
def chunk(lst, n):
    if n <= 0:
        raise ValueError("n must be a positive integer")
    L = len(lst)
    k, r = divmod(L, n)
    sizes = [k + 1] * r + [k] * (n - r)
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(lst[start:end])
        start = end
    return result