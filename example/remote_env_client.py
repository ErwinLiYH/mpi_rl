import requests
import time
import json
from typing import Dict, Any, List, Tuple, Optional

class RLEnvClient:
    """
    RL环境服务器客户端
    用于连接和管理远程RL环境服务器
    """
    
    def __init__(self, env_type, server_url: str = "http://localhost:5000", timeout: int = 600):
        """
        初始化RL环境客户端
        
        Args:
            server_url (str): 环境服务器URL, 默认 http://localhost:5000
            timeout (int): 请求超时时间(秒), 默认 60
        """
        self.server_url = server_url
        self.timeout = timeout
        self.env_id = None  # 当前连接的环境ID
        self.env_type = env_type
    
    def _send_request(self, endpoint: str, method: str = "POST", data: Optional[Dict] = None) -> Optional[Dict]:
        """
        向服务器发送请求
        
        Args:
            endpoint (str): API端点(如 '/connect')
            method (str): HTTP方法 (GET/POST)
            data (dict): 请求数据
            
        Returns:
            dict: 服务器响应或None(如果失败)
        """
        url = f"{self.server_url}{endpoint}"
        
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            else:  # GET
                response = requests.get(url, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
            
        # except requests.exceptions.Timeout:
        #     print(f"错误: 请求超时 ({endpoint})", flush=True)
        # except requests.exceptions.ConnectionError:
        #     print(f"错误: 无法连接到服务器 ({self.server_url})", flush=True)
        # except requests.exceptions.HTTPError as e:
        #     print(f"HTTP错误: {e}\nURL: {url}", flush=True)
        #     if response.status_code >= 500:
        #         print(f"服务器内部错误: {response.text}", flush=True)
        # except json.JSONDecodeError:
        #     print(f"错误: JSON解析失败 ({response.text})", flush=True)
        # except Exception as e:
        #     print(f"错误: {e}", flush=True)
        except Exception as e:
            pass
        
        return None
    
    def connect(self, retry_interval: int = 2) -> Optional[int]:
        """
        连接到指定类型的空闲环境
        如果没有空闲环境，会每隔retry_interval秒重试
        
        Args:
            env_type (str): 环境类型
            retry_interval (int): 重试间隔(秒), 默认 2
            
        Returns:
            int: 环境ID 或 None(如果手动取消)
        """
        #print(f"请求连接到 '{self.env_type}' 类型的环境...")
        
        while True:
            response = self._send_request("/connect", "POST", {"env_type": self.env_type})
            
            if response and "env_id" in response:
                env_id = response["env_id"]
                #print(f"成功连接到环境 ID: {env_id}")
                self.env_id = env_id
                return env_id
            
            # if response and "error" in response and "No environment available" in response["error"]:
            #     print(f"当前没有 '{self.env_type}' 类型的空闲环境，{retry_interval}秒后重试...")
            # elif response and "error" in response:
            #     print(f"连接失败: {response['error']}")
            # else:
            #     print("连接失败: 未知原因")
            
            # 等待一段时间后重试
            time.sleep(retry_interval)
    
    def reset(self, reset_params: Optional[Dict] = None) -> Optional[Any]:
        """
        重置指定环境
        
        Args:
            reset_params (dict): 重置参数, 可选
            
        Returns:
            any: 重置结果或None(如果失败)
        """
        if reset_params is None:
            reset_params = {}
            
        #print(f"重置环境 ID: {self.env_id}")
        response = self._send_request("/reset", "POST", {
            "env_id": self.env_id,
            "reset_params": reset_params
        })
        
        if response and "result" in response:
            #print(f"环境 {self.env_id} 重置成功")
            result = response["result"]
            if isinstance(result, list):
                return tuple(result)
            return result
            
        # if response and "error" in response:
        #     print(f"重置失败: {response['error']}")
        # else:
        #     print("重置失败: 未知原因")
            
        return None

    def reset_id(self, EnvId, reset_params: Optional[Dict] = None):
        response = self._send_request("/reset", "POST", {
            "env_id": EnvId,
            "reset_params": reset_params
        })
        if response and "result" in response:
            #print(f"环境 {self.env_id} 重置成功")
            result = response["result"]
            if isinstance(result, list):
                return tuple(result)
            return result

        # if response and "error" in response:
        #     print(f"重置失败: {response['error']}")
        # else:
        #     print("重置失败: 未知原因")
        #     print(response)

        return None

    def step(self, action: Any) -> Optional[Tuple]:
        """
        在指定环境中执行一步操作
        
        Args:
            action (any): 要执行的动作
            
        Returns:
            tuple: (观测值, 奖励, 完成标志, 信息) 或 None(如果失败)
        """
        #print(f"环境 ID: {self.env_id} 执行动作: {action}")
        response = self._send_request("/step", "POST", {
            "env_id": self.env_id,
            "action": action
        })
        
        if response and "result" in response:
            #print(f"环境 {self.env_id} 步骤执行成功")
            
            # 尝试将结果解析为标准RL四元组
            result = response["result"]
            if isinstance(result, list) and len(result) >= 4:
                return tuple(result)
            return result
            
        # if response and "error" in response:
        #     print(f"步骤执行失败: {response['error']}")
        # else:
        #     print("步骤执行失败: 未知原因")
            
        return None
    
    def close(self) -> bool:
        """
        关闭并释放指定环境
        
        Args:
            env_id (int): 环境ID
            
        Returns:
            bool: 是否成功关闭
        """
        #print(f"关闭环境 ID: {self.env_id}")
        response = self._send_request("/close", "POST", {
            "env_id": self.env_id
        })
        
        if response and "success" in response and response["success"]:
            #print(f"环境 {self.env_id} 已成功关闭")
            # 重新连接环境
            return True
            
        # if response and "error" in response:
        #     print(f"关闭失败: {response['error']}")
        # else:
        #     print("关闭失败: 未知原因")
            
        return False
    
    def get_status(self) -> Optional[Dict]:
        """
        获取服务器上所有环境的状态
        
        Returns:
            dict: 环境状态字典 或 None(如果失败)
        """
        print("获取服务器状态...")
        response = self._send_request("/status", "GET")
        
        if response and "environments" in response:
            print("成功获取服务器状态")
            return response
            
        if response and "error" in response:
            print(f"获取状态失败: {response['error']}")
        else:
            print("获取状态失败: 未知原因")
            
        return None
    
    def print_status_summary(self):
        """打印环境的简单状态摘要"""
        status = self.get_status()
        if not status:
            print("获取状态失败")
            return
        
        # 提取环境列表
        environments = status.get("environments", [])
        status_counts = status.get("status_counts", {})
        
        # 打印标题
        print("\n" + "="*40)
        print(f"{'服务器环境状态':^40}")
        print("="*40)
        
        # 打印每个环境的状态
        print("\n环境列表:")
        for env in environments:
            print(f"ID: {env['id']:<4} | 类型: {env['type'][:10]:<10} | 状态: {env['status']}")
        
        # 打印状态统计
        print("\n状态统计:")
        print(f"总环境数: {status_counts.get('TOTAL', len(environments))}")
        print(f"初始化中: {status_counts.get('INITIALIZING', 0)}")
        print(f"空闲可用: {status_counts.get('IDLE', 0)}")
        print(f"使用中: {status_counts.get('BUSY', 0)}")
        print(f"失败: {status_counts.get('FAILED', 0)}")
        print("="*40 + "\n")

if __name__ == "__main__":
    # 客户端使用示例
    client = RLEnvClient()
    
    # 查看服务器状态
    client.print_status_summary()
    
    # 连接到指定类型的环境
    try:
        # 使用较短的超时时间便于演示重试
        env_id = client.connect("CartPole", retry_interval=1)
        print(f"已连接环境 ID: {env_id}")
        
        # 重置环境
        observation = client.reset(env_id)
        if observation is None:
            print("环境重置失败")
            exit(1)
        
        print(f"初始观测值: {observation}")
        
        # 模拟5步动作
        for step in range(5):
            # 随机动作 - 实际应用中应替换为策略
            action = 0  # 简单示例，总是选择动作0
            
            # 执行一步
            result = client.step(env_id, action)
            if result is None:
                print("步骤执行失败")
                break
                
            # 解析结果
            if isinstance(result, tuple) and len(result) >= 4:
                observation, reward, done, info = result[0], result[1], result[2], result[3]
                print(f"步骤 {step+1}: 动作={action}, 奖励={reward}, 完成={done}")
                
                if done:
                    print("环境结束!")
                    break
            else:
                print(f"步骤 {step+1} 结果: {result}")
    except KeyboardInterrupt:
        print("\n用户中断操作")
    
    # 关闭环境（如果已连接）
    if 'env_id' in locals():
        client.close(env_id)
    
    # 查看服务器状态
    client.print_status_summary()