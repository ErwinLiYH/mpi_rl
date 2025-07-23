import logging
from logging.handlers import TimedRotatingFileHandler
import os
import json
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from Kkit.child_process import RemoteObjectProxy
import time
import sys

PORT = sys.argv[1] if len(sys.argv) > 1 else 5000

from ransim_env.envs import scenario_two

class LoggerManager:
    """集中式日志管理系统"""
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name='rl_server', log_level=logging.DEBUG, log_path='remote_env_server_logs'):
        """获取或创建日志记录器"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # 确保日志目录存在
        os.makedirs(log_path, exist_ok=True)
        
        # 创建文件处理器
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_path, name+'.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(processName)-11s | %(threadName)-15s | %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        cls._loggers[name] = logger
        return logger

class EnvironmentServer:
    def __init__(self, configurations):
        self.logger = LoggerManager.get_logger(name=f"rl_server_{PORT}")
        
        self.configurations = configurations
        self.logger.info("Initializing EnvironmentServer with configurations:")
        for i, config in enumerate(configurations):
            self.logger.info(f"  Config {i+1}: Type='{config['type']}', Count={config['count']}")
        
        self.env_counter = 0
        self.env_id_to_info = {}
        self.idle_envs_by_type = defaultdict(deque)
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=sum(c['count'] for c in configurations), 
            thread_name_prefix="EnvThread"
        )
        self.env_type_to_creator = {}
        self._initialize_environments()

    def _initialize_environments(self):
        """根据配置初始化所有环境实例"""
        self.logger.info("Starting parallel environment initialization...")
        
        for config in self.configurations:
            env_type = config['type']
            creator = config['creator']
            count = config['count']
            
            self.env_type_to_creator[env_type] = creator
            for i in range(count):
                env_id = self.next_id()
                self.logger.debug(f"Creating new environment. ID={env_id}, Type={env_type}")
                
                self.env_id_to_info[env_id] = {
                    'env_type': env_type,
                    'status': 'INITIALIZING',
                    'proxy': None
                }
                self.thread_pool.submit(
                    self._init_single_env, 
                    env_id, 
                    creator
                )
        
        self.logger.info(f"{self.env_counter} environments scheduled for initialization")

    def _init_single_env(self, env_id, creator):
        """在后台线程中初始化单个环境"""
        try:
            self.logger.debug(f"Initializing environment. ID={env_id}")
            
            proxy = RemoteObjectProxy(
                creator,
                paralle_execution=True,
                id=env_id,
            )
            
            with self.lock:
                env_info = self.env_id_to_info.get(env_id)
                if env_info:  # 防止已被删除
                    env_info['proxy'] = proxy
                    env_info['status'] = 'IDLE'
                    self.idle_envs_by_type[env_info['env_type']].append(env_id)
                    self.logger.info(f"Environment initialized successfully. ID={env_id}")
                else:
                    self.logger.warning(f"Environment disappeared during initialization! ID={env_id}")
        except Exception as e:
            self.logger.error(f"Environment initialization failed! ID={env_id}, Error={str(e)}", exc_info=True)
            with self.lock:
                # 标记为失败状态
                if env_id in self.env_id_to_info:
                    self.env_id_to_info[env_id]['status'] = 'FAILED'
                    self.env_id_to_info[env_id]['error'] = str(e)

    def next_id(self):
        """生成唯一环境ID"""
        with self.lock:
            env_id = self.env_counter
            self.env_counter += 1
            return env_id

    def connect(self, env_type):
        """连接到一个空闲的环境"""
        with self.lock:
            if not self.idle_envs_by_type.get(env_type):
                self.logger.warning(f"No available environments for type '{env_type}'")
                return None
            
            env_id = self.idle_envs_by_type[env_type].popleft()
            env_info = self.env_id_to_info[env_id]
            
            # 双重检查状态
            if env_info['status'] == 'IDLE' and env_info['proxy'] is not None:
                env_info['status'] = 'BUSY'
                self.logger.info(f"Environment connected. ID={env_id}, Type={env_type}")
                return env_id
        
        return None

    def reset(self, env_id, reset_params):
        """重置指定环境"""
        start_time = time.time()
        self.logger.debug(f"Resetting environment. ID={env_id}")
        
        with self.lock:
            env_info = self.env_id_to_info.get(env_id)
            if not env_info or env_info['status'] != 'BUSY' or not env_info['proxy']:
                self.logger.error(f"Invalid environment for reset: ID={env_id}")
                return None
            proxy = env_info['proxy']
        
        try:
            # 使用recv()确保阻塞直到完成
            result_pipe = proxy.reset(reset_params)
            result = result_pipe.recv()
            
            duration = time.time() - start_time
            self.logger.info(f"Environment reset completed. ID={env_id}, Duration={duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Environment reset failed! ID={env_id}, Duration={duration:.3f}s, Error={str(e)}", exc_info=True)
            return None

    def step(self, env_id, action):
        """在指定环境中执行一步"""
        start_time = time.time()
        self.logger.debug(f"Stepping environment. ID={env_id}, Action={action}")
        
        with self.lock:
            env_info = self.env_id_to_info.get(env_id)
            if not env_info or env_info['status'] != 'BUSY' or not env_info['proxy']:
                self.logger.error(f"Invalid environment for step: ID={env_id}")
                return None
            proxy = env_info['proxy']
        
        try:
            result_pipe = proxy.step(action)
            result = result_pipe.recv()
            
            duration = time.time() - start_time
            self.logger.info(f"Environment step completed. ID={env_id}, Duration={duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Environment step failed! ID={env_id}, Duration={duration:.3f}s, Error={str(e)}", exc_info=True)
            return None

    def close(self, env_id):
        """关闭并重新初始化环境"""
        self.logger.debug(f"Closing environment. ID={env_id}")
        
        with self.lock:
            env_info = self.env_id_to_info.get(env_id)
            if not env_info or env_info['status'] != 'BUSY' or not env_info['proxy']:
                self.logger.error(f"Invalid environment for close: ID={env_id}")
                return False
            
            # 标记为重新初始化中
            env_info['status'] = 'INITIALIZING'
            env_type = env_info['env_type']
        
        # 在后台重新初始化
        self.logger.info(f"Environment closing and reinitialization scheduled. ID={env_id}")
        self.thread_pool.submit(
            self._reinitialize_env,
            env_id,
            self.env_type_to_creator[env_type]
        )
        return True

    def _reinitialize_env(self, env_id, creator):
        """在后台重新初始化环境"""
        self.logger.debug(f"Reinitializing environment. ID={env_id}")
        
        # 关闭旧环境
        old_proxy = None
        with self.lock:
            if env_id in self.env_id_to_info:
                old_proxy = self.env_id_to_info[env_id]['proxy']
                self.env_id_to_info[env_id]['proxy'] = None
            
        try:
            if old_proxy:
                old_proxy.close()
        except Exception as e:
            self.logger.warning(f"Error closing old environment proxy. ID={env_id}, Error={str(e)}")
        
        try:
            # 创建新环境
            proxy = RemoteObjectProxy(
                creator,
                paralle_execution=True,
                id=env_id,
            )
            
            with self.lock:
                if env_id in self.env_id_to_info:
                    self.env_id_to_info[env_id] = {
                        'env_type': self.env_id_to_info[env_id]['env_type'],
                        'status': 'IDLE',
                        'proxy': proxy
                    }
                    self.idle_envs_by_type[self.env_id_to_info[env_id]['env_type']].append(env_id)
                    self.logger.info(f"Environment reinitialized successfully. ID={env_id}")
                else:
                    self.logger.warning(f"Environment disappeared during reinitialization! ID={env_id}")
        except Exception as e:
            self.logger.error(f"Environment reinitialization failed! ID={env_id}, Error={str(e)}", exc_info=True)
            with self.lock:
                # 标记为失败状态
                if env_id in self.env_id_to_info:
                    self.env_id_to_info[env_id]['status'] = 'FAILED'
                    self.env_id_to_info[env_id]['error'] = str(e)

# Flask服务器配置
app = Flask(__name__)
logger = LoggerManager.get_logger(name=f"rl_server_{PORT}")
# 从配置创建服务器实例（假设已定义configurations）
# configurations = [
#    {'creator': make_env1, 'type': 'EnvType1', 'count': 5},
#    {'creator': make_env2, 'type': 'EnvType2', 'count': 3}
# ]
def make_env1(id):
    return scenario_two.ScenarioTwo(
        f"remote_env", 300,
        unchange_terminate = 0,
        root_path=os.path.join(f"remote_env_logs_{PORT}", f"remote_env_{id}"),
        no_termi=True,
        plot = False,
        num_obs_history=1,
        warmup_times = 50,
        random_seed=50,
    )
configurations = [
    {'creator': make_env1, 'type': 'EnvType1', 'count': 12},
]  # 这里需要根据实际情况填充配置
manager = EnvironmentServer(configurations)

# 请求日志中间件
@app.before_request
def log_request_info():
    """记录HTTP请求信息"""
    logger.debug(f"Request: {request.method} {request.path}, "
                 f"Client: {request.remote_addr}, "
                 f"Content-Type: {request.content_type}")
    
    # 记录小负载的请求体（避免日志过大）
    if request.content_type == 'application/json' and request.content_length and request.content_length < 1024:
        logger.debug(f"Request Body: {json.dumps(request.json, indent=2)}")

@app.after_request
def log_response_info(response):
    """记录HTTP响应信息"""
    # 简化响应日志（避免记录大数据）
    log_data = {
        'status': response.status,
        'content_type': response.content_type
    }
    
    # 添加小负载内容
    if (response.content_type == 'application/json' and 
        response.content_length and 
        response.content_length < 1024):
        try:
            log_data['content'] = response.json
        except:
            pass
    
    logger.debug(f"Response: {json.dumps(log_data, indent=2)}")
    return response

# 错误处理器
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """处理未预期异常"""
    logger.critical(f"Unhandled exception occurred! {str(error)}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# 接口路由
@app.route('/connect', methods=['POST'])
def connect_handler():
    """连接环境处理器"""
    try:
        data = request.json
        env_type = data['env_type']
        
        logger.info(f"Connect request received. Type={env_type}")
        env_id = manager.connect(env_type)
        
        if env_id is None:
            logger.warning(f"No environment available for type '{env_type}'")
            return jsonify({'error': 'No available environment'}), 404
            
        return jsonify({'env_id': env_id})
    except Exception as e:
        logger.error(f"Connect handler failed! Error={str(e)}", exc_info=True)
        return jsonify({'error': 'Internal error'}), 500

@app.route('/reset', methods=['POST'])
def reset_handler():
    """重置环境处理器"""
    try:
        data = request.json
        env_id = data['env_id']
        reset_params = data.get('reset_params', {})
        
        logger.info(f"Reset request received. ID={env_id}")
        result = manager.reset(env_id, reset_params)
        result = list(result)
        result[0] = result[0].tolist()

        logger.info(f"Reset completed for env {env_id}, reset result={result}")
        
        if result is None:
            logger.warning(f"Invalid reset request for env {env_id}")
            return jsonify({'error': 'Invalid environment'}), 400
            
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Reset handler failed! Error={str(e)}", exc_info=True)
        return jsonify({'error': 'Internal error'}), 500

@app.route('/step', methods=['POST'])
def step_handler():
    """步进环境处理器"""
    try:
        data = request.json
        env_id = data['env_id']
        action = data['action']
        
        logger.info(f"Step request received. ID={env_id}")
        result = manager.step(env_id, action)
        result = list(result)
        result[0] = result[0].tolist()
        
        if result is None:
            logger.warning(f"Invalid step request for env {env_id}")
            return jsonify({'error': 'Invalid environment'}), 400
            
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Step handler failed! Error={str(e)}", exc_info=True)
        return jsonify({'error': 'Internal error'}), 500

@app.route('/close', methods=['POST'])
def close_handler():
    """关闭环境处理器"""
    try:
        data = request.json
        env_id = data['env_id']
        
        logger.info(f"Close request received. ID={env_id}")
        success = manager.close(env_id)
        
        if not success:
            logger.warning(f"Invalid close request for env {env_id}")
            return jsonify({'error': 'Invalid environment'}), 400
            
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Close handler failed! Error={str(e)}", exc_info=True)
        return jsonify({'error': 'Internal error'}), 500
    
@app.route('/status', methods=['GET'])
def status_handler():
    """获取所有环境状态信息"""
    try:
        logger.info("Status request received")
        
        # 准备环境列表
        environments = []
        with manager.lock:
            for env_id, info in manager.env_id_to_info.items():
                environments.append({
                    'id': env_id,
                    'type': info['env_type'],
                    'status': info['status']
                })
        
        # 计算状态统计
        status_counts = {
            'INITIALIZING': 0,
            'IDLE': 0,
            'BUSY': 0,
            'FAILED': 0,
            'TOTAL': len(environments)
        }
        
        for env in environments:
            status_counts[env['status']] += 1
        
        response_data = {
            'environments': environments,
            'status_counts': status_counts
        }
        
        logger.info(f"Status report generated. Total envs: {status_counts['TOTAL']}")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Status handler failed! Error={str(e)}", exc_info=True)
        return jsonify({'error': 'Internal error'}), 500

if __name__ == '__main__':
    logger.info(f"Starting RL Environment Server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
