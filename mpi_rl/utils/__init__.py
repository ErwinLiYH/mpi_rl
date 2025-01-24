import gymnasium as gym
from pettingzoo import ParallelEnv
from pettingzoo import AECEnv

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