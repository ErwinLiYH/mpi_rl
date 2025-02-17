from typing import Literal, Dict, TypeAlias, List
from dataclasses import dataclass
from torch import nn


agent_name: TypeAlias = str
model_name: TypeAlias = str

@dataclass
class RolloutConfig:
    count_by: Literal["step", "episode"]
    explore:bool
    num_of_step:int|None = None
    num_of_episode:int|None = None
    complete_episode:bool=True
    pzp_episode_cut_method:Literal["strict_pzp", "any", "all"]="all"
    def __post_init__(self):
        if self.count_by == "step" and self.num_of_step is None:
            raise ValueError("num_of_step should be set if count_by is 'step'")
        if self.count_by == "episode" and self.num_of_episode is None:
            raise ValueError("num_of_episode should be set if count_by is 'episode'")
        if self.num_of_step is not None and self.num_of_episode is not None:
            raise ValueError("num_of_step and num_of_episode should not be set at the same time")
        assert self.pzp_episode_cut_method in ["strict_pzp", "any", "all"], (
            "pzp_episode_cut_method should be one of ['strict_pzp', 'any', 'all']"
        )
        assert self.count_by in ["step", "episode"], (
            "count_by should be one of ['step', 'episode']"
        )


@dataclass
class ModelConfig:
    agent_model_map: Dict[agent_name, model_name]|None          = None
    actors:          Dict[model_name, nn.Module]|None|nn.Module = None
    actor_targets:   Dict[model_name, nn.Module]|None|nn.Module = None
    critics:         Dict[model_name, nn.Module]|None|nn.Module = None
    critic_targets:  Dict[model_name, nn.Module]|None|nn.Module = None
    mixer:           None|nn.Module                             = None