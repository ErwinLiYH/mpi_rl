default_system_prompt_zh = ""

default_system_prompt_en = """\
"""

default_obs_none_zh = "当前为初始步骤,所以没有observation"

default_obs_none_en = ""

class ActTemplate:
    def __init__(self, system_desc:str, act_prompt:str):
        self.template = """\
{system_desc}
{obs}
{act_prompt}""".format(
            system_desc = system_desc,
            act_prompt = act_prompt
        )

    def format(self, obs:str) -> str:
        return self.template.format(
            obs = obs
        )

class ReflexTemplate:
    def __init__(self, system_desc:str, reflex_prompt:str):
        self.template = """\
{system_desc}
{episode}
{reflex_prompt}""".format(
            system_desc = system_desc,
            reflex_prompt = reflex_prompt
        )

    def format(self, episode:str) -> str:
        return self.template.format(
            episode = episode
        )