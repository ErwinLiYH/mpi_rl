import re

default_system_prompt_zh = ""

default_system_prompt_en = """\
"""

default_obs_none_zh = "当前为初始步骤,所以没有observation"

default_obs_none_en = ""

def safe_escape(text):
    # This function is used to escape the curly braces in the text
    # by doubling them, so that they are not interpreted as format specifiers
    return re.sub(r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})', lambda m: m.group() * 2, text)

class ActTemplate:
    def __init__(self, system_desc:str, act_prompt:str):
        self.template = """\
{system_desc}
{partial_episode}
{obs}
{act_prompt}""".format(
            system_desc = safe_escape(system_desc),
            act_prompt = safe_escape(act_prompt),
            obs = "{obs}",
            partial_episode = "{partial_episode}"
        )

    def format(self, obs:str, partial_episode:str) -> str:
        return self.template.format(
            obs = safe_escape(obs),
            partial_episode = safe_escape(partial_episode)
        )

class ReflexTemplate:
    def __init__(self, system_desc:str, reflex_prompt:str):
        self.template = """\
{system_desc}
{episode}
{episode_metrics}
{reflex_prompt}""".format(
            system_desc = safe_escape(system_desc),
            reflex_prompt = safe_escape(reflex_prompt),
            episode = "{episode}",
            episode_metrics = "{episode_metrics}"
        )

    def format(self, episode:str, episode_metrics:str) -> str:
        return self.template.format(
            episode = safe_escape(episode),
            episode_metrics = safe_escape(episode_metrics)
        )
    
