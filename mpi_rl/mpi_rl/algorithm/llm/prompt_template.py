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
    
class TeachTemplate:
    def __init__(self, system_desc:str, teach_prompt:str):
        self.template = """\
{system_desc}
{partial_episode}
{reasoning_content}
{original_action_analysis}
{reflection}
{teach_prompt}""".format(
            system_desc = safe_escape(system_desc),
            teach_prompt = safe_escape(teach_prompt),
            reasoning_content = "{reasoning_content}",
            partial_episode = "{partial_episode}",
            original_action_analysis = "{original_action_analysis}",
            reflection = "{reflection}"
        )
    def format(self, partial_episode:str, reasoning_content: str, original_action_analysis:str, reflection:str) -> str:
        return self.template.format(
            partial_episode = safe_escape(partial_episode),
            reasoning_content = safe_escape(reasoning_content),
            original_action_analysis = safe_escape(original_action_analysis),
            reflection = safe_escape(reflection)
        )
