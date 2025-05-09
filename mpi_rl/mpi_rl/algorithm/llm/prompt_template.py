import re
from typing import Literal
import textwrap

def safe_escape(text):
    # This function is used to escape the curly braces in the text
    # by doubling them, so that they are not interpreted as format specifiers
    return re.sub(r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})', lambda m: m.group() * 2, text)

class ActTemplate:
    def __init__(self, system_desc:str, act_prompt:str, language:Literal["zh", "en"] = "zh"):
        if language == "zh":
            self.template = textwrap.dedent(
                """\
                {system_desc}

                下面是按顺序排列的历史步骤:
                (如果为None则代表这是初始步,当前无历史记录)

                {partial_episode}

                下面是 step {step} 的观测:

                {obs}

                {act_prompt}
                """
            )
        elif language == "en":
            self.template =textwrap.dedent(
                """\
                {system_desc}

                Below is the history of steps in order:
                (If None, it means this is the initial step and there is no history)

                {partial_episode}
                
                Below is the observation for step {step}:

                {obs}

                {act_prompt}
                """ 
            )
        else:
            raise ValueError("Language must be either 'zh' or 'en'")
        
        # Formatting system_desc and act_prompt, leaving placeholders for obs and partial_episode
        self.template = self.template.format(
            system_desc = safe_escape(system_desc),
            act_prompt = safe_escape(act_prompt),
            obs = "{obs}",
            partial_episode = "{partial_episode}",
            step = "{step}"
        )

    def format(self, obs:str, partial_episode:str, step:str) -> str:
        return self.template.format(
            obs = safe_escape(obs),
            partial_episode = safe_escape(partial_episode),
            step = safe_escape(step)
        )

class ReflexTemplate:
    def __init__(self, system_desc:str, reflex_prompt:str, language:Literal["zh", "en"] = "zh"):
        
        if language == "zh":
            self.template = textwrap.dedent(
                """\
                {system_desc}

                下面是按时间顺序排列的一个episode中的完整步骤, 这一步主要关注用<<<< >>>包裹的片段Z:

                {episode}

                这是上面这个完整episode的性能指标统计值:

                {episode_metrics}

                {reflex_prompt}
                """
            )
        elif language == "en":
            self.template = textwrap.dedent(
                """\
                {system_desc}

                Below is the complete step in an episode arranged in chronological order, this step mainly focuses on the segment Z wrapped in <<<< >>>:

                {episode}

                This is the performance metric statistics of the complete episode above:

                {episode_metrics}

                {reflex_prompt}
                """
            )
        else:
            raise ValueError("Language must be either 'zh' or 'en'")

        self.template = self.template.format(
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
