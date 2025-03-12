from mpi_rl.mpi_rl.algorithm.llm.llm_utils import BaseInvoker
from mpi_rl.mpi_rl.algorithm.llm.llm_utils import Invoker_outputT, MessagesT

from typing import Callable, Any


class MultiAgentDebate(BaseInvoker):
    def __init__(self, invoker: BaseInvoker, n_agents: int, max_rounds: int,
                 filter_fn: Callable[[Invoker_outputT], Any] = lambda x: x[1]["action"],
                 zh=False):
        self.invoker = invoker
        self.n_agents = n_agents
        self.max_rounds = max_rounds
        self.filter_fn = filter_fn
        self.debate_prompt_en = "I provide you some solutions from other assistants, these solutions may wrong or correct, analyze whether they are correct and give me a updated response. the response should contain analysis and final result, final result is json format like: {\"answer\": \"the answer\"}"
        self.debate_prompt_zh = "以下是其他代理的解决方案，请参考后给出更新后的回答："
        self.zh = zh
        self.conversations = {f"agent_{i}": [] for i in range(n_agents)}
        self.actions_each_round = []

    def __call__(self, messages: MessagesT) -> Invoker_outputT:
        total_input_tokens = 0
        total_output_tokens = 0
        final_output = None

        for agent_id in self.conversations:
            self.conversations[agent_id] = []  # clear previous conversation
            self.conversations[agent_id].extend(messages.copy())

        self.actions_each_round = [] # clear previous actions

        for _ in range(self.max_rounds):
            outputs_this_round = {}
            actions_this_round = []

            for agent_id in self.conversations:
                agent_messages = self.conversations[agent_id]
                output = self.invoker(agent_messages)
                
                content, filtered_content, _, in_tokens, out_tokens = output
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                
                self.conversations[agent_id].append({
                    "role": "assistant",
                    "content": content
                })
                actions_this_round.append(self.filter_fn(filtered_content))
                outputs_this_round[agent_id] = output

            self.actions_each_round.append(actions_this_round)
            
            if self._check_consensus(actions_this_round):
                first_agent = list(outputs_this_round.keys())[0]
                final_output = outputs_this_round[first_agent]
                break

            for agent_id in self.conversations:
                other_agents = [a for a in self.conversations if a != agent_id]
                other_responses = [
                    (aid, self.conversations[aid][-1]["content"])
                    for aid in other_agents
                ]

                debate_content = "\n\n".join(
                    [f"{aid}:\n{resp}" for aid, resp in other_responses]
                )
                prompt = f"{self.debate_prompt_zh if self.zh else self.debate_prompt_en}\n{debate_content}"
                self.conversations[agent_id].append({
                    "role": "user",
                    "content": prompt
                })

        if final_output is None:
            first_agent = list(self.conversations.keys())[0]
            final_output = outputs_this_round[first_agent]

        return (
            final_output[0],
            final_output[1],
            final_output[2],
            total_input_tokens,
            total_output_tokens
        )

    def _check_consensus(self, actions: list) -> bool:
        return all(a == actions[0] for a in actions)