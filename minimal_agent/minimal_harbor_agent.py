"""The playground's minimal agent, as a harness Harbor can run on Terminal-Bench.

The same loop as agent.py, with four changes so Harbor can run it:
  - it is a class with setup() and run(), which Harbor calls for each attempt
  - each command runs inside the task's main container, through
    environment.exec, instead of in a folder on your machine
  - the model is whatever Harbor's -m flag names, as an OpenRouter model id
  - it uses the asynchronous client, so Harbor can run many attempts at once
It also records token usage in Harbor's context.

The loop runs on your machine, in Harbor's own process, and reads
OPENROUTER_API_KEY from your environment; nothing is installed in the container.

Usage, from this folder, with Harbor and the openai package installed:
  PYTHONPATH=. harbor run -d terminal-bench/terminal-bench@4.0.0 \\
      -a minimal_harbor_agent:MinimalAgent -m <OpenRouter model id> -k 5
"""
import os
import re

from openai import AsyncOpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

SYSTEM_PROMPT = """You are an agent working in a Linux shell to complete
a task. Each reply must contain exactly one shell command, in a ```bash
code block. You will get back the command's exit code and output; use
them to choose the next command. When the task is complete, reply
without any code block, saying briefly what you did."""

MAX_STEPS = 200
COMMAND_BLOCK = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.DOTALL)


class MinimalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "minimal-agent"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass  # nothing to install: this loop runs on the host

    async def run(self, instruction: str,
                  environment: BaseEnvironment,
                  context: AgentContext) -> None:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction}]
        context.n_input_tokens = 0
        context.n_output_tokens = 0
        for step in range(MAX_STEPS):
            response = await client.chat.completions.create(
                model=self.model_name, messages=messages)
            context.n_input_tokens += response.usage.prompt_tokens
            context.n_output_tokens += response.usage.completion_tokens
            reply = response.choices[0].message.content
            if not reply:
                raise RuntimeError("The model returned no usable reply.")
            messages.append({"role": "assistant", "content": reply})
            commands = COMMAND_BLOCK.findall(reply)
            if not commands:
                return  # no command: the agent says it is done
            if len(commands) > 1:
                output = ("Error: your reply had several code blocks. "
                          "Send exactly one command per reply.")
            else:
                result = await environment.exec(commands[0])
                output = (f"exit code {result.return_code}\n"
                          f"{result.stdout or ''}{result.stderr or ''}")
            messages.append({"role": "user", "content": output})
