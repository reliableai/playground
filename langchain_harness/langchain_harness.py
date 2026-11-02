"""Our own harness, built on a LangChain agent, packaged so Harbor can run it.

For each attempt, Harbor creates LangChainAgent, starts the task's containers,
calls setup(), then run() with the task's instruction, and afterwards runs the
task's tests. The agent itself is LangChain's standard tool-calling loop
(create_agent): the model either calls a tool or answers, and the loop ends when
it answers without calling one. It has one tool, run_shell, which runs a
command inside the task's main container through Harbor's environment.exec.

The model is whatever Harbor's -m flag names, reached through OpenRouter with
OPENROUTER_API_KEY from your environment. The loop runs on your machine, in
Harbor's own process; nothing is installed in the container.

Run from this folder (see README.md):
  PYTHONPATH=. uv run harbor run -d terminal-bench/terminal-bench@4.0.0 \\
      -a langchain_harness:LangChainAgent -m <OpenRouter model id> -k 5
"""
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

SYSTEM_PROMPT = """You are an agent working in a Linux shell to complete a task.
Use the run_shell tool to run commands; it returns the exit code and the output.
When the task is complete, stop calling tools and say briefly what you did."""

OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_MODEL_CALLS = 200


def make_shell_tool(environment):
    """The agent's only tool: run one shell command inside the task's main container."""

    @tool
    async def run_shell(command: str) -> str:
        """Run a shell command in the task's Linux environment; returns the exit code and output."""
        result = await environment.exec(command)
        return f"exit code {result.return_code}\n{result.stdout or ''}{result.stderr or ''}"

    return run_shell


def record_usage(context, usage):
    """Copy the token counts LangChain collected, summed over models, into Harbor's context."""
    totals = list(usage.usage_metadata.values())
    context.n_input_tokens = sum(entry["input_tokens"] for entry in totals)
    context.n_output_tokens = sum(entry["output_tokens"] for entry in totals)


class LangChainAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "langchain-agent"

    def version(self) -> str | None:
        return "0.1.0"

    def make_model(self):
        """The chat model named by -m, reached through OpenRouter."""
        if not self.model_name:
            raise RuntimeError("No model given: pass an OpenRouter model id with Harbor's -m flag.")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        return ChatOpenAI(model=self.model_name, base_url=OPENROUTER_URL, api_key=api_key)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the agent runs in Harbor's process, on the host."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        agent = create_agent(self.make_model(), tools=[make_shell_tool(environment)],
                             system_prompt=SYSTEM_PROMPT)
        usage = UsageMetadataCallbackHandler()
        # Each model call and each tool call is one graph step; the limit caps model calls.
        config = {"callbacks": [usage], "recursion_limit": 2 * MAX_MODEL_CALLS + 1}
        try:
            await agent.ainvoke({"messages": [{"role": "user", "content": instruction}]}, config=config)
        finally:
            # Also on errors and on Harbor's time limit, so the counts are never lost.
            record_usage(context, usage)
