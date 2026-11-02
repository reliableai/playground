"""Tests for the LangChain harness, with a scripted fake model and a fake container: no API key, no cost.

Run from this folder:  uv run python -m unittest test_harness.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

import langchain_harness
from langchain_harness import LangChainAgent


class ScriptedModel(GenericFakeChatModel):
    """Replies with a fixed list of messages and remembers what it was sent."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def reply(content="", command=None, input_tokens=10, output_tokens=5):
    """One model reply: a run_shell call if a command is given, otherwise a final answer."""
    tool_calls = [{"name": "run_shell", "args": {"command": command}, "id": "call_1"}] if command else []
    return AIMessage(content=content, tool_calls=tool_calls, response_metadata={"model_name": "scripted"},
                     usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                     "total_tokens": input_tokens + output_tokens})


class FakeContainer:
    """Stands in for Harbor's environment: records commands and returns a fixed result."""

    def __init__(self, stdout="file.txt\n", return_code=0, error=None):
        self.commands = []
        self.result = ExecResult(stdout=stdout, stderr="", return_code=return_code)
        self.error = error

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if self.error:
            raise self.error
        return self.result


def scripted_agent(replies, logs_dir):
    model = ScriptedModel(messages=iter(replies), seen=[])
    agent = LangChainAgent(logs_dir=logs_dir, model_name="scripted/model")
    agent.make_model = lambda: model
    return agent, model


class LangChainHarnessTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logs_dir = Path(tempfile.mkdtemp())

    async def test_runs_the_models_command_in_the_container(self):
        agent, _ = scripted_agent([reply(command="ls"), reply(content="Done.")], self.logs_dir)
        container = FakeContainer()
        await agent.run("List the files.", container, AgentContext())
        self.assertEqual(container.commands, ["ls"])

    async def test_command_output_goes_back_to_the_model(self):
        agent, model = scripted_agent([reply(command="ls"), reply(content="Done.")], self.logs_dir)
        await agent.run("List the files.", FakeContainer(stdout="file.txt\n", return_code=0), AgentContext())
        tool_messages = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
        self.assertEqual([m.content for m in tool_messages], ["exit code 0\nfile.txt\n"])

    async def test_the_instruction_reaches_the_model(self):
        agent, model = scripted_agent([reply(content="Done.")], self.logs_dir)
        await agent.run("List the files.", FakeContainer(), AgentContext())
        self.assertEqual(model.seen[0][-1].content, "List the files.")

    async def test_a_reply_without_a_tool_call_ends_the_run(self):
        agent, model = scripted_agent([reply(content="Done."), reply(command="never")], self.logs_dir)
        container = FakeContainer()
        await agent.run("Do nothing.", container, AgentContext())
        self.assertEqual((container.commands, len(model.seen)), ([], 1))

    async def test_token_usage_is_recorded_in_the_context(self):
        agent, _ = scripted_agent([reply(command="ls", input_tokens=10, output_tokens=5),
                                   reply(content="Done.", input_tokens=20, output_tokens=3)], self.logs_dir)
        context = AgentContext()
        await agent.run("List the files.", FakeContainer(), context)
        self.assertEqual((context.n_input_tokens, context.n_output_tokens), (30, 8))

    async def test_usage_is_recorded_even_when_the_run_fails(self):
        agent, _ = scripted_agent([reply(command="ls", input_tokens=10, output_tokens=5)], self.logs_dir)
        context = AgentContext()
        with self.assertRaises(ConnectionError):
            await agent.run("List the files.", FakeContainer(error=ConnectionError("container gone")), context)
        self.assertEqual((context.n_input_tokens, context.n_output_tokens), (10, 5))

    def test_a_missing_key_fails_loudly(self):
        agent = LangChainAgent(logs_dir=self.logs_dir, model_name="some/model")
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                agent.make_model()

    def test_a_missing_model_fails_loudly(self):
        agent = LangChainAgent(logs_dir=self.logs_dir, model_name=None)
        with self.assertRaisesRegex(RuntimeError, "-m flag"):
            agent.make_model()

    def test_the_model_is_reached_through_openrouter(self):
        agent = LangChainAgent(logs_dir=self.logs_dir, model_name="some/model")
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            model = agent.make_model()
        self.assertEqual((model.model_name, model.openai_api_base),
                         ("some/model", langchain_harness.OPENROUTER_URL))

    def test_harbor_builds_the_agent_from_the_command_line(self):
        # What `harbor run -a langchain_harness:LangChainAgent -m anthropic/claude-sonnet-5` asks
        # Harbor to build. The OpenRouter model id must reach the agent unchanged.
        config = AgentConfig(name="langchain_harness:LangChainAgent", model_name="anthropic/claude-sonnet-5")
        AgentFactory.run_preflight(config)
        agent = AgentFactory.create_agent_from_config(config, logs_dir=self.logs_dir)
        self.assertEqual((type(agent), agent.model_name), (LangChainAgent, "anthropic/claude-sonnet-5"))


if __name__ == "__main__":
    unittest.main()
