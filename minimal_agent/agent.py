"""A minimal coding agent: a model proposes shell commands, this program runs them.

The whole agent is the loop in run_agent():
  1. send the model the task, through its API
  2. the model replies with one shell command
  3. run the command in the task's folder and send the model what it printed
  4. repeat until the model replies without a command, or the step limit is reached

Usage:  python agent.py <folder> "<instruction>"
The agent runs every command it is given inside <folder>, on your machine.
"""
import os
import re
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """You are an agent working in a Linux shell to complete a task.
Each reply must contain exactly one shell command, in a ```bash code block.
You will get back the command's exit code and output; use them to choose the next command.
When the task is complete, reply without any code block, saying briefly what you did."""

MAX_STEPS = 20
COMMAND_TIMEOUT_SECONDS = 60
COMMAND_BLOCK = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.DOTALL)


def required_setting(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set: add it to .env (see .env.example).")
    return value


def make_client():
    """An OpenAI-SDK client pointed at OpenRouter, and the model to use, both from .env."""
    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=required_setting("OPENROUTER_API_KEY"))
    return client, required_setting("OPENROUTER_MODEL")


def ask_model(client, model, messages):
    response = client.chat.completions.create(model=model, messages=messages)
    choice = response.choices[0]
    if choice.finish_reason == "length" or not choice.message.content:
        raise RuntimeError(f"The model returned no usable reply (finish_reason={choice.finish_reason}).")
    return choice.message.content


def run_command(command, workdir):
    try:
        result = subprocess.run(command, shell=True, cwd=workdir, capture_output=True, text=True,
                                timeout=COMMAND_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return f"The command timed out after {COMMAND_TIMEOUT_SECONDS} seconds."
    return f"exit code {result.returncode}\n{result.stdout}{result.stderr}"


def run_agent(instruction, workdir, client, model):
    """Run the loop on one task; the returned conversation is the agent's trajectory."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}]
    for step in range(MAX_STEPS):
        reply = ask_model(client, model, messages)
        messages.append({"role": "assistant", "content": reply})
        commands = COMMAND_BLOCK.findall(reply)
        if not commands:
            print(f"Agent: {reply.strip()}")
            return messages
        if len(commands) > 1:
            output = (f"Error: your reply had {len(commands)} code blocks. "
                      "Send exactly one command per reply.")
        else:
            print(f"$ {commands[0].strip()}")
            output = run_command(commands[0], workdir)
        print(output)
        messages.append({"role": "user", "content": output})
    print(f"Stopped: the step limit of {MAX_STEPS} was reached.")
    return messages


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python agent.py <folder> "<instruction>"')
    client, model = make_client()
    run_agent(sys.argv[2], sys.argv[1], client, model)
