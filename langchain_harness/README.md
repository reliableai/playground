# Our own harness on Terminal-Bench, built with LangChain

A benchmark tests an **agent**: a **harness** (the program that loops, prompts the model and runs its tool calls) plus a **model**. On Terminal-Bench you can bring both. This folder is a harness we build ourselves, with a LangChain agent, and the command that points Terminal-Bench at it.

Terminal-Bench is run by **Harbor**, an open-source program: for each attempt it starts the task's containers, hands our harness the task's instruction, stops it at the time limit, and then runs the task's tests in a separate container. Harbor finds our harness through `-a langchain_harness:LangChainAgent`: the file `langchain_harness.py`, the class `LangChainAgent`.

## Files

| File | What it is |
|---|---|
| `langchain_harness.py` | The harness: a LangChain agent with one tool, `run_shell`, wrapped in the class Harbor calls. |
| `test_harness.py` | Tests with a scripted fake model and a fake container: no API key, no cost. |
| `pyproject.toml` | The three packages it needs: `harbor`, `langchain`, `langchain-openai`. |

`../minimal_agent/minimal_harbor_agent.py` does the same job with a hand-written loop instead of LangChain's.

## How the harness works

Harbor calls two methods of `LangChainAgent` on each attempt:

- `setup(environment)` does nothing: the harness runs on your machine, in Harbor's own process, and nothing is installed in the task's container.
- `run(instruction, environment, context)` builds LangChain's standard tool-calling agent (`create_agent`) and runs it on the instruction. The model either calls `run_shell`, whose command runs inside the task's `main` container through Harbor's `environment.exec`, or answers; an answer without a tool call ends the run. Token counts go into Harbor's `context` as they come in, so they survive a crash or the time limit.

The model is whatever you pass with `-m`, as an OpenRouter model id; Harbor hands it to the harness unchanged, and the harness calls it through OpenRouter with `OPENROUTER_API_KEY` from your environment.

## Set up and test

Every command runs inside `playground/langchain_harness`. It needs Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m unittest test_harness.py
```

The tests check that the model's command runs in the container, that its output goes back to the model, that a reply without a tool call ends the run, that token counts are recorded even when a run fails, that a missing key or model fails loudly, and that Harbor builds the harness from the command line's `-a` and `-m`.

## Run it on Terminal-Bench

This needs Docker running and an OpenRouter key, and it costs money: the full benchmark is 66 tasks, each allowed up to 8 hours. Start with one task and one attempt:

```bash
export OPENROUTER_API_KEY="..."
PYTHONPATH=. uv run harbor run -d terminal-bench/terminal-bench@4.0.0 -a langchain_harness:LangChainAgent -m <OpenRouter model id> -i legacy-utility-triage -k 1
```

Then the full run, 5 attempts per task, as on the leaderboard:

```bash
PYTHONPATH=. uv run harbor run -d terminal-bench/terminal-bench@4.0.0 -a langchain_harness:LangChainAgent -m <OpenRouter model id> -k 5
```

`PYTHONPATH=.` lets Harbor import `langchain_harness.py` from this folder. `-i` picks tasks by name and `-k` sets the attempts per task. The 4.0 leaderboard does not take outside submissions, so this score is your own.
