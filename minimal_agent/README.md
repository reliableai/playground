# Minimal agent — a real agent on a one-task benchmark

**Lesson: Hill-Climbing (benchmarks)**

The code companion to the *Climbing the Benchmark* playground. The playground simulates an agent fixing a nightly export; here the same task is real. You get a coding agent of under 90 lines of Python, comments included, the task folder, and a small eval runner that scores the agent the way an agent benchmark such as Terminal-Bench does: run it several times, check each attempt with hidden tests, report the share that passed.

---

## Files

| File | What it is |
|------|-----------|
| `agent.py` | The agent: a loop that asks a model for shell commands and runs them. |
| `run_eval.py` | The benchmark runner: trials, hidden tests, rewards, score. |
| `task/instruction.md` | What the agent is asked to do. |
| `task/environment/` | The files the agent works on: `export.py` and its data, `orders.json`. |
| `task/tests/` | The hidden tests, `test_export.py`, and the answer fixed in advance, `expected.csv`. The agent is never given them. |
| `task/solution/solve.sh` | The reference solution, which shows the task can be solved. |
| `test_lab.py` | Tests for this lab, with a scripted fake model: no API key, no cost. |
| `pyproject.toml` | The two Python packages the lab needs, `openai` and `python-dotenv`. |
| `.env.example` | Template for the OpenRouter key and model. |

---

## The agent, in four steps

An agent is an ordinary program running a loop (`run_agent` in `agent.py`):

1. It sends the model the task, through OpenRouter's API, with your key.
2. The model replies with one shell command, for example `cat export.py`.
3. The program runs the command in the task's folder and sends the model the exit code and what it printed.
4. Repeat, until the model replies without a command (it thinks it is done) or 20 steps have passed.

The model never touches your computer: it only writes text. The program decides to run what it wrote. The agents on real leaderboards, such as Claude Code or Codex, are this same loop with more tools and longer prompts.

## The benchmark, in four steps

Each **trial** (`run_trial` in `run_eval.py`):

1. Copies `task/environment/` into a fresh temporary folder. This is all the agent is given.
2. Runs the agent there on `task/instruction.md`.
3. After the agent stops, runs the hidden tests on that folder. They run `export.py` and compare its output with `expected.csv`, column by column: row count, dates, everything else.
4. Gives **reward 1** if every test passes, otherwise **0**.

The **score** is the share of trials with reward 1. Five trials by default, because the same agent on the same task does not always do the same thing.

---

## Setup

Install [uv](https://docs.astral.sh/uv/) once. Then get the code and make your `.env`:

```bash
git clone https://github.com/reliableai/playground.git
cd playground/minimal_agent
cp -n .env.example .env
```

Edit `.env`: set `OPENROUTER_API_KEY` to your key from openrouter.ai, and `OPENROUTER_MODEL` to the model to test.

Every command below runs inside `playground/minimal_agent`. The first `uv run` installs the two packages into a local `.venv` folder; later runs reuse it.

> **Warning: the agent runs real commands on your computer.** Whatever the model writes, `agent.py` runs in the task's temporary folder, with your user's permissions. Nothing stops a command from reaching outside that folder. Read the commands as they scroll by, and stop the run (Ctrl-C) if one looks wrong. Real benchmarks run the agent inside a container: a separate, disposable computer.

---

## Step 1: check the lab, without a model

No key needed and nothing is spent. These show that the tests work before any model is involved.

```bash
uv run python -m unittest test_lab.py
```

```bash
uv run python run_eval.py --oracle
```

`--oracle` runs the reference solution instead of the agent. Every trial should get reward 1: the task can be solved, and the tests accept a correct answer. (`test_lab.py` also checks that the untouched export fails the date test and nothing else.)

## Step 2: run the agent once, by hand

Give the agent a scratch copy of the task, then look at what it did:

```bash
cp -r task/environment /tmp/agent_try
uv run python agent.py /tmp/agent_try "$(cat task/instruction.md)"
python3 /tmp/agent_try/export.py
```

You see every command the model chose (lines starting with `$`), what each printed, and the model's last message. The last line prints the export as the agent left it.

## Step 3: run the eval

```bash
uv run python run_eval.py
```

For each of the 5 trials you see the agent's commands, then the hidden tests (`PASS` or `FAIL` for each), then the reward. The last line is the score, for example `Score: 4 of 5 trials passed (80%)`. Use `--trials 10` for more trials.

---

## How to read the score

- **It is about one task.** The score says how often this model, inside this agent, fixed this export. It says nothing about other tasks; a benchmark has many tasks for that reason.
- **Five trials are few.** 4 passes out of 5 is compatible with a true pass rate well below 80% or close to 100%. Put your counts into the [accuracy estimator](https://reliableai.github.io/playground/accuracy_estimator/index.html) to see how wide the range is.
- **The score belongs to the pair.** Change the model in `.env`, or the prompt in `agent.py`, and you measure a different system.
- **Hidden only by convention.** The agent is never given `task/tests/`, but it runs on your computer and could find the folder if it searched. Real benchmarks keep the tests out of the agent's container and run them only after the agent stops.

## Things to try

1. Run the eval with two different models and compare the scores, with their ranges.
2. Read a failed trial: which test failed, and what did the agent do?
3. Copy `task/tests/` into the trial's folder before the agent starts (in `run_trial`), and tell the agent in the instruction. Does the score change? What does it now measure?
