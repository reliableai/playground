"""A one-task benchmark: run the agent several times on the task, check each trial with the hidden tests.

Each trial:
  1. copies task/environment/ into a fresh temporary folder (all the agent is given)
  2. runs the agent there on task/instruction.md
  3. after the agent stops, runs task/tests/test_export.py on the folder (the agent never gets the tests)
  4. reward 1 if every test passes, else 0
The score is the share of trials with reward 1.

Usage:
  python run_eval.py              # the agent, 5 trials (needs .env)
  python run_eval.py --trials 10
  python run_eval.py --oracle     # the reference solution instead of the agent (no API key needed)
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from agent import make_client, run_agent

TASK = Path(__file__).parent / "task"


def solve_with_oracle(workdir):
    subprocess.run(["bash", str(TASK / "solution" / "solve.sh")], cwd=workdir, check=True)


def run_trial(solve):
    """One trial: a fresh copy of the environment, the solver, then the hidden tests. Returns the reward."""
    with tempfile.TemporaryDirectory() as temporary_folder:
        workdir = Path(temporary_folder) / "app"
        shutil.copytree(TASK / "environment", workdir)
        solve(workdir)
        tests = subprocess.run([sys.executable, str(TASK / "tests" / "test_export.py"), str(workdir)],
                               capture_output=True, text=True)
    if tests.returncode not in (0, 1):
        raise RuntimeError(f"The hidden tests crashed:\n{tests.stdout}{tests.stderr}")
    print("Hidden tests:\n" + tests.stdout)
    return 1 if tests.returncode == 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=5, help="how many times to run the task (default 5)")
    parser.add_argument("--oracle", action="store_true", help="run the reference solution instead of the agent")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    if args.oracle:
        solve = solve_with_oracle
    else:
        client, model = make_client()
        instruction = (TASK / "instruction.md").read_text()
        print(f"Model: {model}")
        solve = lambda workdir: run_agent(instruction, workdir, client, model)

    rewards = []
    for trial_number in range(1, args.trials + 1):
        print(f"\n=== Trial {trial_number} of {args.trials} ===")
        rewards.append(run_trial(solve))
        print(f"Reward: {rewards[-1]}")
    passed = sum(rewards)
    print(f"\n=== Score: {passed} of {args.trials} trials passed ({passed / args.trials:.0%}) ===")


if __name__ == "__main__":
    main()
