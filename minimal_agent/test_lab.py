"""Tests for the lab, with no API calls: a scripted fake model stands in for the real one.

Run from this folder:  python -m unittest test_lab.py
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agent
from run_eval import TASK, run_trial, solve_with_oracle

FIX_DATES = "```bash\npython3 -c \"from pathlib import Path; p = Path('export.py'); " \
            "p.write_text(p.read_text().replace('%m/%d/%Y', '%Y-%m-%d'))\"\n```"


class ScriptedModel:
    """Looks like an OpenAI client to the agent; returns the given replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages):
        self.received.append(list(messages))
        reply = self.replies.pop(0)
        choice = SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")
        return SimpleNamespace(choices=[choice])


def hidden_tests(folder):
    return subprocess.run([sys.executable, str(TASK / "tests" / "test_export.py"), str(folder)],
                          capture_output=True, text=True)


class HiddenTestsTest(unittest.TestCase):
    def test_untouched_environment_fails_only_the_date_test(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copytree(TASK / "environment", Path(folder) / "app")
            tests = hidden_tests(Path(folder) / "app")
        self.assertEqual(tests.returncode, 1)
        self.assertIn("FAIL test_dates_are_iso", tests.stdout)
        self.assertIn("PASS test_row_count", tests.stdout)
        self.assertIn("PASS test_other_columns_unchanged", tests.stdout)

    def test_oracle_trial_gets_reward_1(self):
        self.assertEqual(run_trial(solve_with_oracle), 1)

    def test_broken_export_fails_every_test(self):
        def break_export(workdir):
            (workdir / "export.py").write_text("raise SystemExit(3)\n")
        self.assertEqual(run_trial(break_export), 0)

    def test_split_customer_column_fails_other_columns(self):
        def unquoted_join(workdir):
            export = workdir / "export.py"
            export.write_text(export.read_text().replace("%m/%d/%Y", "%Y-%m-%d").replace(
                "writer = csv.writer(sys.stdout)", "writer = csv.writer(sys.stdout, quoting=csv.QUOTE_NONE, escapechar=' ')"))
        with tempfile.TemporaryDirectory() as folder:
            shutil.copytree(TASK / "environment", Path(folder) / "app")
            unquoted_join(Path(folder) / "app")
            tests = hidden_tests(Path(folder) / "app")
        self.assertIn("FAIL test_other_columns_unchanged", tests.stdout)


class AgentLoopTest(unittest.TestCase):
    def test_scripted_agent_solves_the_task(self):
        model = ScriptedModel([FIX_DATES, "I changed both date formats to ISO."])
        reward = run_trial(lambda workdir: agent.run_agent("Fix the dates.", workdir, model, "scripted"))
        self.assertEqual(reward, 1)

    def test_command_output_goes_back_to_the_model(self):
        model = ScriptedModel(["```bash\necho hello\n```", "Done."])
        with tempfile.TemporaryDirectory() as folder:
            messages = agent.run_agent("Say hello.", folder, model, "scripted")
        self.assertEqual(messages[3], {"role": "user", "content": "exit code 0\nhello\n"})
        self.assertEqual(model.received[1][-1]["content"], "exit code 0\nhello\n")
        self.assertEqual(len(messages), 5)

    def test_commands_run_in_the_given_folder(self):
        model = ScriptedModel(["```bash\necho hi > note.txt\n```", "Done."])
        with tempfile.TemporaryDirectory() as folder:
            agent.run_agent("Write a note.", folder, model, "scripted")
            self.assertEqual((Path(folder) / "note.txt").read_text(), "hi\n")

    def test_two_commands_in_one_reply_get_an_error_back(self):
        model = ScriptedModel(["```bash\necho a\n```\n```bash\necho b\n```", "Done."])
        with tempfile.TemporaryDirectory() as folder:
            messages = agent.run_agent("Task.", folder, model, "scripted")
        self.assertIn("had 2 code blocks", messages[3]["content"])

    def test_agent_stops_at_the_step_limit(self):
        model = ScriptedModel(["```bash\ntrue\n```"] * agent.MAX_STEPS)
        with tempfile.TemporaryDirectory() as folder:
            messages = agent.run_agent("Loop.", folder, model, "scripted")
        self.assertEqual(len(messages), 2 + 2 * agent.MAX_STEPS)

    def test_slow_command_times_out(self):
        with mock.patch.object(agent, "COMMAND_TIMEOUT_SECONDS", 1), tempfile.TemporaryDirectory() as folder:
            self.assertEqual(agent.run_command("sleep 3", folder), "The command timed out after 1 seconds.")

    def test_empty_reply_raises(self):
        model = ScriptedModel([None])
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(RuntimeError):
            agent.run_agent("Task.", folder, model, "scripted")

    def test_missing_setting_stops_with_a_message(self):
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(SystemExit) as stop:
            agent.required_setting("OPENROUTER_API_KEY")
        self.assertIn("OPENROUTER_API_KEY is not set", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
