import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_coordination import make_check_plan, make_independent_plan, parse_assignment, run_one, score_centralized, run_independent_trial
from analyze_matched_study import score_trial
from robot_demo.engine import validate_scenario
from robot_demo.prompts import build_agent_prompt
from diagnose_coordination import run_check
from argparse import Namespace


class DiagnosticTests(unittest.TestCase):
    def test_resume_after_identical_model_download_preserves_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "plan.json"
            path.write_text(json.dumps(self.plan), encoding="utf-8")
            before = path.read_bytes()
            args = Namespace(reference=ROOT / "outputs/matched-study/plan.json", model=self.plan["modelInfo"]["name"],
                output=output, reasoning_effort=None, thinking=False, max_tokens=None,
                thinking_profile="qwen", no_thinking=False, temperature=None, stop_after=0)
            info = dict(self.plan["modelInfo"], modified_at="new download time")
            with patch("diagnose_coordination.local_model", return_value=(info, self.plan["ollamaVersion"])):
                run_check(args)
            self.assertEqual(path.read_bytes(), before)
            info["digest"] = "different weights"
            with patch("diagnose_coordination.local_model", return_value=(info, self.plan["ollamaVersion"])):
                with self.assertRaises(ValueError):
                    run_check(args)

    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((ROOT / "outputs/matched-study/plan.json").read_text(encoding="utf-8"))
        cls.plan = make_check_plan(cls.reference, cls.reference["modelInfo"], cls.reference["ollamaVersion"])

    def test_selected_subset_and_original_prompts_unchanged(self):
        self.assertEqual(len(self.plan["trials"]), 36)
        self.assertEqual({t["rotation"] for t in self.plan["trials"]}, {0})
        for trial in self.reference["trials"]:
            self.assertEqual([build_agent_prompt(trial["scenario"], i) for i in range(6)], trial["prompts"])
        for m in (2, 3, 4):
            self.assertEqual(len([t for t in self.plan["trials"] if t["m"] == m]), 12)

    def test_parser_separates_format_errors_from_conflicting_decisions(self):
        self.assertEqual(parse_assignment('{"goals":["A","A","A","A","A","A"]}'), [0]*6)
        for value in ({"goals": ["A"]}, {"goals": [True]*6}, {"goals": ["AB"]*6},
                      {"goals": ["A"]*6, "explanation": "test"}, {"goal": "A"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_assignment(json.dumps(value))

    def test_reasoning_profile_preserves_cases_and_is_sent_to_ollama(self):
        plan = make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"], thinking=True)
        self.assertEqual(plan["trials"], self.plan["trials"])
        self.assertEqual(plan["options"]["num_predict"], 4096)
        trial = plan["trials"][0]
        value = {"done": True, "done_reason": "length", "message": {"content": "", "thinking": "unfinished"}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("diagnose_coordination.request", return_value=(value, json.dumps(value))) as request:
                record = run_one(trial, plan, Path(directory) / "record.json")
                self.assertTrue(request.call_args.args[1]["think"])
                self.assertEqual(request.call_args.kwargs["timeout"], 600)
            self.assertEqual(record["rawThinking"], "unfinished")
            self.assertEqual(score_centralized(trial, record, plan)["invalidResponses"], 1)

    def test_saved_response_scoring_and_resume(self):
        trial = self.plan["trials"][0]
        target = validate_scenario(trial["scenario"])["oracleAssignment"]
        value = {"done": True, "message": {"content": json.dumps({"goals": ["ABCDEF"[i] for i in target]})}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with patch("diagnose_coordination.request", return_value=(value, json.dumps(value))) as request:
                record = run_one(trial, self.plan, path)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(score_centralized(trial, record, self.plan)["success"], 1)
            with patch("diagnose_coordination.request") as request:
                self.assertEqual(run_one(trial, self.plan, path), record)
                request.assert_not_called()
            record["choices"][0] = (record["choices"][0] + 1) % 6
            with self.assertRaises(ValueError):
                score_centralized(trial, record, self.plan)

    def test_larger_budget_is_a_separate_plan_with_room_for_prompt(self):
        plan = make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"], thinking=True, max_tokens=16384)
        self.assertEqual(plan["trials"], self.plan["trials"])
        self.assertEqual(plan["options"]["num_predict"], 16384)
        self.assertGreater(plan["options"]["num_ctx"], 16384)
        self.assertNotEqual(plan["sha256"], self.plan["sha256"])
        self.assertEqual(self.reference["options"]["num_predict"], 256)
        gemma = make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"], thinking=True, max_tokens=8192, thinking_profile="gemma")
        self.assertEqual(gemma["trials"], plan["trials"])
        self.assertEqual(gemma["options"]["top_k"], 64)
        self.assertNotIn("presence_penalty", gemma["options"])

    def test_invalid_output_is_saved_as_failure(self):
        trial = self.plan["trials"][0]
        value = {"done": True, "message": {"content": '{"goals":["A"]}'}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("diagnose_coordination.request", return_value=(value, json.dumps(value))):
                record = run_one(trial, self.plan, Path(directory) / "record.json")
            result = score_centralized(trial, record, self.plan)
            self.assertEqual(result["success"], 0)
            self.assertEqual(result["invalidResponses"], 1)
            self.assertEqual(result["duplicateGoals"], 0)

    def test_independent_check_uses_same_cases_but_isolated_original_prompts(self):
        central = make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"], max_tokens=2048, disable_thinking=True)
        plan = make_independent_plan(central, self.reference)
        self.assertEqual([t["scenario"] for t in plan["trials"]], [t["scenario"] for t in central["trials"]])
        trial = plan["trials"][0]
        target = validate_scenario(trial["scenario"])["oracleAssignment"]
        replies = [{"done": True, "message": {"content": json.dumps({"goal": "ABCDEF"[g]})}} for g in target]
        with tempfile.TemporaryDirectory() as directory:
            with patch("run_matched_study.request", side_effect=[(v, json.dumps(v)) for v in replies]) as request:
                record = run_independent_trial(trial, plan, Path(directory) / "record.json")
            self.assertEqual(len(request.call_args_list), 6)
            for i, call in enumerate(request.call_args_list):
                self.assertFalse(call.args[1]["think"])
                self.assertEqual(call.args[1]["messages"], [{"role": "user", "content": trial["prompts"][i]}])
            self.assertEqual(score_trial(trial, record, plan)["success"], 1)
            record["agents"][0]["think"] = True
            with self.assertRaises(ValueError):
                score_trial(trial, record, plan)

    def test_named_reasoning_level_is_preserved_in_request(self):
        plan = make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"], reasoning_effort="medium")
        self.assertEqual(plan["trials"], self.plan["trials"])
        self.assertEqual(plan["think"], "medium")
        value = {"done": True, "message": {"content": '{"goals":["A","B","C","D","E","F"]}'}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("diagnose_coordination.request", return_value=(value, json.dumps(value))) as request:
                record = run_one(plan["trials"][0], plan, Path(directory) / "record.json")
                self.assertEqual(request.call_args.args[1]["think"], "medium")
            score_centralized(plan["trials"][0], record, plan)

    def test_deepseek_profile_is_paired_without_changing_prompts(self):
        plan = make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"],
                               thinking=True, thinking_profile="deepseek")
        paired = make_independent_plan(plan, self.reference)
        self.assertEqual(plan["trials"], self.plan["trials"])
        self.assertEqual(paired["options"], plan["options"])
        self.assertTrue(paired["think"])
        self.assertEqual(plan["options"]["temperature"], 0.6)
        self.assertEqual(plan["options"]["top_p"], 0.95)
        self.assertEqual(plan["options"]["num_predict"], 8192)
        self.assertNotIn("presence_penalty", plan["options"])
        value = {"done": True, "message": {"content": '{"goals":["A","B","C","D","E","F"]}'}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("diagnose_coordination.request", return_value=(value, json.dumps(value))) as request:
                run_one(plan["trials"][0], plan, Path(directory) / "record.json")
                self.assertTrue(request.call_args.args[1]["think"])
                self.assertEqual(request.call_args.args[1]["options"]["temperature"], 0.6)
        with self.assertRaises(ValueError):
            make_check_plan(self.reference, self.reference["modelInfo"], self.reference["ollamaVersion"],
                            disable_thinking=True, thinking_profile="deepseek")

    def test_self_cost_compliance_does_not_imply_joint_success(self):
        trial = self.plan["trials"][0]
        scenario = trial["scenario"]
        target = validate_scenario(scenario)["oracleAssignment"]
        i, goal = next((i, g) for i in range(6) for g in range(6)
            if target[i] == scenario["initial"][i] and g != target[i]
            and scenario["costs"][i][g] < scenario["costs"][i][target[i]])
        target[i] = goal
        value = {"done": True, "message": {"content": json.dumps({"goals": ["ABCDEF"[g] for g in target]})}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("diagnose_coordination.request", return_value=(value, json.dumps(value))):
                record = run_one(trial, self.plan, Path(directory) / "record.json")
            row = score_centralized(trial, record, self.plan)
        self.assertEqual(row["selfCostCompliance"], 1)
        self.assertEqual(row["duplicateGoals"], 1)
        self.assertEqual(row["success"], 0)


if __name__ == "__main__":
    unittest.main()
