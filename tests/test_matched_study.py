import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from robot_demo.engine import validate_scenario, resolve_choices
from run_matched_study import make_plan, parse_goal, relabel, run_one, verify_plan
from analyze_matched_study import bootstrap_family_means, score_trial
from export_demo import make_demo


class MatchedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads((ROOT / "data/matched-suite.json").read_text())
        cls.plan = make_plan(cls.suite, {"name": "test-local", "digest": "test"}, {"version": "test"})

    def test_generated_plan_matches_recorded_prompts_and_settings(self):
        saved = json.loads((ROOT / "outputs/matched-study/plan.json").read_text(encoding="utf-8"))
        rebuilt = make_plan(self.suite, saved["modelInfo"], saved["ollamaVersion"], saved["planSeed"])
        for key in saved:
            if key not in ("createdAt", "sha256"):
                self.assertEqual(rebuilt[key], saved[key], key)

    def test_export_preserves_recorded_choices_and_paths(self):
        expected = json.loads((ROOT / "outputs/matched-study/real-trial.json").read_text(encoding="utf-8"))
        path = ROOT / "outputs/matched-study/records" / (expected["run"]["trialId"] + ".json")
        trace = json.loads(path.read_text(encoding="utf-8"))
        actual = make_demo(trace["scenario"], trace)
        for key in ("scenario", "analysis", "paths"):
            self.assertEqual(actual[key], expected[key])
        for key in ("source", "model", "trialId", "choices", "result"):
            self.assertEqual(actual["run"][key], expected["run"][key])

    def test_exact_controls_and_independent_exhaustive_validation(self):
        for family in self.suite["families"]:
            conditions = family["conditions"]
            for field in ("initialTotal", "meanGain", "improvingEdges"):
                self.assertEqual(len({item["audit"][field] for item in conditions}), 1)
            self.assertEqual({c["scenario"]["m"] for c in conditions}, {2, 3, 4})
            for condition in conditions:
                s = condition["scenario"]
                self.assertEqual(s["costs"], conditions[0]["scenario"]["costs"])
                self.assertTrue(all(v >= 0 for row in s["costs"] for v in row))
                analysis = validate_scenario(s)
                self.assertEqual(len(analysis["beneficialAssignments"]), 1)
                self.assertEqual(analysis["oracleAssignment"], condition["audit"]["target"])
                self.assertTrue(resolve_choices(s, analysis["oracleAssignment"])["success"])

    def test_rotations_balance_every_physical_robot_and_goal(self):
        for family in self.suite["families"]:
            trials = [t for t in self.plan["trials"] if t["familyId"] == family["id"] and t["m"] == 2]
            for field in ("robotOrder", "goalOrder"):
                for physical in range(6):
                    self.assertEqual({t[field].index(physical) for t in trials}, set(range(6)))

    def test_label_change_preserves_payoffs_and_target(self):
        condition = self.suite["families"][0]["conditions"][2]
        s = condition["scenario"]
        robots, goals = [4, 1, 5, 0, 2, 3], [2, 5, 0, 4, 1, 3]
        rotated = relabel(s, robots, goals)
        target = [goals.index(condition["audit"]["target"][i]) for i in robots]
        self.assertEqual(validate_scenario(rotated)["oracleAssignment"], target)
        self.assertEqual(resolve_choices(s, condition["audit"]["target"])["afterTotal"],
                         resolve_choices(rotated, target)["afterTotal"])

    def test_parser_rejects_substrings_types_and_extra_text(self):
        self.assertEqual(parse_goal('{"goal":"F"}'), 5)
        for raw in ('{"goal":"AB"}', '{"goal":""}', '{"goal":true}', '{"goal":[]}',
                    '{"goal":1}', '{"goal":"A","reason":"ok"}', 'choose A', '["A"]'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                parse_goal(raw)

    def test_tampering_invalidates_frozen_plan(self):
        changed = dict(self.plan, planSeed=-1)
        with self.assertRaises(ValueError):
            verify_plan(changed)

    def test_execution_comes_from_responses_and_completed_trials_are_not_retried(self):
        trial = self.plan["trials"][0]
        target = validate_scenario(trial["scenario"])["oracleAssignment"]
        replies = []
        for choice in target:
            value = {"done": True, "message": {"content": json.dumps({"goal": "ABCDEF"[choice]})}}
            replies.append((value, json.dumps(value)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with patch("run_matched_study.request", side_effect=replies) as mock:
                record = run_one(trial, self.plan, path)
                self.assertEqual(mock.call_count, 6)
                self.assertTrue(record["result"]["success"])
                self.assertEqual(score_trial(trial, record, self.plan)["participantAccuracy"], 1)
            with patch("run_matched_study.request") as mock:
                resumed = run_one(trial, self.plan, path)
                mock.assert_not_called()
                self.assertEqual(record, resumed)
            record["choices"][0] = (record["choices"][0] + 1) % 6
            with self.assertRaises(ValueError):
                score_trial(trial, record, self.plan)

    def test_infrastructure_interruption_keeps_completed_agent_decisions(self):
        trial = self.plan["trials"][0]
        value = {"done": True, "message": {"content": '{"goal":"A"}'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with patch("run_matched_study.request", side_effect=[(value, json.dumps(value)), OSError("offline")]):
                with self.assertRaises(OSError):
                    run_one(trial, self.plan, path)
            partial = json.loads(path.read_text())
            self.assertEqual(partial["choices"], [0])
            self.assertEqual(partial["status"], "interrupted")

    def test_constant_outcomes_do_not_get_false_zero_width_confidence(self):
        interval = bootstrap_family_means([0] * 12)
        self.assertEqual(interval["mean"], 0)
        self.assertIsNone(interval["low"])
        varied = bootstrap_family_means([0, .5, 1], draws=1000)
        self.assertLess(varied["low"], varied["mean"])
        self.assertGreater(varied["high"], varied["mean"])

    def test_pending_atomic_save_recovers_without_an_extra_model_call(self):
        trial = self.plan["trials"][0]
        value = {"done": True, "message": {"content": '{"goal":"A"}'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with patch("run_matched_study.request", return_value=(value, json.dumps(value))):
                record = run_one(trial, self.plan, path)
            pending = path.with_suffix(".json.tmp")
            pending.write_text(json.dumps(record))
            old = dict(record, agents=record["agents"][:5], choices=record["choices"][:5], status="running", result=None)
            path.write_text(json.dumps(old))
            with patch("run_matched_study.request") as mock:
                recovered = run_one(trial, self.plan, path)
                mock.assert_not_called()
                self.assertEqual(recovered, record)


if __name__ == "__main__":
    unittest.main()
