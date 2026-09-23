import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_demo.engine import resolve_choices, shortest_path, validate_scenario
from robot_demo.storage import save_json


def make_demo(scenario, trace):
    if trace.get("source") != "local-llm" or trace.get("status") != "completed":
        raise ValueError("A completed local-model record is required.")
    choices = trace.get("choices")
    if not isinstance(choices, list) or len(choices) != 6 or any(type(c) is not int or not 0 <= c < 6 for c in choices):
        raise ValueError("The record has missing or invalid choices.")
    analysis = validate_scenario(scenario)
    result = resolve_choices(scenario, choices)
    return {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "scenario": scenario,
            "analysis": analysis, "run": {"source": "local-llm", "model": trace["model"],
            "trialId": trace["id"], "choices": choices, "result": result},
            "paths": {"before": [shortest_path(scenario, r, goal) for r, goal in enumerate(scenario["initial"])],
                      "after": [shortest_path(scenario, r, goal) for r, goal in enumerate(result["assignment"])]},
            "limitations": ["Paths ignore robot-robot collisions."]}


def main():
    parser = argparse.ArgumentParser(description="Export a recorded trial for replay.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8-sig"))
    save_json(args.output, make_demo(trace["scenario"], trace))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
