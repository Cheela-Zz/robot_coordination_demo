from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_demo.engine import validate_scenario, resolve_choices
from robot_demo.storage import digest, save_json
from robot_demo.prompts import build_agent_prompt

LABELS = "ABCDEF"
PROMPT_VERSION = "matched-six-agent-v2"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Redirect refused: inference must stay on localhost.")


def request(endpoint, body=None, timeout=120):
    opener = build_opener(ProxyHandler({}), NoRedirect())
    payload = json.dumps(body).encode() if body is not None else None
    req = Request("http://127.0.0.1:11434" + endpoint, data=payload,
                  headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    value = json.loads(raw)
    if value.get("error"):
        raise RuntimeError(value["error"])
    return value, raw


def parse_goal(raw):
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {"goal"}
            or not isinstance(value["goal"], str) or value["goal"] not in list(LABELS)):
        raise ValueError('Expected exactly {"goal":"A"} with a single letter A-F.')
    return LABELS.index(value["goal"])


def relabel(scenario, robot_order, goal_order):
    """Orders map displayed indices to original physical indices."""
    result = copy.deepcopy(scenario)
    goal_inverse = {old: new for new, old in enumerate(goal_order)}
    result["robots"] = [scenario["robots"][i] for i in robot_order]
    result["goals"] = [scenario["goals"][i] for i in goal_order]
    result["costs"] = [[scenario["costs"][i][g] for g in goal_order] for i in robot_order]
    result["initial"] = [goal_inverse[scenario["initial"][i]] for i in robot_order]
    return result


def local_model(model):
    tags, _ = request("/api/tags")
    info = next((item for item in tags.get("models", []) if item["name"] == model), None)
    if info is None or "cloud" in model.lower():
        raise ValueError("Choose an already installed local model.")
    details, _ = request("/api/show", {"model": model})
    if any(item.get(key) for item in (info, details) for key in ("remote_host", "remote_model")):
        raise ValueError("Cloud-backed model refused.")
    version, _ = request("/api/version")
    return info, version


def make_plan(suite, model_info, version, seed=81042):
    check = dict(suite)
    expected = check.pop("sha256")
    if digest(check) != expected:
        raise ValueError("Scenario suite hash does not match its contents.")
    rng = random.Random(seed)
    trials = []
    for family_index, family in enumerate(suite["families"]):
        robots, goals = list(range(6)), list(range(6))
        rng.shuffle(robots)
        rng.shuffle(goals)
        for rotation in range(6):
            robot_order = robots[rotation:] + robots[:rotation]
            shift = 5 * rotation % 6
            goal_order = goals[shift:] + goals[:shift]
            for condition in family["conditions"]:
                scenario = relabel(condition["scenario"], robot_order, goal_order)
                trial_id = f"{family['id']}-m{scenario['m']}-r{rotation}"
                scenario["id"] = trial_id
                validate_scenario(scenario)
                prompts = [build_agent_prompt(scenario, i) for i in range(6)]
                trials.append({"id": trial_id, "familyId": family["id"], "m": scenario["m"],
                               "rotation": rotation, "robotOrder": robot_order, "goalOrder": goal_order,
                               "seed": seed + 100 * family_index + 10 * rotation,
                               "scenario": scenario, "prompts": prompts})
    rng.shuffle(trials)
    plan = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
            "suiteHash": expected, "modelInfo": model_info, "ollamaVersion": version,
            "promptVersion": PROMPT_VERSION, "planSeed": seed,
            "options": {"temperature": 0.7, "num_ctx": 8192, "num_predict": 256},
            "design": "12 matched families by default; three m conditions; six balanced label rotations. One stochastic joint decision per rotation. Family is the uncertainty unit.",
            "seedRule": "paired family/rotation seed across m, plus displayed robot index",
            "execution": "randomized trial order; six isolated sequential chats; no cross-agent history or retries",
            "primaryOutcome": "successful beneficial joint exchange, among all attempted planned trials",
            "secondaryOutcomes": ["unique-target choice accuracy for participating robots", "unique-target choice accuracy for nonparticipants", "invalid responses", "normalized realized saving"],
            "uncertainty": "paired family bootstrap (10000 resamples), descriptive for this selected scenario distribution; floor warning if all outcomes equal",
            "replaySelection": "first completed trial in randomized plan order, regardless of success",
            "trials": trials}
    plan["sha256"] = digest(plan)
    return plan


def verify_plan(plan):
    check = dict(plan)
    expected = check.pop("sha256")
    if digest(check) != expected:
        raise ValueError("Plan was modified after freezing.")


def run_one(trial, plan, path):
    pending = path.with_suffix(path.suffix + ".tmp")
    if pending.exists():
        recovered = json.loads(pending.read_text(encoding="utf-8"))
        if recovered.get("planHash") != plan["sha256"] or recovered.get("id") != trial["id"]:
            raise ValueError("Pending save does not match this trial.")
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            count = len(prior["agents"])
            if recovered["agents"][:count] != prior["agents"] or recovered["choices"][:count] != prior["choices"]:
                raise ValueError("Pending save conflicts with durable decisions.")
        save_json(path, recovered)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("planHash") != plan["sha256"]:
            raise ValueError("Record belongs to a different plan.")
        if record["status"] in ("completed", "error"):
            return record
    else:
        record = {"schemaVersion": 1, "source": "local-llm", "kind": "matched-study-trial",
                  "id": trial["id"], "familyId": trial["familyId"], "m": trial["m"],
                  "rotation": trial["rotation"], "planHash": plan["sha256"],
                  "scenario": trial["scenario"], "scenarioId": trial["scenario"]["id"],
                  "model": plan["modelInfo"]["name"], "modelInfo": plan["modelInfo"],
                  "promptVersion": plan["promptVersion"], "localOnly": True,
                  "startedAt": datetime.now(timezone.utc).isoformat(),
                  "agents": [], "choices": [], "status": "running", "result": None}
        save_json(path, record)
    for index in range(len(record["agents"]), 6):
        options = dict(plan["options"], seed=trial["seed"] + index)
        prompt = trial["prompts"][index]
        agent = {"agentId": LABELS[index], "prompt": prompt, "options": options,
                 "goal": None, "rawContent": None, "rawResponse": None, "error": None}
        if "think" in plan:
            agent["think"] = plan["think"]
        start = time.monotonic()
        try:
            payload = {"model": record["model"], "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "format": "json", "options": options, "keep_alive": "10m"}
            if "think" in plan:
                payload["think"] = plan["think"]
            value, raw = request("/api/chat", payload, timeout=plan.get("requestTimeoutSeconds", 120))
            agent["rawResponse"] = raw
            agent["rawContent"] = value.get("message", {}).get("content")
            if value.get("done") is not True or value.get("done_reason") == "length":
                raise ValueError("Incomplete model response.")
            choice = parse_goal(agent["rawContent"])
            agent["goal"] = LABELS[choice]
        except (ValueError, TypeError) as exc:
            choice = None
            agent["error"] = {"code": "INVALID_RESPONSE", "message": str(exc)}
        except Exception as exc:
            # Stop on infrastructure failure. Previous responses remain saved; resume the missing call.
            record["status"] = "interrupted"
            record.setdefault("infrastructureErrors", []).append({"agentIndex": index, "message": str(exc),
                                                                  "at": datetime.now(timezone.utc).isoformat()})
            save_json(path, record)
            raise
        agent["durationMs"] = round(1000 * (time.monotonic() - start))
        record["agents"].append(agent)
        record["choices"].append(choice)
        save_json(path, record)
    record["result"] = resolve_choices(trial["scenario"], record["choices"])
    record["status"] = "completed" if all(c is not None for c in record["choices"]) else "error"
    record["completedAt"] = datetime.now(timezone.utc).isoformat()
    record["durationMs"] = sum(agent["durationMs"] for agent in record["agents"])
    save_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description="Run or resume the local-model study.")
    parser.add_argument("--suite", type=Path, default=ROOT / "data/matched-suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/matched-study")
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    info, version = local_model(args.model)
    manifest = args.output / "plan.json"
    if manifest.exists():
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        verify_plan(plan)
        if plan["modelInfo"]["digest"] != info["digest"] or plan["modelInfo"]["name"] != info["name"]:
            raise ValueError("Installed model differs from frozen plan.")
        if plan["ollamaVersion"] != version:
            raise ValueError("Ollama version differs from frozen plan; do not mix versions.")
    else:
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        plan = make_plan(suite, info, version)
        save_json(manifest, plan)
    print(f"Frozen plan: {len(plan['trials'])} trials; SHA256 {plan['sha256']}", flush=True)
    if args.plan_only:
        return
    for index, trial in enumerate(plan["trials"]):
        record = run_one(trial, plan, args.output / "records" / (trial["id"] + ".json"))
        print(f"{index+1}/{len(plan['trials'])} {trial['id']}: {record['status']}, success={record['result']['success']}", flush=True)


if __name__ == "__main__":
    main()
