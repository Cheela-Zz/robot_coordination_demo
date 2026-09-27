import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_demo.engine import resolve_choices, validate_scenario
from robot_demo.prompts import build_centralized_prompt
from robot_demo.storage import digest, save_json
from run_matched_study import local_model, request, verify_plan, run_one as run_independent_trial
from analyze_matched_study import score_trial, write_csv, bootstrap_family_means

LABELS = list("ABCDEF")


def same_model_info(saved, current):
    return ({k: v for k, v in saved.items() if k != "modified_at"}
            == {k: v for k, v in current.items() if k != "modified_at"})


def parse_assignment(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"goals"}:
        raise ValueError('Expected {"goals": [six letters in robot order]}.')
    goals = value["goals"]
    if not isinstance(goals, list) or len(goals) != 6 or any(type(g) is not str or g not in LABELS for g in goals):
        raise ValueError("Expected six goal letters A-F.")
    return [LABELS.index(g) for g in goals]


def make_check_plan(reference, info, version, thinking=False, max_tokens=None, thinking_profile="qwen", disable_thinking=False, reasoning_effort=None, temperature=None):
    verify_plan(reference)
    trials = [{"id": t["id"], "familyId": t["familyId"], "m": t["m"],
               "rotation": t["rotation"], "seed": t["seed"], "scenario": t["scenario"],
               "prompt": build_centralized_prompt(t["scenario"])}
              for t in reference["trials"] if t["rotation"] == 0]
    plan = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
            "mode": "centralized", "promptVersion": "whole-assignment-v1",
            "referencePlanHash": reference["sha256"], "modelInfo": info, "ollamaVersion": version,
            "options": reference["options"], "selection": "All 12 families and all m; rotation zero only, selected before inference.",
            "budget": "One call with 256 output tokens per centralized trial; independent trials use six calls with 256 each.",
            "trials": trials}
    if disable_thinking:
        plan["think"] = False
        plan["profile"] = "qwen-direct-256-v1"
        plan["options"] = dict(reference["options"], temperature=0.7, top_p=0.8, top_k=20,
                               min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0)
        if thinking_profile == "gemma":
            plan["profile"] = "gemma-direct-256-v1"
            plan["options"] = dict(reference["options"], temperature=1.0, top_p=0.95, top_k=64)
    if thinking:
        plan["think"] = True
        plan["profile"] = "qwen-thinking-4096-v1"
        plan["options"] = dict(reference["options"], num_predict=4096, temperature=1.0,
                               top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0)
        plan["requestTimeoutSeconds"] = 600
        plan["budget"] = "One call with up to 4096 output tokens including reasoning; different model, sampling settings and budget from Gemma."
        if thinking_profile == "gemma":
            plan["profile"] = "gemma-thinking-4096-v1"
            plan["options"] = dict(reference["options"], num_predict=4096, temperature=1.0, top_p=0.95, top_k=64)
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("Token budget must be positive.")
        plan["options"] = dict(plan["options"], num_predict=max_tokens, num_ctx=max(8192, max_tokens + 2048))
        plan["profile"] = f"{thinking_profile}-thinking-{max_tokens}-v1" if thinking else f"direct-{max_tokens}-v1"
        plan["budget"] = f"One call with up to {max_tokens} output tokens, including any reasoning."
    if thinking_profile == "deepseek":
        if not thinking:
            raise ValueError("The DeepSeek-R1 profile requires --thinking.")
        tokens = max_tokens or 8192
        plan["profile"] = f"deepseek-thinking-{tokens}-v1"
        plan["options"] = dict(reference["options"], num_predict=tokens, num_ctx=max(8192, tokens + 2048),
                               temperature=0.6, top_p=0.95, top_k=0, min_p=0.0, repeat_penalty=1.0)
        plan["requestTimeoutSeconds"] = 1200
        plan["budget"] = f"One call with up to {tokens} tokens including reasoning; capped local evaluation, not the published 32768-token benchmark."
    if reasoning_effort:
        tokens = max_tokens or 8192
        plan["think"] = reasoning_effort
        plan["profile"] = f"gpt-oss-{reasoning_effort}-{tokens}-v1"
        plan["options"] = dict(reference["options"], num_predict=tokens, num_ctx=max(8192, tokens + 2048),
                               temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repeat_penalty=1.0)
        plan["requestTimeoutSeconds"] = 1200
        plan["budget"] = f"One call with up to {tokens} tokens including {reasoning_effort} reasoning."
    if temperature is not None:
        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        plan["options"] = dict(plan["options"], temperature=temperature)
        plan["profile"] = plan.get("profile", "default") + f"-temp{temperature:g}"
    plan["sha256"] = digest(plan)
    return plan


def run_one(trial, plan, path):
    pending = path.with_suffix(".json.tmp")
    if pending.exists():
        saved = json.loads(pending.read_text(encoding="utf-8"))
        if saved.get("planHash") != plan["sha256"] or saved.get("id") != trial["id"]:
            raise ValueError("Pending record belongs to another trial.")
        save_json(path, saved)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("planHash") != plan["sha256"]:
            raise ValueError("Record belongs to another plan.")
        if saved["status"] in ("completed", "error"):
            return saved
    else:
        saved = {"id": trial["id"], "source": "local-llm", "mode": "centralized",
                 "planHash": plan["sha256"], "model": plan["modelInfo"]["name"],
                 "scenario": trial["scenario"], "prompt": trial["prompt"],
                 "options": dict(plan["options"], seed=trial["seed"]),
                 "think": plan.get("think"),
                 "startedAt": datetime.now(timezone.utc).isoformat(), "status": "running",
                 "rawContent": None, "rawResponse": None, "choices": None, "error": None}
        save_json(path, saved)
    started = time.monotonic()
    try:
        payload = {"model": saved["model"], "stream": False, "format": "json",
            "messages": [{"role": "user", "content": saved["prompt"]}],
            "options": saved["options"], "keep_alive": "10m"}
        if plan.get("think") is not None:
            payload["think"] = plan["think"]
        value, raw = request("/api/chat", payload, timeout=plan.get("requestTimeoutSeconds", 120))
    except Exception as exc:
        saved["status"] = "interrupted"
        saved.setdefault("infrastructureErrors", []).append(str(exc))
        save_json(path, saved)
        raise
    saved["rawResponse"] = raw
    saved["rawContent"] = value.get("message", {}).get("content")
    saved["rawThinking"] = value.get("message", {}).get("thinking")
    try:
        if value.get("done") is not True or value.get("done_reason") == "length":
            raise ValueError("Incomplete response.")
        saved["choices"] = parse_assignment(saved["rawContent"])
        saved["status"] = "completed"
    except (ValueError, TypeError) as exc:
        saved["choices"] = [None] * 6
        saved["status"] = "error"
        saved["error"] = str(exc)
    saved["result"] = resolve_choices(trial["scenario"], saved["choices"])
    saved["durationMs"] = round(1000 * (time.monotonic() - started))
    saved["completedAt"] = datetime.now(timezone.utc).isoformat()
    save_json(path, saved)
    return saved


def score_centralized(trial, record, plan):
    if (record["planHash"] != plan["sha256"] or record["scenario"] != trial["scenario"]
            or record["prompt"] != trial["prompt"] or record["model"] != plan["modelInfo"]["name"]
            or record.get("think") != plan.get("think")
            or record["options"] != dict(plan["options"], seed=trial["seed"])):
        raise ValueError("Record differs from centralized plan.")
    if record["status"] not in ("completed", "error"):
        raise ValueError("Incomplete centralized trial.")
    raw = json.loads(record["rawResponse"])
    if raw.get("message", {}).get("content") != record["rawContent"]:
        raise ValueError("Raw content mismatch.")
    try:
        if raw.get("done") is not True or raw.get("done_reason") == "length":
            raise ValueError("Incomplete response.")
        choices = parse_assignment(record["rawContent"])
        invalid = False
    except (ValueError, TypeError):
        choices = [None] * 6
        invalid = True
    result = resolve_choices(trial["scenario"], choices)
    if record["choices"] != choices or record["result"] != result or bool(record["error"]) != invalid:
        raise ValueError("Saved outcome disagrees with raw response.")
    target = validate_scenario(trial["scenario"])["oracleAssignment"]
    participants = [i for i in range(6) if target[i] != trial["scenario"]["initial"][i]]
    return {"trial": trial["id"], "family": trial["familyId"], "m": trial["m"],
            "success": int(result["success"]), "invalidResponses": int(invalid),
            "truncatedResponses": int(raw.get("done_reason") == "length"),
            "duplicateGoals": int(not invalid and len(set(choices)) < 6),
            "participantAccuracy": sum(choices[i] == target[i] for i in participants) / len(participants),
            "allAgentAccuracy": sum(a == b for a, b in zip(choices, target)) / 6,
            "selfCostCompliance": sum(g is not None and (g == trial["scenario"]["initial"][i]
                or trial["scenario"]["costs"][i][g] < trial["scenario"]["costs"][i][trial["scenario"]["initial"][i]])
                for i, g in enumerate(choices)) / 6,
            "choices": " ".join("?" if c is None else LABELS[c] for c in choices),
            "target": " ".join(LABELS[c] for c in target), "reason": result["reason"]}


def read_check(directory):
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    verify_plan(plan)
    rows = [score_centralized(t, json.loads((directory / "records" / (t["id"] + ".json")).read_text(encoding="utf-8")), plan)
            for t in plan["trials"]]
    return plan, rows


def make_independent_plan(central, reference):
    verify_plan(central)
    verify_plan(reference)
    if central["referencePlanHash"] != reference["sha256"]:
        raise ValueError("Centralized and independent reference differ.")
    ids = {t["id"] for t in central["trials"]}
    trials = [t for t in reference["trials"] if t["id"] in ids]
    if len(trials) != len(ids) or any(a["scenario"] != b["scenario"] for a, b in zip(trials, central["trials"])):
        raise ValueError("Centralized cases differ from reference.")
    plan = dict(reference, createdAt=datetime.now(timezone.utc).isoformat(), trials=trials,
                modelInfo=central["modelInfo"], ollamaVersion=central["ollamaVersion"], options=central["options"],
                mode="independent", centralizedPlanHash=central["sha256"],
                design="Same 36 rotation-zero cases as centralized check; six isolated calls per trial.",
                uncertainty="12 matched families; one rotation each. Exploratory model/profile selection.")
    plan.pop("sha256")
    for key in ("think", "profile", "requestTimeoutSeconds"):
        if key in central:
            plan[key] = central[key]
    plan["sha256"] = digest(plan)
    return plan


def read_independent_check(directory):
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    verify_plan(plan)
    rows = [score_trial(t, json.loads((directory / "records" / (t["id"] + ".json")).read_text(encoding="utf-8")), plan)
            for t in plan["trials"]]
    return plan, rows


def run_independent_check(args):
    central = json.loads((args.centralized / "plan.json").read_text(encoding="utf-8"))
    verify_plan(central)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    info, version = local_model(central["modelInfo"]["name"])
    if info["digest"] != central["modelInfo"]["digest"] or version != central["ollamaVersion"]:
        raise ValueError("Installed model or Ollama version changed.")
    expected = make_independent_plan(central, reference)
    path = args.output / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        verify_plan(plan)
        if any(plan[k] != expected[k] for k in expected if k not in ("createdAt", "sha256")):
            raise ValueError("Independent plan changed; use another output folder.")
    else:
        plan = expected
        save_json(path, plan)
    for index, trial in enumerate(plan["trials"]):
        if args.stop_after is not None and index >= args.stop_after:
            print("Stopped at requested prefix; the remaining independent cases are unrun.", flush=True)
            return
        record = run_independent_trial(trial, plan, args.output / "records" / (trial["id"] + ".json"))
        print(f"{index+1}/{len(plan['trials'])} {trial['id']}: {record['status']}, success={record['result']['success']}", flush=True)
    _, rows = read_independent_check(args.output)
    save_json(args.output / "results.json", {"model": info["name"], "mode": "independent", "conditions": summarize(rows)})
    write_csv(args.output / "trials.csv", rows)


def summarize(rows):
    return {str(m): {"trials": len(group), "successes": sum(r["success"] for r in group),
                     "invalidResponses": sum(r["invalidResponses"] for r in group),
                     "truncatedResponses": sum(r.get("truncatedResponses", 0) for r in group),
                     "duplicateGoals": sum(r["duplicateGoals"] for r in group),
                     "selfCostCompliance": sum(r["selfCostCompliance"] for r in group) / len(group),
                     "participantAccuracy": sum(r["participantAccuracy"] for r in group) / len(group)}
            for m in (2, 3, 4) if (group := [r for r in rows if r["m"] == m])}


def failure_counts(model, mode, sample):
    counts = {"success": 0, "duplicate goals": 0, "non-beneficial move": 0, "no exchange": 0, "token limit": 0, "invalid response": 0}
    for row in sample:
        category = ("success" if row["success"] else "token limit" if row.get("truncatedResponses") else "invalid response" if row["invalidResponses"]
                    else "duplicate goals" if row["duplicateGoals"] else "no exchange"
                    if row["reason"].startswith("All robots kept") else "non-beneficial move")
        counts[category] += 1
    return dict(model=model, mode=mode, trials=len(sample), **counts)


def run_check(args):
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    info, version = local_model(args.model)
    manifest = args.output / "plan.json"
    if args.reasoning_effort:
        details, _ = request("/api/show", {"model": args.model})
        levels = details.get("thinking", {}).get("values", [])
        if levels and args.reasoning_effort not in levels:
            raise ValueError("Model does not support the requested reasoning level.")
    expected = make_check_plan(reference, info, version, args.thinking, args.max_tokens, args.thinking_profile, args.no_thinking, args.reasoning_effort, args.temperature)
    if manifest.exists():
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        verify_plan(plan)
        if (not same_model_info(plan["modelInfo"], expected["modelInfo"])
                or any(plan[k] != expected[k] for k in expected if k not in ("createdAt", "sha256", "modelInfo"))):
            raise ValueError("Centralized plan differs; use a separate output folder.")
    else:
        plan = expected
        save_json(manifest, plan)
    print(f"Centralized plan frozen: {len(plan['trials'])} trials, {args.model}", flush=True)
    for index, trial in enumerate(plan["trials"]):
        if args.stop_after is not None and index >= args.stop_after:
            print("Stopped at requested prefix of the frozen plan; remaining cases have not been run.", flush=True)
            return
        record = run_one(trial, plan, args.output / "records" / (trial["id"] + ".json"))
        print(f"{index+1}/{len(plan['trials'])} {trial['id']}: {record['status']}, success={record['result']['success']}", flush=True)
    _, rows = read_check(args.output)
    save_json(args.output / "results.json", {"model": args.model, "mode": "centralized", "thinking": plan.get("think", False),
              "options": plan["options"], "conditions": summarize(rows)})
    write_csv(args.output / "trials.csv", rows)


def compare(args):
    previous = args.output / "results.json"
    keys = ("reasoning_check", "independent_check", "incomplete_check")
    if not any(getattr(args, key) for key in keys) and previous.exists():
        saved_inputs = json.loads(previous.read_text(encoding="utf-8")).get("inputs", {})
        for key in keys:
            setattr(args, key, [ROOT / path for path in saved_inputs.get(key, [])])
    rows, paired, all_data = [], {}, {}
    failures = []
    reference = None
    for size in ("2b", "9b"):
        independent_dir = ROOT / "outputs" / ("matched-study" if size == "2b" else "matched-study-9b")
        independent = json.loads((independent_dir / "plan.json").read_text(encoding="utf-8"))
        verify_plan(independent)
        if reference is None:
            reference = independent
        elif any(independent[k] != reference[k] for k in ("trials", "options", "suiteHash", "promptVersion", "planSeed")):
            raise ValueError("Independent model experiments are not matched.")
        independent_rows = [score_trial(t, json.loads((independent_dir / "records" / (t["id"] + ".json")).read_text(encoding="utf-8")), independent)
                            for t in independent["trials"]]
        central, central_rows = read_check(ROOT / "outputs" / ("centralized-" + size))
        if central["referencePlanHash"] != reference["sha256"] or central["modelInfo"]["digest"] != independent["modelInfo"]["digest"]:
            raise ValueError("Centralized check uses a different reference or model.")
        subset = [r for r in independent_rows if r["rotation"] == 0]
        if {r["trial"] for r in subset} != {r["trial"] for r in central_rows}:
            raise ValueError("Centralized and independent subsets differ.")
        all_data[size] = independent_rows
        for mode, sample in (("independent-r0", subset), ("centralized-r0", central_rows)):
            failures.append(failure_counts(independent["modelInfo"]["name"], mode, sample))
        for mode, sample in (("independent-all", independent_rows), ("independent-r0", subset), ("centralized-r0", central_rows)):
            for m, result in summarize(sample).items():
                rows.append(dict(model=independent["modelInfo"]["name"], mode=mode, m=int(m), **result))
        paired[size] = bootstrap_family_means([
            sum(r["success"] for r in central_rows if r["family"] == f) / 3
            - sum(r["success"] for r in subset if r["family"] == f) / 3
            for f in sorted({r["family"] for r in subset})])
    size_differences = {}
    for size, sample in all_data.items():
        size_differences[size] = bootstrap_family_means([
            sum(r["success"] for r in sample if r["family"] == f and r["m"] == 4) / 6
            - sum(r["success"] for r in sample if r["family"] == f and r["m"] == 2) / 6
            for f in sorted({r["family"] for r in sample})])
    result = {"rows": rows, "pairedFailureCounts": failures, "centralizedMinusIndependentR0": paired, "m4MinusM2": size_differences,
              "limits": "Exploratory checks after the initial null result. Centralized outputs and total inference budget differ from independent calls; this does not isolate coordination causally. One rotation per centralized scenario. Models differ in more than parameter count."}
    result["inputs"] = {key: [(path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path.resolve()).as_posix() for path in getattr(args, key) or []] for key in keys}
    result["reasoningChecks"] = []
    central_plans = {}
    central_samples = {}
    for directory in args.reasoning_check or []:
        reasoning, sample = read_check(directory)
        if reasoning["referencePlanHash"] != reference["sha256"] or {r["trial"] for r in sample} != {r["trial"] for r in central_rows}:
            raise ValueError("Reasoning check uses different cases.")
        model = reasoning["modelInfo"]["name"]
        central_plans[reasoning["sha256"]] = reasoning
        central_samples[reasoning["sha256"]] = sample
        conditions = summarize(sample)
        mode = f"centralized-{reasoning.get('profile', 'default')}-r0"
        result["reasoningChecks"].append({"model": model, "planHash": reasoning["sha256"], "think": reasoning.get("think"),
            "options": reasoning["options"], "conditions": conditions,
            "note": "Exploratory centralized condition; sampling settings and output budget differ from Gemma 2."})
        rows.extend(dict(model=model, mode=mode, m=int(m), **counts) for m, counts in conditions.items())
        failure = failure_counts(model, mode, sample)
        failure["label"] = f"{model} centralized"
        failures.append(failure)
    for directory in args.independent_check or []:
        plan, sample = read_independent_check(directory)
        central = central_plans.get(plan.get("centralizedPlanHash"))
        if central is None or plan["options"] != central["options"] or plan.get("think") != central.get("think"):
            raise ValueError("Independent check needs its matching centralized check.")
        expected = make_independent_plan(central, reference)
        if any(plan[k] != expected[k] for k in expected if k not in ("createdAt", "sha256")):
            raise ValueError("Independent check differs from expected paired plan.")
        model = plan["modelInfo"]["name"]
        rows.extend(dict(model=model, mode="independent-r0", m=int(m), **counts) for m, counts in summarize(sample).items())
        failure = failure_counts(model, "independent-r0", sample)
        failure["label"] = f"{model} independent"
        failures.append(failure)
        result.setdefault("extraIndependentChecks", []).append({"model": model, "planHash": plan["sha256"],
            "centralizedPlanHash": central["sha256"], "options": plan["options"], "think": plan.get("think"),
            "m4MinusM2": bootstrap_family_means([
                next(r["success"] for r in sample if r["family"] == f and r["m"] == 4)
                - next(r["success"] for r in sample if r["family"] == f and r["m"] == 2)
                for f in sorted({r["family"] for r in sample})]),
            "centralizedMinusIndependent": bootstrap_family_means([
                sum(r["success"] for r in central_samples[central["sha256"]] if r["family"] == f)/3
                - sum(r["success"] for r in sample if r["family"] == f)/3
                for f in sorted({r["family"] for r in sample})])})
    result["incompleteChecks"] = []
    for directory in args.incomplete_check or []:
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        verify_plan(plan)
        if plan["referencePlanHash"] != reference["sha256"]:
            raise ValueError("Incomplete check used a different reference.")
        sample = []
        for trial in plan["trials"]:
            path = directory / "records" / (trial["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] in ("completed", "error"):
                    sample.append(score_centralized(trial, record, plan))
        result["incompleteChecks"].append({"model": plan["modelInfo"]["name"], "planHash": plan["sha256"],
            "options": plan["options"], "planned": len(plan["trials"]), "finished": len(sample),
            "conditions": summarize(sample), "note": "Stopped early; unfinished cases are not scored. Excluded from matched comparison plot."})
    model_order = list(dict.fromkeys(r["model"] for r in rows))
    failures.sort(key=lambda row: (model_order.index(row["model"]), row["mode"].startswith("centralized")))
    save_json(args.output / "results.json", result)
    write_csv(args.output / "comparison.csv", rows)
    for row in rows:
        print(f"{row['model']:10} {row['mode']:16} m={row['m']}: {row['successes']}/{row['trials']}")
    if args.plot:
        plot_comparison(args.output, result)


def plot_comparison(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1.45]})
    models = list(dict.fromkeys(r["model"] for r in result["rows"] if r["mode"] == "independent-r0"))
    width = .8 / len(models)
    for offset, model in enumerate(models):
        color = plt.get_cmap("tab10")(offset % 10)
        sample = [r for r in result["rows"] if r["model"] == model and r["mode"] == "independent-r0"]
        x = [r["m"] + (offset-(len(models)-1)/2)*width for r in sample]
        y = [r["successes"]/r["trials"] for r in sample]
        left.bar(x, y, width=width*.9, label=model, color=color)
        left.scatter(x, y, color=color, s=24, zorder=3, clip_on=False)
        for position, value, row in zip(x, y, sample):
            left.text(position, value+.025, f"{row['successes']}/{row['trials']}", ha="center", color=color, fontsize=8, rotation=90 if len(models)>2 else 0)
    left.set(title="Independent decisions: exchange success", xlabel="Minimum exchange size m", xticks=[2, 3, 4], ylim=(0, 1.1))
    left.yaxis.set_major_formatter(PercentFormatter(1))
    left.legend(frameon=False, loc="upper right")
    counts = result["pairedFailureCounts"]
    bases = [0]*len(counts)
    colors = ["#14846b", "#b66338", "#8370ad", "#a3a9ae", "#cc9235", "#d04960"]
    for category, color in zip(("success", "duplicate goals", "non-beneficial move", "no exchange", "token limit", "invalid response"), colors):
        values = [row[category] for row in counts]
        right.barh(range(len(counts)), values, left=bases, color=color, label=category)
        for i, value in enumerate(values):
            if value:
                right.text(bases[i]+value/2, i, str(value), ha="center", va="center", color="white", fontsize=10)
        bases = [a+b for a, b in zip(bases, values)]
    labels = [row.get("label", "Gemma " + row["model"].split(":")[1].upper() + " " + row["mode"].replace("-r0", "")) for row in counts]
    right.set(title="Joint outcomes on the same 36 cases", yticks=range(len(counts)), yticklabels=labels,
              xlabel="Joint trials (one label rotation per scenario)", xlim=(0, 36))
    right.invert_yaxis()
    right.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Goal assignment: independent and centralized decisions", fontsize=17, fontweight="bold")
    caption = "Both panels: same 36 cases per condition (12 families x 3 exchange sizes), one label rotation per scenario.\nCentralized solving changes the output task and uses one call rather than six; this does not isolate coordination causally."
    if result["reasoningChecks"]:
        caption += "\nGemma 2: 256 tokens/call. Other model profiles have different token caps and sampling settings; see saved plans. Exploratory comparison."
    if result.get("incompleteChecks"):
        caption += "\nEarly-stopped reasoning checks are reported separately in results.json; unfinished cases are not scored as failures."
    fig.text(.04, .02, caption, fontsize=9, color="#4c5864")
    fig.subplots_adjust(left=.065, right=.98, top=.83, bottom=.32, wspace=.45)
    fig.savefig(output / "comparison.png", dpi=250)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Centralized assignment check and matched model comparison.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--reference", type=Path, default=ROOT / "outputs/matched-study/plan.json")
    thinking = run.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", action="store_true", help="Enable reasoning; default cap is 4096 tokens including reasoning.")
    thinking.add_argument("--no-thinking", action="store_true", help="Disable reasoning with the selected family's sampling settings.")
    thinking.add_argument("--reasoning-effort", choices=("low", "medium", "high"), help="Named gpt-oss reasoning level; default cap 8192.")
    run.add_argument("--thinking-profile", choices=("qwen", "gemma", "deepseek"), default="qwen", help="Model-family sampling settings; used with --thinking.")
    run.add_argument("--max-tokens", type=int, help="Override output budget in a separate frozen plan.")
    run.add_argument("--temperature", type=float, help="Override sampling temperature as a separate experimental profile.")
    run.add_argument("--stop-after", type=int, help="Run only this prefix; resume without this option to finish the same plan.")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--output", type=Path, default=ROOT / "outputs/model-comparison")
    comparison.add_argument("--plot", action="store_true")
    comparison.add_argument("--reasoning-check", type=Path, action="append", help="Additional complete centralized check; repeat for multiple conditions.")
    comparison.add_argument("--independent-check", type=Path, action="append", help="Paired independent check for an additional centralized condition.")
    comparison.add_argument("--incomplete-check", type=Path, action="append", help="Report a stopped centralized run separately; never score missing cases as failures.")
    paired = commands.add_parser("independent", help="Six isolated calls on the same cases and profile as a frozen centralized plan.")
    paired.add_argument("--centralized", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--stop-after", type=int, help="Run only this prefix of the frozen independent plan.")
    paired.add_argument("--reference", type=Path, default=ROOT / "outputs/matched-study/plan.json")
    args = parser.parse_args()
    if args.command == "run":
        run_check(args)
    elif args.command == "independent":
        run_independent_check(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
