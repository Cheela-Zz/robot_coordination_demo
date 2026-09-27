from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_demo.engine import validate_scenario, resolve_choices, greedy_choices
from robot_demo.storage import save_json
from run_matched_study import parse_goal, verify_plan

MS = (2, 3, 4)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_family_means(values, draws=10000, seed=94152):
    """Resample independent scenario families, never the correlated rotations."""
    point = statistics.mean(values)
    if len(set(values)) == 1:
        return {"mean": point, "low": None, "high": None,
                "note": "No between-family variation; percentile interval is uninformative."}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return {"mean": point, "low": samples[int(.025*draws)], "high": samples[int(.975*draws)],
            "note": "95% percentile bootstrap over matched families; exploratory."}


def score_trial(trial, record, plan):
    if record.get("planHash") != plan["sha256"] or record["scenario"] != trial["scenario"]:
        raise ValueError(f"Record does not match the frozen plan: {trial['id']}")
    if record["status"] not in ("completed", "error") or len(record["agents"]) != 6:
        raise ValueError(f"Incomplete trial: {trial['id']}. Resume the runner first.")
    choices = []
    truncated = 0
    for index, agent in enumerate(record["agents"]):
        if agent["prompt"] != trial["prompts"][index]:
            raise ValueError("Prompt mismatch.")
        if agent["options"] != dict(plan["options"], seed=trial["seed"]+index):
            raise ValueError("Sampling settings mismatch.")
        if agent.get("think") != plan.get("think"):
            raise ValueError("Thinking setting mismatch.")
        raw = json.loads(agent["rawResponse"])
        truncated += raw.get("done_reason") == "length"
        if raw.get("message", {}).get("content") != agent["rawContent"]:
            raise ValueError("Raw response content mismatch.")
        try:
            if raw.get("done") is not True or raw.get("done_reason") == "length":
                raise ValueError("Incomplete response")
            choice = parse_goal(agent["rawContent"])
        except (ValueError, TypeError):
            choice = None
        if (choice is None) != (agent["error"] is not None):
            raise ValueError("Error flag disagrees with raw response.")
        choices.append(choice)
    if choices != record["choices"]:
        raise ValueError("Saved choices disagree with raw model responses.")
    s = trial["scenario"]
    analysis = validate_scenario(s)
    target = analysis["oracleAssignment"]
    result = resolve_choices(s, choices)
    if result != record["result"]:
        raise ValueError("Saved outcome disagrees with independently scored choices.")
    participants = [i for i in range(6) if target[i] != s["initial"][i]]
    others = [i for i in range(6) if i not in participants]
    target_hits = sum(choices[i] == target[i] for i in range(6))
    correct_participants = sum(choices[i] == target[i] for i in participants)
    correct_others = sum(choices[i] == target[i] for i in others)
    valid = all(c is not None for c in choices)
    greedy = resolve_choices(s, greedy_choices(s))
    return {"trial": trial["id"], "family": trial["familyId"], "m": trial["m"],
            "rotation": trial["rotation"], "success": int(result["success"]),
            "participantAccuracy": correct_participants / len(participants),
            "nonparticipantAccuracy": correct_others / len(others),
            "allAgentAccuracy": target_hits / 6,
            "selfCostCompliance": sum(g is not None and (g == s["initial"][i] or s["costs"][i][g] < s["costs"][i][s["initial"][i]])
                                      for i, g in enumerate(choices)) / 6,
            "correctParticipants": correct_participants, "correctNonparticipants": correct_others,
            "invalidResponses": sum(c is None for c in choices),
            "truncatedResponses": truncated,
            "duplicateGoals": int(valid and len(set(choices)) < 6),
            "initialTotal": result["beforeTotal"], "afterTotal": result["afterTotal"],
            "possibleSaving": analysis["initialTotal"] - analysis["oracleTotal"],
            "realizedSavingFraction": (result["beforeTotal"]-result["afterTotal"]) / (analysis["initialTotal"]-analysis["oracleTotal"]),
            "beforeMakespan": result["beforeMakespan"], "afterMakespan": result["afterMakespan"],
            "greedySuccess": int(greedy["success"]), "oracleSuccess": 1,
            "choices": " ".join("?" if c is None else "ABCDEF"[c] for c in choices),
            "target": " ".join("ABCDEF"[c] for c in target), "reason": result["reason"]}


def analyze(directory, plot=False):
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    verify_plan(plan)
    trials, replay = [], None
    for trial in plan["trials"]:
        path = directory / "records" / (trial["id"] + ".json")
        if not path.exists():
            raise ValueError(f"Missing {trial['id']}; complete the frozen plan before final analysis.")
        record = json.loads(path.read_text(encoding="utf-8"))
        trials.append(score_trial(trial, record, plan))
        if replay is None and record["status"] == "completed":
            replay = record
    families = sorted({row["family"] for row in trials})
    family_rows = []
    metrics = ("success", "participantAccuracy", "nonparticipantAccuracy", "allAgentAccuracy",
               "realizedSavingFraction", "greedySuccess")
    for family in families:
        for m in MS:
            rows = [row for row in trials if row["family"] == family and row["m"] == m]
            if len(rows) != 6:
                raise ValueError("Each family-condition must have all six rotations.")
            family_rows.append(dict(family=family, m=m, **{metric: statistics.mean(r[metric] for r in rows) for metric in metrics}))
    summary = {"model": plan["modelInfo"]["name"], "modelDigest": plan["modelInfo"]["digest"],
               "planHash": plan["sha256"], "families": len(families), "trials": len(trials),
               "conditions": {}, "pairedDifferences": {}}
    for m in MS:
        rows = [row for row in trials if row["m"] == m]
        group = [row for row in family_rows if row["m"] == m]
        summary["conditions"][str(m)] = dict(trials=len(rows), successes=sum(r["success"] for r in rows),
            invalidResponses=sum(r["invalidResponses"] for r in rows), duplicates=sum(r["duplicateGoals"] for r in rows),
            **{metric: bootstrap_family_means([r[metric] for r in group]) for metric in metrics})
    for metric in metrics:
        delta = [next(r[metric] for r in family_rows if r["family"] == f and r["m"] == 4)
                 - next(r[metric] for r in family_rows if r["family"] == f and r["m"] == 2) for f in families]
        summary["pairedDifferences"][metric] = bootstrap_family_means(delta)
    summary["limitations"] = [
        "Exploratory local-model pilot; not a universal causal claim about coordination.",
        "Exact within-family controls: map/cost matrix, initial total, mean participant gain, cheaper-choice count, one beneficial assignment.",
        "Residual differences: per-robot costs/gains, makespan and cheaper-choice distribution. Total available saving rises with m.",
        "Six label rotations are correlated repeated measurements, not six independent scenario families.",
        "Label rotations and model seeds vary together; this pilot does not separately estimate stochastic and label effects.",
        "All agents see the full cost table; these results mix game reasoning, instruction-following and joint coordination.",
        "No collisions, physical robot dynamics, communication, learned policy training or Nash benchmark.",
        "Families were selected for matchability, not sampled from all possible warehouses."]
    save_json(directory / "results.json", summary)
    write_csv(directory / "trials.csv", trials)
    write_csv(directory / "families.csv", family_rows)
    audit = []
    for family in families:
        for m in MS:
            t = next(t for t in plan["trials"] if t["familyId"] == family and t["m"] == m and t["rotation"] == 0)
            s = t["scenario"]
            target = validate_scenario(s)["oracleAssignment"]
            initial_costs = [s["costs"][i][g] for i, g in enumerate(s["initial"])]
            gains = [initial_costs[i]-s["costs"][i][target[i]] for i in range(6) if target[i] != s["initial"][i]]
            degrees = [sum(cost < initial_costs[i] for cost in s["costs"][i]) for i in range(6)]
            audit.append({"family": family, "m": m, "initialTotal": sum(initial_costs),
                          "meanParticipantGain": statistics.mean(gains), "cheaperChoices": sum(degrees),
                          "beneficialAssignments": 1, "initialMakespan": max(initial_costs),
                          "participantGains": str(sorted(gains)), "initialCosts": str(sorted(initial_costs)),
                          "cheaperChoicesPerRobot": str(sorted(degrees))})
    write_csv(directory / "matching-audit.csv", audit)
    if replay:
        from export_demo import make_demo
        demo = make_demo(replay["scenario"], trace=replay)
        demo["limitations"] = ["One actual trial from the matched pilot; see results.json for the comparison.",
                                "Paths illustrate costs and ignore robot-robot collisions."]
        demo["run"]["planHash"] = plan["sha256"]
        save_json(directory / "real-trial.json", demo)
        summary["replayTrial"] = replay["id"]
        save_json(directory / "results.json", summary)
    if plot:
        plot_results(directory, summary, family_rows)
    for m in MS:
        condition = summary["conditions"][str(m)]
        print(f"m={m}: {condition['successes']}/{condition['trials']} successful exchanges")
    print(f"Verified {len(trials)*6} raw decisions; outputs: {directory}")



def plot_results(directory, summary, family_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4))
    specs = [("success", "Successful joint exchanges", "#286a98"),
             ("participantAccuracy", "Participating robots: target choice", "#168777"),
             ("nonparticipantAccuracy", "Other robots: retain initial goal", "#b26538")]
    for ax, (metric, title, color) in zip(axes, specs):
        for family in sorted({r["family"] for r in family_rows}):
            values = [next(r[metric] for r in family_rows if r["family"] == family and r["m"] == m) for m in MS]
            ax.plot(MS, values, color="#c8d0d6", lw=.8, alpha=.8, zorder=1)
        values = [summary["conditions"][str(m)][metric]["mean"] for m in MS]
        ax.plot(MS, values, "o-", color=color, linewidth=2.6, markersize=7, zorder=3)
        for m, value in zip(MS, values):
            ci = summary["conditions"][str(m)][metric]
            if ci["low"] is not None:
                ax.errorbar(m, value, yerr=[[value-ci["low"]], [ci["high"]-value]], color=color, capsize=5, zorder=4)
            label = f"{value:.1%}"
            if metric == "success":
                c = summary["conditions"][str(m)]
                label = f"{c['successes']}/{c['trials']}"
            ax.annotate(label, (m, value), xytext=(0, 12), textcoords="offset points", ha="center", color=color, fontweight="bold")
        ax.set(title=title, xlabel="Required exchange size m", xticks=MS, xlim=(1.7, 4.3), ylim=(-.045, 1.08))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"{summary['model']}: coordination across matched six-robot tasks", fontsize=17, fontweight="bold", y=.97)
    caption = f"{summary['families']} matched families × 3 conditions × 6 label rotations = {summary['trials']} real trials. Gray lines: family means.\nBars: exploratory 95% bootstrap intervals over families; omitted where between-family variation is zero."
    if all(summary["conditions"][str(m)]["successes"] == 0 for m in MS):
        caption += "\nNo successful exchanges at any m: this pilot cannot establish increasing coordination difficulty."
    fig.text(.035, .035, caption, fontsize=10, color="#455568", linespacing=1.5)
    fig.subplots_adjust(left=.055, right=.98, top=.82, bottom=.27, wspace=.33)
    fig.savefig(directory / "comparison.png", dpi=250)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Audit recorded decisions and rebuild results.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/matched-study")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    analyze(args.input, args.plot)


if __name__ == "__main__":
    main()
