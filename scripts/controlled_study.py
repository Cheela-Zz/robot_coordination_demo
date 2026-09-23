from __future__ import annotations

import argparse
from collections import deque
from itertools import permutations, product
from pathlib import Path
import random
import sys
import time
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_demo.engine import validate_scenario
from robot_demo.storage import digest, save_json

PERMUTATIONS = list(permutations(range(6)))


def grid_costs(width, height, obstacles, robots, goals):
    blocked = set(map(tuple, obstacles))
    rows = []
    for robot in robots:
        queue = deque([tuple(robot)])
        distances = {tuple(robot): 0}
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                cell = x + dx, y + dy
                if (0 <= cell[0] < width and 0 <= cell[1] < height
                        and cell not in blocked and cell not in distances):
                    distances[cell] = distances[x, y] + 1
                    queue.append(cell)
        if any(tuple(goal) not in distances for goal in goals):
            return None
        rows.append([distances[tuple(goal)] for goal in goals])
    return rows


def unique_improvement(costs, initial):
    """Enumerate only permitted assignments, stopping after two improvements."""
    options = [[initial[i]] + [g for g in range(6) if costs[i][g] < costs[i][initial[i]]]
               for i in range(6)]
    order = sorted(range(6), key=lambda i: len(options[i]))
    solution = list(initial)
    improvements = []

    def visit(depth, used):
        if len(improvements) > 1:
            return
        if depth == 6:
            if solution != list(initial):
                improvements.append(list(solution))
            return
        robot = order[depth]
        for goal in options[robot]:
            if not used & (1 << goal):
                solution[robot] = goal
                visit(depth + 1, used | (1 << goal))

    visit(0, 0)
    if len(improvements) != 1:
        return None
    target = improvements[0]
    changed = [i for i in range(6) if target[i] != initial[i]]
    if len(changed) not in (2, 3, 4):
        return None
    gains = [costs[i][initial[i]] - costs[i][target[i]] for i in changed]
    return {"initial": list(initial), "target": target, "m": len(changed),
            "initialTotal": sum(costs[i][initial[i]] for i in range(6)),
            "initialCosts": [costs[i][initial[i]] for i in range(6)],
            "participantGains": gains, "meanGain": sum(gains) / len(gains),
            "improvingEdges": sum(len(row) - 1 for row in options)}


def generate_world(rng):
    width, height = 17, 13
    obstacles = []
    for x in range(width):
        for y in range(height):
            border = x in (0, width - 1) or y in (0, height - 1)
            shelf = x in (3, 4, 7, 8, 11, 12) and y in (3, 4, 8, 9)
            if border or shelf:
                obstacles.append([x, y])
    blocked = set(map(tuple, obstacles))
    open_cells = [[x, y] for x in range(1, width - 1) for y in range(1, height - 1)
                  if (x, y) not in blocked]
    points = rng.sample(open_cells, 12)
    world = {"width": width, "height": height, "obstacles": obstacles,
             "robots": points[:6], "goals": points[6:]}
    world["costs"] = grid_costs(width, height, obstacles, world["robots"], world["goals"])
    return world


def generate_suite(output, families=12, max_worlds=150000, seed=20260923):
    """Select one matched trio per geometry, without any model responses."""
    rng = random.Random(seed)
    selected = []
    started = time.monotonic()
    for world_index in range(max_worlds):
        world = generate_world(rng)
        groups = {}
        for initial in PERMUTATIONS:
            item = unique_improvement(world["costs"], initial)
            if item is not None:
                key = (item["initialTotal"], Fraction(sum(item["participantGains"]), item["m"]),
                       item["improvingEdges"])
                groups.setdefault(key, {}).setdefault(item["m"], []).append(item)
        matches = [trio for by_m in groups.values() if set(by_m) == {2, 3, 4}
                   for trio in product(by_m[2], by_m[3], by_m[4])]
        if matches:
            def mismatch(trio):
                gain_spread = sum(sum((g - item["meanGain"]) ** 2 for g in item["participantGains"])
                                  for item in trio)
                makespans = [max(item["initialCosts"]) for item in trio]
                return (gain_spread, max(makespans)-min(makespans),
                        tuple(tuple(item["initial"]) for item in trio))
            trio = min(matches, key=mismatch)
            family_id = f"family-{len(selected)+1:02}"
            scenarios = []
            for item in trio:
                scenario = dict(world, initial=item["initial"], m=item["m"],
                                id=f"{family_id}-m{item['m']}", name=f"{family_id}, m={item['m']}",
                                description="Matched six-robot exchange; unique beneficial reassignment.")
                analysis = validate_scenario(scenario)
                if len(analysis["beneficialAssignments"]) != 1 or analysis["oracleAssignment"] != item["target"]:
                    raise ValueError("Independent exhaustive validation disagrees with generator.")
                scenarios.append({"scenario": scenario, "audit": item})
            selected.append({"id": family_id, "worldIndex": world_index, "conditions": scenarios})
            print(f"Matched {len(selected)}/{families}: world {world_index}; per-agent gain residual {mismatch(trio)[0]:g}", flush=True)
        if len(selected) >= families:
            break
        if world_index and world_index % 2000 == 0:
            print(f"Searched {world_index} geometries; {len(selected)} matched families", flush=True)
    if len(selected) != families:
        raise ValueError(f"Found only {len(selected)} families; no incomplete suite was saved.")
    suite = {"schemaVersion": 1, "generatorSeed": seed, "worldsSearched": world_index+1,
             "selection": "First distinct geometries admitting exact total/mean-gain/cheaper-edge matching; prefer lower gain variance, then makespan spread, then lexicographic assignments.",
             "exactControls": ["six robots", "same geometry and full cost matrix within family", "initial total cost",
                               "mean gain per participating robot", "number of strictly cheaper row choices",
                               "exactly one beneficial reassignment"],
             "residualDifferences": ["individual gains", "individual initial costs", "initial makespan",
                                     "distribution of cheaper choices across robots", "total available saving equals m times mean gain"],
             "families": selected}
    suite["sha256"] = digest(suite)
    save_json(output, suite)
    print(f"Saved {output}; {len(selected)*3} validated scenarios in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate matched six-robot scenarios.")
    parser.add_argument("--generate", type=Path, required=True)
    parser.add_argument("--families", type=int, default=12)
    args = parser.parse_args()
    generate_suite(args.generate, args.families)
