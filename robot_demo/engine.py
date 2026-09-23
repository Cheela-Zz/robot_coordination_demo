"""Exact grid costs, exhaustive exchange verification, and execution rules."""

from collections import deque
from itertools import permutations

ROBOT_LABELS = "ABCDEF"

def _key(point):
    return tuple(point)

def grid_path(scenario, start, end):
    blocked = {_key(point) for point in scenario["obstacles"]}
    width, height = scenario["width"], scenario["height"]
    def valid(point):
        x, y = point
        return 0 <= x < width and 0 <= y < height and _key(point) not in blocked
    start, end = list(start), list(end)
    if not valid(start) or not valid(end):
        return []
    queue = deque([start])
    previous = {_key(start): None}
    while queue:
        current = queue.popleft()
        if current == end:
            path, cell = [], current
            while cell is not None:
                path.append(cell)
                cell = previous[_key(cell)]
            return path[::-1]
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nxt = [current[0] + dx, current[1] + dy]
            if valid(nxt) and _key(nxt) not in previous:
                previous[_key(nxt)] = current
                queue.append(nxt)
    return []

def shortest_path(scenario, robot_index, goal_index):
    return grid_path(scenario, scenario["robots"][robot_index], scenario["goals"][goal_index])

def compute_costs(scenario):
    return [[len(shortest_path(scenario, robot, goal)) - 1 for goal in range(6)] for robot in range(6)]

def assignment_costs(scenario, assignment=None):
    assignment = scenario["initial"] if assignment is None else assignment
    return [scenario["costs"][robot][goal] for robot, goal in enumerate(assignment)]

def all_permutations(values):
    return [list(value) for value in permutations(values)]

def validate_choices(scenario, choices):
    return isinstance(choices, list) and len(choices) == 6 and all(isinstance(goal, int) and 0 <= goal < 6 for goal in choices)

def analyze_scenario(scenario):
    initial_costs = assignment_costs(scenario)
    beneficial = []
    for assignment in all_permutations(scenario["initial"]):
        changed = [i for i, goal in enumerate(assignment) if goal != scenario["initial"][i]]
        if changed and all(scenario["costs"][i][assignment[i]] < initial_costs[i] for i in changed):
            beneficial.append(assignment)
    ordered = sorted(beneficial, key=lambda assignment: (sum(assignment_costs(scenario, assignment)), assignment))
    oracle = ordered[0] if ordered else list(scenario["initial"])
    minimum = min((sum(goal != scenario["initial"][i] for i, goal in enumerate(a)) for a in beneficial), default=None)
    return {"m": minimum, "beneficialAssignments": beneficial, "oracleAssignment": oracle,
            "initialTotal": sum(initial_costs), "oracleTotal": sum(assignment_costs(scenario, oracle)),
            "permutationCount": 720}

def validate_scenario(scenario):
    if not isinstance(scenario, dict) or len(scenario.get("robots", [])) != 6 or len(scenario.get("goals", [])) != 6:
        raise ValueError("This experiment requires exactly six robots and six goals.")
    width, height = scenario["width"], scenario["height"]
    obstacles = {_key(point) for point in scenario["obstacles"]}
    for field in ("obstacles", "robots", "goals"):
        points = scenario.get(field)
        if not isinstance(points, list) or len({_key(p) for p in points}) != len(points):
            raise ValueError(f"Invalid or duplicate {field} coordinates.")
        if any(len(p) != 2 or not all(isinstance(v, int) for v in p) or not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
            raise ValueError(f"Invalid {field} coordinate.")
    if any(_key(point) in obstacles for point in scenario["robots"] + scenario["goals"]):
        raise ValueError("Robots and goals must be walkable.")
    if sorted(scenario["initial"]) != list(range(6)):
        raise ValueError("Initial assignment must be a bijection over six goals.")
    if len(scenario["costs"]) != 6 or any(len(row) != 6 for row in scenario["costs"]):
        raise ValueError("Costs must be a six-by-six matrix.")
    computed = compute_costs(scenario)
    if computed != scenario["costs"]:
        raise ValueError("Stored costs do not match exact shortest-path distances.")
    analysis = analyze_scenario(scenario)
    if analysis["m"] != scenario.get("m"):
        raise ValueError(f"Stored m={scenario.get('m')} differs from exhaustive m={analysis['m']}.")
    return analysis

def resolve_choices(scenario, choices):
    requested = list(choices) if isinstance(choices, list) else []
    before = assignment_costs(scenario)
    valid = validate_choices(scenario, requested)
    participants = [i for i, goal in enumerate(requested) if valid and goal != scenario["initial"][i]]
    if not valid:
        reason = "Every robot must choose one valid goal."
        success = False
    elif len(set(requested)) != 6:
        reason = "Conflicting choices: more than one robot chose the same goal."
        success = False
    elif not participants:
        reason = "All robots kept their initial goals; no exchange occurred."
        success = False
    elif any(scenario["costs"][i][requested[i]] >= before[i] for i in participants):
        reason = "At least one participating robot would not strictly benefit. The exchange was cancelled."
        success = False
    else:
        reason = f"All {len(participants)} participating robots strictly benefit."
        success = True
    assignment = requested if success else list(scenario["initial"])
    after = assignment_costs(scenario, assignment)
    return {"assignment": assignment, "requested": requested, "success": success, "reason": reason,
            "participants": participants, "gains": [a - b for a, b in zip(before, after)],
            "beforeTotal": sum(before), "afterTotal": sum(after),
            "beforeMakespan": max(before), "afterMakespan": max(after)}

def greedy_choices(scenario):
    return [min(range(6), key=lambda goal: (row[goal], goal)) for row in scenario["costs"]]
