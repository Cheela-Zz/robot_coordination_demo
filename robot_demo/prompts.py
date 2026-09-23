from .engine import ROBOT_LABELS


def build_agent_prompt(scenario, agent_index):
    labels = list(ROBOT_LABELS)
    rows = "\n".join(f"{labels[i]}: {', '.join(map(str, row))}" for i, row in enumerate(scenario["costs"]))
    initial = ", ".join(f"{labels[i]}={labels[goal]}" for i, goal in enumerate(scenario["initial"]))
    return f"""You are robot {labels[agent_index]}. Select one goal for yourself in a single simultaneous goal-assignment decision.
All six robots use the same local language model. Each receives this complete cost table and its own robot identity. Decisions are independent: you cannot communicate, see other responses, or revise your response.

Goal columns in order: {', '.join(labels)}
Travel costs in grid steps (smaller is better):
{rows}
Initial goal assignment: {initial}

Rules: every robot must select exactly one goal. Each goal must be selected by exactly one robot. If any goal is duplicated, any choice is invalid, or any robot's cost increases, the entire proposed reassignment is rejected and all robots keep their initial goals. A valid reassignment executes together. A robot that changes goals must strictly reduce its own cost; a robot that keeps its goal has unchanged cost. At least one robot must improve for a successful exchange. Your task is to select a goal that can help form such a mutually beneficial exchange while minimizing your travel cost. Staying with your initial goal is allowed. Robot paths visualize assignment costs only; collisions and travel timing are not part of this game.

Return exactly one JSON object containing only your selected goal, for example {{"goal":"A"}}. Allowed goal values: {', '.join(labels)}. Do not choose for other robots and do not include explanatory text."""
