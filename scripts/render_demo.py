from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1600, 1240
INK = "#132B46"
MUTED = "#52677D"
BG = "#F3F6FA"
BLUE = "#487DAD"
ORANGE = "#D66D2C"
ROBOT_COLORS = ["#2166AC", "#D16525", "#16847A", "#7653A7", "#B33C52", "#558338", "#99721A", "#586B7E"]


def font_file(bold: bool) -> str | None:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    return next((str(path) for path in candidates if path.exists()), None)


class Painter:
    def __init__(self, scale: float):
        self.scale = scale
        self.image = Image.new("RGB", (round(WIDTH * scale), round(HEIGHT * scale)), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}

    def box(self, values):
        return tuple(round(v * self.scale) for v in values)

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            path = font_file(bold)
            self.fonts[key] = ImageFont.truetype(path, round(size * self.scale)) if path else ImageFont.load_default(size=round(size * self.scale))
        return self.fonts[key]

    def text(self, xy, text, size=24, fill=INK, bold=False, anchor=None):
        self.draw.text(self.box(xy), str(text), font=self.font(size, bold), fill=fill, anchor=anchor)

    def fit_text(self, xy, text, max_width, size=24, fill=INK, bold=False):
        while size > 12 and self.draw.textlength(text, font=self.font(size, bold)) > max_width * self.scale:
            size -= 1
        self.text(xy, text, size, fill, bold)

    def rect(self, bounds, fill, outline=None, width=1, radius=0):
        if radius:
            self.draw.rounded_rectangle(self.box(bounds), radius=round(radius * self.scale), fill=fill, outline=outline, width=max(1, round(width * self.scale)))
        else:
            self.draw.rectangle(self.box(bounds), fill=fill, outline=outline, width=max(1, round(width * self.scale)))

    def line(self, points, fill, width=1):
        self.draw.line([self.box(point) for point in points], fill=fill, width=max(1, round(width * self.scale)), joint="curve")

    def circle(self, xy, radius, fill, outline=None, width=1):
        x, y = xy
        self.draw.ellipse(self.box((x - radius, y - radius, x + radius, y + radius)), fill=fill, outline=outline, width=max(1, round(width * self.scale)))


def goal_label(index: int) -> str:
    return chr(65 + index) if index < 26 else str(index + 1)


def check_paths(data):
    if data["run"]["source"] != "local-llm":
        raise ValueError("Replay requires a recorded local-model trial.")
    scenario = data["scenario"]
    n = len(scenario["robots"])
    if not 1 <= n <= len(ROBOT_COLORS):
        raise ValueError("The poster renderer supports 1–8 robots.")
    assignments = {"before": scenario["initial"], "after": data["run"]["result"]["assignment"]}
    obstacles = set(map(tuple, scenario["obstacles"]))
    for phase, assignment in assignments.items():
        paths = data["paths"][phase]
        if len(paths) != n or sorted(assignment) != list(range(n)):
            raise ValueError(f"Invalid {phase} assignment or paths.")
        for agent, path in enumerate(paths):
            if not path or path[0] != scenario["robots"][agent] or path[-1] != scenario["goals"][assignment[agent]]:
                raise ValueError(f"Incorrect endpoints: {phase}, robot {agent + 1}.")
            for point in path:
                if tuple(point) in obstacles or not (0 <= point[0] < scenario["width"] and 0 <= point[1] < scenario["height"]):
                    raise ValueError(f"Path leaves walkable grid: {phase}, robot {agent + 1}.")
            for a, b in zip(path, path[1:]):
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                    raise ValueError(f"Nonadjacent path step: {phase}, robot {agent + 1}.")
            if len(path) - 1 != scenario["costs"][agent][assignment[agent]]:
                raise ValueError(f"Path does not match assignment cost: {phase}, robot {agent + 1}.")


def mix(color, whiten=0.62):
    components = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return tuple(round(c * (1 - whiten) + 255 * whiten) for c in components)


def along_path(path, elapsed):
    if elapsed >= len(path) - 1:
        return path[-1]
    i = max(0, math.floor(elapsed))
    fraction = elapsed - i
    return [path[i][axis] * (1 - fraction) + path[i + 1][axis] * fraction for axis in (0, 1)]


def draw_grid(p, data, phase, x0, elapsed):
    s = data["scenario"]
    initial = phase == "before"
    assignment = s["initial"] if initial else data["run"]["result"]["assignment"]
    paths = data["paths"][phase]
    total = sum(len(path) - 1 for path in paths)
    panel = (x0, 271, x0 + 736, 914)
    p.rect(panel, "#FFFFFF", "#DCE4ED", radius=16)
    title = "01  ORIGINAL ASSIGNMENT" if initial else ("02  EXECUTED ASSIGNMENT" if data["run"]["result"]["success"] else "02  ORIGINAL GOALS RETAINED")
    p.fit_text((x0 + 24, 290), title, 688, 26, bold=True)
    show_request = not initial
    subtitle = f"Total travel: {total} grid steps"
    if show_request and not data["run"]["result"]["success"]:
        choices = data["run"]["choices"]
        if len(choices) != len(s["robots"]) or any(not isinstance(goal, int) or not 0 <= goal < len(s["goals"]) for goal in choices):
            subtitle = "Invalid or missing choice; exchange cancelled."
        elif len(set(choices)) != len(choices):
            subtitle = "Conflicting goal choices; exchange cancelled."
        elif choices == s["initial"]:
            subtitle = "No exchange requested; original goals retained."
        else:
            subtitle = "At least one robot would not benefit; exchange cancelled."
    p.fit_text((x0 + 24, 329 if show_request else 331), subtitle, 688, 20 if show_request else 22, MUTED)
    if show_request:
        requested = "   ".join(f"R{i + 1} → {goal_label(goal) if isinstance(goal, int) and 0 <= goal < len(s['goals']) else '?'}" for i, goal in enumerate(data["run"]["choices"]))
        p.fit_text((x0 + 24, 356), f"Requested:  {requested}", 688, 18, INK, True)
    map_top = 393
    map_height = 863 - map_top
    cell = min(678 / s["width"], map_height / s["height"])
    gx = x0 + (736 - cell * s["width"]) / 2
    gy = map_top + (map_height - cell * s["height"]) / 2

    def center(point):
        return gx + (point[0] + .5) * cell, gy + (point[1] + .5) * cell

    p.rect((gx, gy, gx + s["width"] * cell, gy + s["height"] * cell), "#F9FBFD", "#C9D4E0")
    for column in range(1, s["width"]):
        p.line([(gx + column * cell, gy), (gx + column * cell, gy + s["height"] * cell)], "#E2E8EF")
    for row in range(1, s["height"]):
        p.line([(gx, gy + row * cell), (gx + s["width"] * cell, gy + row * cell)], "#E2E8EF")
    for x, y in s["obstacles"]:
        p.rect((gx + x * cell + 1, gy + y * cell + 1, gx + (x + 1) * cell - 1, gy + (y + 1) * cell - 1), "#BCC8D4")
    for agent, path in enumerate(paths):
        points = [center(point) for point in path]
        if len(points) > 1:
            p.line(points, mix(ROBOT_COLORS[agent]), max(2, cell * .075))
        if elapsed is not None:
            completed = [center(point) for point in path[:min(len(path), math.floor(elapsed) + 1)]]
            completed.append(center(along_path(path, elapsed)))
            p.line(completed, ROBOT_COLORS[agent], max(2, cell * .055))
    radius = min(17, cell * .22)
    for index, goal in enumerate(s["goals"]):
        cx, cy = center(goal)
        cx -= cell * .20
        cy -= cell * .20
        p.rect((cx - radius, cy - radius, cx + radius, cy + radius), "#FFFFFF", INK, 1.5, radius=3)
        p.text((cx, cy), goal_label(index), max(11, radius * 1.04), INK, True, "mm")
    # Start marks remain visible once the animated robots have moved.
    for agent, start in enumerate(s["robots"]):
        cx, cy = center(start)
        p.circle((cx + cell * .15, cy + cell * .15), min(6, cell * .085), mix(ROBOT_COLORS[agent], .15), "#FFFFFF", 1)
    for agent, path in enumerate(paths):
        cx, cy = center(s["robots"][agent] if elapsed is None else along_path(path, elapsed))
        cx += cell * .15
        cy += cell * .15
        p.circle((cx, cy), radius, ROBOT_COLORS[agent], "#FFFFFF", 2)
        p.text((cx, cy), agent + 1, max(11, radius * 1.06), "#FFFFFF", True, "mm")
    labels = "    ".join(f"R{i + 1} → {goal_label(goal)}" for i, goal in enumerate(assignment))
    p.fit_text((x0 + 24, 875), labels, 688, size=20, fill=MUTED)


def draw_costs(p, data):
    s = data["scenario"]
    assignment = data["run"]["result"]["assignment"]
    n = len(assignment)
    p.text((48, 938), "EACH ROBOT'S TRAVEL COST", 25, bold=True)
    p.rect((1139, 945, 1160, 960), BLUE, radius=3)
    p.text((1168, 936), "Before", 21, MUTED)
    p.rect((1292, 945, 1313, 960), ORANGE, radius=3)
    p.text((1321, 936), "After", 21, MUTED)
    gap = 12
    width = (1504 - gap * (n - 1)) / n
    max_cost = max(max(row) for row in s["costs"])
    for agent, goal in enumerate(assignment):
        left = 48 + agent * (width + gap)
        before = s["costs"][agent][s["initial"][agent]]
        after = s["costs"][agent][goal]
        gain = before - after
        p.rect((left, 981, left + width, 1153), "#FFFFFF", "#DCE4ED", radius=10)
        p.circle((left + 23, 1007), 11, ROBOT_COLORS[agent])
        p.text((left + 23, 1007), agent + 1, 13, "#FFFFFF", True, "mm")
        p.fit_text((left + 42, 991), f"{goal_label(s['initial'][agent])} → {goal_label(goal)}", width - 52, 22, bold=True)
        chart_width = width - 62
        for value, y, color in [(before, 1040, BLUE), (after, 1075, ORANGE)]:
            p.rect((left + 15, y, left + 15 + chart_width, y + 17), "#E9EEF4", radius=3)
            if value:
                p.rect((left + 15, y, left + 15 + chart_width * value / max(1, max_cost), y + 17), color, radius=3)
            p.text((left + width - 13, y + 8), value, 20, INK, True, "rm")
        label = f"Saves {gain}" if gain > 0 else ("Unchanged" if gain == 0 else f"Loses {-gain}")
        p.fit_text((left + 15, 1115), label, width - 30, 21, "#226F54" if gain > 0 else MUTED, True)


def render(data, scale=1, elapsed=None, max_steps=None):
    p = Painter(scale)
    s = data["scenario"]
    run = data["run"]
    result = run["result"]
    source = run["source"]
    source_label = "LOCAL LLM TRIAL"
    p.rect((0, 0, WIDTH, 12), BLUE)
    p.text((48, 37), "WHEN ROBOTS MUST SWAP TOGETHER", 43, bold=True)
    p.text((48, 99), "Does coordination become harder when more agents must change together?", 27, MUTED)
    p.rect((48, 151, 1552, 250), INK, radius=14)
    before = sum(s["costs"][i][g] for i, g in enumerate(s["initial"]))
    after = sum(s["costs"][i][g] for i, g in enumerate(result["assignment"]))
    metrics = [
        (74, "MINIMUM EXCHANGE", f"{data['analysis']['m']} robots"),
        (451, "TRAVEL DISTANCE", f"{before} → {after} steps"),
        (875, "TOTAL SAVED", f"{before - after} steps"),
        (1210, "EXCHANGE", "Executed" if result["success"] else "Not executed"),
    ]
    for x, label, value in metrics:
        p.text((x, 167), label, 18, "#B6C8DA", True)
        p.fit_text((x, 196), value, 315, 29, "#FFFFFF", True)
    draw_grid(p, data, "before", 48, elapsed)
    draw_grid(p, data, "after", 816, elapsed)
    draw_costs(p, data)
    provenance = source_label
    if source == "local-llm" and run.get("model"):
        provenance += f"  ·  {run['model']}"
    if source == "local-llm" and run.get("trialId"):
        provenance += f"  ·  Trial: {run['trialId']}"
    p.fit_text((48, 1172), provenance, 1450, 20, INK, True)
    p.text((48, 1205), "Goal assignment only; paths ignore robot–robot collisions.", 18, MUTED)
    if elapsed is not None and max_steps is not None:
        p.text((1552, 1213), f"Travel time: {min(elapsed, max_steps):.1f} / {max_steps} steps", 17, MUTED, anchor="rm")
    return p.image


def main():
    parser = argparse.ArgumentParser(description="Render a recorded trial as PNG and GIF.")
    parser.add_argument("--input", required=True, type=Path, help="JSON exported by export_demo.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-gif", action="store_true", help="Render only the poster image")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    check_paths(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    poster_path = args.output_dir / "poster-example.png"
    render(data, scale=1.5).save(poster_path, dpi=(300, 300))
    print(f"Poster figure: {poster_path.resolve()}")
    if not args.skip_gif:
        max_steps = max(len(path) - 1 for paths in data["paths"].values() for path in paths)
        frame_count = min(85, max(24, max_steps * 3 + 1))
        frames = [render(data, scale=.8, elapsed=max_steps * i / (frame_count - 1), max_steps=max_steps) for i in range(frame_count)]
        # A single shared palette avoids per-frame color flicker.
        palette = frames[0].quantize(colors=200)
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        gif_path = args.output_dir / "replay.gif"
        indexed[0].save(gif_path, save_all=True, append_images=indexed[1:], duration=[1400] + [100] * (frame_count - 2) + [2000], loop=0, disposal=2, optimize=False)
        print(f"Animated replay: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
