# Robot goal coordination

Six independent LLM agents choose goals from a shared travel-cost table. An exchange executes only if the goals are distinct and every robot that changes goals strictly reduces its cost. Otherwise, the initial assignment stays in place.

We compare minimum exchange sizes **m = 2, 3, 4** with six robots fixed: 12 matched position layouts × 3 conditions × 6 label rotations = **216 trials** using local `gemma2:2b`.

**Result:** 0/72 successful exchanges at each m; 215 joint decisions had duplicate goals and one kept all initial goals. These results do not establish an exchange-size effect.

[Comparison figure](outputs/matched-study/comparison.png) · [Trial data](outputs/matched-study/trials.csv) · [Recorded replay](outputs/matched-study/real-replay/replay.gif)

## Run

Python 3.10+. From this directory:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/analyze_matched_study.py --plot
python scripts/render_demo.py --input outputs/matched-study/real-trial.json --output-dir outputs/matched-study/real-replay
```

These commands verify and render the saved results without calling a model. Model execution requires Ollama running locally with `gemma2:2b` installed:

```powershell
python scripts/run_matched_study.py
```

The runner resumes the frozen plan and skips completed trials. Raw prompts, replies, seeds, model digest and outcomes are saved in `outputs/matched-study/records/`; the plan is `outputs/matched-study/plan.json`.

## Code

| File | Responsibility |
|---|---|
| `robot_demo/engine.py` | Grid distances, exhaustive assignment verification, exchange execution |
| `robot_demo/prompts.py` | Exact cost-table prompt supplied to each robot |
| `robot_demo/storage.py` | Atomic JSON writes and content hashes |
| `scripts/controlled_study.py` | Generate and match scenarios → `data/matched-suite.json` |
| `scripts/run_matched_study.py` | Label rotations, frozen plan, isolated Ollama calls and resume |
| `scripts/analyze_matched_study.py` | Audit raw responses, score trials, export CSV/JSON and comparison plots |
| `scripts/export_demo.py` | Convert a recorded decision to replay paths |
| `scripts/render_demo.py` | Draw the recorded trial as PNG/GIF |
| `tests/test_matched_study.py` | Matching, relabeling, parsing, scoring and record recovery |

To regenerate the scenario suite: `python scripts/controlled_study.py --generate data/matched-suite.json`.

## Design

All layouts use the same obstacle grid. Within each matched family, the cost matrix, initial total cost, mean participant gain, cheaper-choice count and number of beneficial reassignments are equal. Each scenario has exactly one beneficial reassignment, verified over all 720 permutations. Individual costs/gains and makespans can differ; see `outputs/matched-study/matching-audit.csv`.

Agents see costs, not map images or solver answers. Calls have separate contexts and no communication. Uncertainty uses a paired bootstrap over the 12 families, not 216 independent trials. Model choices are recorded; animated paths are deterministic and ignore robot–robot collisions. This extension has no Nash comparison or model training.
