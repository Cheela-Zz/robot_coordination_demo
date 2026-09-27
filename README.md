# Robot goal coordination

Six independent LLM agents choose goals from a shared travel-cost table. An exchange executes only if the goals are distinct and every robot that changes goals strictly reduces its cost. Otherwise, the initial assignment stays in place.

We compare minimum exchange sizes **m = 2, 3, 4** with six robots fixed: 12 matched position layouts × 3 conditions × 6 label rotations = **216 trials** using local `gemma2:2b`.

**Initial 2B result:** 0/72 successful exchanges at each m; 215 joint decisions had duplicate goals and one kept all initial goals. These results do not establish an exchange-size effect.

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
| `scripts/diagnose_coordination.py` | Paired centralized/independent checks with additional local models |
| `scripts/export_demo.py` | Convert a recorded decision to replay paths |
| `scripts/render_demo.py` | Draw the recorded trial as PNG/GIF |
| `tests/test_matched_study.py` | Matching, relabeling, parsing, scoring and record recovery |
| `tests/test_diagnostics.py` | Paired cases, thinking settings, output scoring and resume |

To regenerate the scenario suite: `python scripts/controlled_study.py --generate data/matched-suite.json`.

## Design

All layouts use the same obstacle grid. Within each matched family, the cost matrix, initial total cost, mean participant gain, cheaper-choice count and number of beneficial reassignments are equal. Each scenario has exactly one beneficial reassignment, verified over all 720 permutations. Individual costs/gains and makespans can differ; see `outputs/matched-study/matching-audit.csv`.

Agents see costs, not map images or solver answers. Calls have separate contexts and no communication. Uncertainty uses a paired bootstrap over the 12 families, not 216 independent trials. Model choices are recorded; animated paths are deterministic and ignore robot–robot collisions. This extension has no Nash comparison or model training.

## Additional local models

The comparison uses the same **36 cases** (12 families × 3 exchange sizes, rotation zero). Centralized: one call chooses all six goals. Independent: six isolated calls choose one goal each. Successful exchanges:

| Local model/profile | Independent | Centralized |
|---|---:|---:|
| Gemma 2 2B, 256 tokens/call | 0/36 | 0/36 |
| Gemma 2 9B, 256 tokens/call | 0/36 | 0/36 |
| Qwen3.5 9B, no thinking, 2,048 tokens/call | 0/36 | 1/36 |
| Gemma 4 12B, no thinking, 2,048 tokens/call | 0/36 | 1/36 |
| gpt-oss 20B, medium reasoning, 8,192 tokens/call | 3/36 | 27/36 |
| DeepSeek-R1-Distill-Qwen 14B, thinking, 8,192 tokens/call | 0/36 | 3/36 |

gpt-oss by exchange size: Independent (m=2: 3/12, m=3: 0/12, m=4: 0/12); 9/36 trials contained a truncated response. Centralized (m=2: 10/12, m=3: 7/12, m=4: 10/12); 1/36 trials contained a truncated response.

The independent m=4 minus m=2 success-rate difference was -25 percentage points (exploratory 95% matched-family bootstrap: -50 to 0). This is suggestive, not conclusive evidence of an exchange-size effect. Independent failures comprised 24 goal conflicts and 9 trials with truncated responses. The centralized/independent gap also changes the output task, individual objective and total call budget, so it does not isolate a pure coordination effect.

DeepSeek by exchange size: Independent (m=2: 0/12, m=3: 0/12, m=4: 0/12), 30/36 trials with truncated responses. centralized (m=2: 0/12, m=3: 2/12, m=4: 1/12), 26/36 trials with truncated responses.

Gemma 2 9B also completed all 216 rotations: 0/72 successes at every m. Extra models and profiles were chosen after earlier failures; this is exploratory, with different sampling settings and budgets. Centralized success does not establish independent coordination or an exchange-size effect.

DeepSeek uses the same 36 cases, prompts and seeds; thinking enabled, temperature 0.6, top-p 0.95, no top-k truncation or repetition penalty, and an 8,192-token cap including reasoning. It is the Qwen2.5-based R1 14B distillation, not the full R1 model. The [published evaluation](https://github.com/deepseek-ai/DeepSeek-R1#4-evaluation-results) used a larger token budget; these local results describe this capped configuration.

Incomplete centralized checks are retained: Qwen thinking at 4,096 tokens (13 finished attempts, all truncated; stopped early), Qwen at 16,384 (one truncated attempt), and Gemma 4 at 8,192 (one truncated attempt at temperature 1.0, one at 0.2). Each plan has 36 cases; unrun/interrupted cases are not counted as failed responses.

The first frozen gpt-oss case has real paired replays: [centralized success](outputs/centralized-gptoss/real-replay/replay.gif) and [independent failure](outputs/independent-gptoss/real-replay/replay.gif). Centralized travel fell from 43 to 35 steps; independent robots B and F both chose goal E, so the exchange was rejected. This example alone says nothing about an exchange-size trend. A separate [first successful independent replay](outputs/independent-gptoss/first-success/replay.gif) shows family-03-m2-r0 (travel 50 to 42); it is selected for success, not representative of the success rate.

[Comparison figure](outputs/model-comparison/comparison.png) · [Counts](outputs/model-comparison/comparison.csv) · [All conditions and settings](outputs/model-comparison/results.json).

Output folders contain `plan.json` (frozen prompts/settings), `records/` (raw replies and outcomes), and, when complete, `results.json` and `trials.csv`. `matched-study*` holds the full 216-trial runs; `centralized-*` and `independent-*` hold paired checks; `model-comparison` holds audited summaries.

Rebuild the last saved comparison without model calls:

```powershell
python scripts/diagnose_coordination.py compare --plot
```

Example local execution (Ollama and the model must already be installed). Completed records are skipped; use another output folder for changed settings:

```powershell
python scripts/diagnose_coordination.py run --model qwen3.5:9b --no-thinking --max-tokens 2048 --output outputs/centralized-qwen-direct
python scripts/diagnose_coordination.py independent --centralized outputs/centralized-qwen-direct --output outputs/independent-qwen-direct
```

Use `run --help` for Gemma sampling and named gpt-oss reasoning levels. `--stop-after` runs a prefix of the frozen plan; omit it to finish that same plan. Raw thinking is saved inside each model response; reaching a token cap is recorded as truncation, never repaired into an answer.
