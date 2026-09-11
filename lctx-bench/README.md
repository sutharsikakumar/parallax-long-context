# lctx-bench

A long-context evaluation harness that measures how model accuracy degrades as
**context length** and **information position** vary — with confidence intervals,
explicit control conditions, and pluggable backends for hosted APIs, local
weights, and custom attention implementations.

The design priority is methodological correctness over feature count. Most of
this README is about what the harness refuses to do silently.

```bash
pip install -e '.[dev,tiktoken]'
lctx smoke          # full pipeline offline in ~30s: no network, no API key, no GPU
```

`lctx smoke` runs a small grid against a simulator and writes heatmaps, curves,
and CSV summaries to `runs/smoke-*/report/`.

---

## What it measures

Each trial buries randomly generated facts (the *needles*) in neutral filler (the
*haystack*), asks a question that only those facts can answer, and scores the
answer deterministically. Sweeping context length and needle depth produces the
accuracy surface; the headline number is **effective context length** — the
longest context at which a model still clears an accuracy bar, with the bar
applied to the lower confidence bound rather than the point estimate.

---

## The six invariants

These are the load-bearing claims. Each is enforced by a test, not by convention.

**1. Length is decoupled from difficulty.** Growing the context adds filler and
nothing else. A task generator never sees `target_tokens` as an input to content:
the same seed produces byte-identical needles, questions, and answers at 1k
tokens and at 1M. Task parameters that *do* change difficulty (`num_items`,
`chain_length`) are set explicitly in config and held fixed across the sweep.
→ `test_generators.py::test_content_independent_of_length`

**2. Zero parametric leakage.** Every key, value, entity name, and identifier is
drawn fresh from the trial's RNG over an arbitrary symbol alphabet. Nothing is
answerable from pretraining, nothing is reused between trials, and the filler is
checked to never contain the answer.
→ `test_generators.py::test_ground_truth_absent_from_filler`, `::test_seeds_produce_distinct_content`

**3. Length is measured in tokens, with the model's own tokenizer, including
chat-template overhead.** The packer solves for the filler length that makes the
*fully rendered request* hit the target, measuring real messages rather than
summing parts (tokenization is not additive across concatenation boundaries).
Realized token counts, needle token positions, realized depths, and template
overhead are all recorded per trial, and `report/packing_audit.csv` is the
evidence. In practice every cell lands within 1 token of target.
→ `test_packing.py::test_token_exactness` (three tokenizers)

**4. Every cell reports an interval.** N independent seeds per cell; accuracy
gets a **Wilson** interval (the normal approximation misbehaves exactly where
long-context evaluation lives — small n, proportions near 0 and 1), continuous
scores get a deterministic **percentile bootstrap**.
→ `test_stats.py`

**5. Controls are part of the design, not an add-on.**

| Condition | What it controls for |
|---|---|
| `no_haystack` | The ceiling. Accuracy on the task with no filler. A degradation curve is uninterpretable until this is high — the report warns when it isn't. **On by default.** |
| `shuffled_haystack` | Whether degradation is driven by *coherent competing content* or by raw sequence length. |
| `needle_absent` | False positives. The evidence is removed and the correct answer becomes `ABSENT`. |
| `negative_retrieval` (task) | The same, as a standalone task with near-miss keys present. |

The abstention instruction appears in **every** prompt, in all conditions, so its
presence is never itself a cue.

**6. Deterministic and reproducible.** Greedy decoding by default. Config, seeds,
git commit, adapter identity, prompt hashes, raw outputs, token counts, latency,
and cost are all persisted. Prompts regenerate byte-identically from
`(task, seed, length, depth, condition)` alone.
→ `test_pipeline.py::test_prompts_are_reproducible_from_the_log`

---

## Tasks

Every task ships a generator, an **oracle** that answers from the structured data
alone, and a scorer. CI asserts the oracle scores 1.0 on every task × distractor
× condition × seed × depth combination — a generator that emits an unanswerable
or mis-keyed item fails the build.

| Task | Question | Scorer |
|---|---|---|
| `single_needle` | Value of one random key | exact match |
| `multi_needle` | Values of *k* random keys | set F1 |
| `multi_hop` | Follow a chain of assignments scattered across the document | exact match |
| `aggregation` | Sum / count / max over scattered numeric items | numeric tolerance |
| `selective_copy` | Every item matching a predicate | set F1 |
| `ordering` | Which entry immediately precedes a given entry | exact match |
| `negative_retrieval` | A key that is not present — correct answer is `ABSENT` | absent detection |

`multi_hop` places chain links in *shuffled* document order, so the chain cannot
be followed by reading linearly. `ordering` entries carry no index or timestamp,
so position is the only signal.

**Distractors** (`distractor_types` in config) probe different failure modes:
`near_duplicate_key` (keys one or two characters off — is retrieval exact or
approximate?), `semantic_lure` (the right key with a *qualified* attribute, plus
similar keys with the right attribute — is the relation bound, or just the key
spotted?), `repeated_decoy` (one wrong fact repeated throughout — does frequency
beat correctness?).

Run `lctx tasks` for the full list with parameters.

---

## Scoring

Deterministic scorers are the default; the LLM judge is a fallback for free-form
answers only. Answers are read from the `<answer>` tags the prompt requests, with
fallbacks for common phrasings, then normalized for presentation only — case,
whitespace, quoting, markdown, and punctuation inside identifiers are ignored;
a different identifier, number, or missing item is not.

The judge is deliberately hard to trust. Enabling it also scores a sample
deterministically and reports **raw agreement, Cohen's kappa** (raw agreement is
inflated when one label dominates, which it does here), false-positive and
false-negative rates, and the full list of disagreements to hand-label. Below
`judge.min_agreement` the report states the judge is unreliable and the
deterministic scores stand.

```bash
lctx judge runs/<run>      # writes report/judge_agreement.json
```

---

## Outputs

```
runs/<name>-<timestamp>/
  config.yaml          snapshot of the exact config used
  manifest.json        seeds, git commit, adapter identity, totals
  trials.jsonl         one record per trial — checkpointed as it completes
  prompts/*.json.gz    full prompts (store_prompts: full)
  report/
    heatmap__<model>__<task>__<condition>__<distractors>.png
    curves_by_task.png, curves_by_distractor__*.png, curves_by_condition__*.png
    summary_effective_context.csv     <- the headline table
    cells.csv, curves.csv, controls.csv
    packing_audit.csv                 <- token-exactness evidence
    failure_modes.csv                 <- abstained / copied a decoy / fabricated
    cost_latency.csv                  <- cost, p50/p95/max latency
    summary.json                      <- everything, machine-readable
```

**Effective context length** is reported two ways, because they answer different
questions. `effective_context_strict` is the longest length such that *every*
tested length up to it clears the bar — it refuses to credit a model that fails
at 32k and recovers at 64k. `effective_context_max` is the longest length that
clears the bar at all. When they differ, the curve is non-monotonic, and the
report says so rather than hiding it.

The report emits **integrity warnings** before any numbers: token-tolerance
violations, errored trials, a low no-haystack ceiling, and under-powered grids.
The last one matters more than it looks — a Wilson lower bound is strict, so a
*perfect* cell needs n ≥ 16 pooled trials before it can clear a 0.8 bar at all.
`lctx validate -c <config>` reports this before you spend anything:

```
power:  a perfect length (n=50 pooled over depths) has a 95% lower bound of 0.929
```

---

## Adding a model

Adapters implement one interface (`src/lctx/models/base.py`): `agenerate`,
a tokenizer with `count_message_tokens`, optional `capture_attention` /
`capture_logits`. Built in: `mock`, `echo`, `anthropic`, `openai`, `hf`,
`custom_attn`. Nothing about any model is hardcoded — adapter, tokenizer,
context limit, and pricing all come from config.

`openai` covers any OpenAI-compatible server via `base_url` — vLLM, SGLang,
llama.cpp, TGI — which is the fastest path for long-context sweeps against a
locally served model:

```yaml
- name: local-vllm
  adapter: openai
  model_id: my-org/my-model
  tokenizer: {kind: hf, name_or_path: my-org/my-model}
  params: {base_url: "http://localhost:8000/v1", require_key: false}
```

A third-party adapter needs no changes to lctx-bench — give a dotted path:

```yaml
- name: mine
  adapter: mypkg.adapters:build_my_adapter
```

---

## Evaluating a custom attention implementation

This is what `custom_attn` exists for: running a swapped attention mechanism
through the same grid, packing, scoring, and statistics as any hosted model. The
useful comparison is **the same weights loaded two ways**, so any difference in
the curves is attributable to the mechanism. `configs/local_custom_attention.yaml`
is set up exactly that way.

Start from `examples/custom_attention_stub.py` and run its self-check first — it
verifies output/weight shapes, row normalization, causality, and that the
mechanism actually changed the attention pattern, all without downloading a model:

```bash
python examples/custom_attention_stub.py
```

Three plug-in routes, in increasing order of invasiveness:

**Route 1 — a registered attention function** (preferred for kernel swaps). Publish
the kernel into `ALL_ATTENTION_FUNCTIONS` and select it by name. Module structure
is untouched.

```yaml
params:
  register_hook: mypkg.attn:register        # called before the model is built
  attn_implementation: my_attn
```

**Route 2 — a patch function**, when the mechanism changes module structure
(extra projections, learned state, a different cache layout):

```yaml
params:
  attention_patch: mypkg.attn:patch_model   # (model, **kwargs) -> patched model
  patch_kwargs: {window: 512}
```

**Route 3 — a complete loader**, when it is not an `AutoModelForCausalLM` at all:

```yaml
params:
  model_loader: mypkg.loading:load_my_model
  loader_kwargs: {checkpoint: /path/to/ckpt.pt}
```

`describe()` records which route was taken and whether the patch actually applied,
and it lands in `manifest.json` — a run that reports "custom attention" while
having silently measured the stock model is the failure mode this guards against.
`patch_model` in the stub raises rather than patching nothing, for the same reason.

---

## Probes (optional)

Probes answer "where did attention go?", which separates a *retrieval* failure
from a downstream one: a model with no attention mass on the needle never saw it.
Per layer, from the position that generates the answer: **sink mass** (mass on
token 0), **normalized entropy** (peaked vs. diffuse, comparable across lengths),
**needle mass**, and mean attention distance.

```bash
lctx run -c config.yaml --probes        # or: lctx probe runs/<run>
```

Probes require materialized attention weights, which fused kernels do not produce
— load with `attn_implementation: eager`. When weights are unavailable the harness
says so and returns nothing rather than fabricating a number. A custom mechanism
can expose its own by returning `attn_weights` (route 1) or implementing
`model.capture_attention(input_ids)`.

The needle-mass/recall correlation in `summary.json` is **descriptive only** —
probed cells are a deliberately spread, non-random sample of the grid.

---

## Adding a task

Subclass `Task`, implement `_generate` and `oracle`, and register it. The oracle
is not optional: it is what the acceptance test uses to prove the task is
solvable and correctly keyed.

```python
from lctx.tasks.base import Needle, Task, TaskInstance, TaskSpec, register_task
from lctx.tasks.primitives import random_code, unique

class MyTask(Task):
    name = "my_task"
    scorer = "exact_match"
    defaults = {"n_items": 5}          # unknown params in config are rejected

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        rng = spec.content_rng()       # never keyed on target_tokens or depth
        seen = set()
        key = unique(lambda: random_code(rng), seen)
        value = unique(lambda: random_code(rng), seen)
        return TaskInstance(
            spec=spec,
            needles=[Needle(f"The widget {key} has serial {value}.", spec.depth)],
            question=f"What is the serial of widget {key}?",
            system=self.build_system("Report the serial exactly as it appears."),
            ground_truth=value,
            scorer=self.scorer,
            structured={"value": value},
        )

    def oracle(self, inst: TaskInstance) -> str:
        return "ABSENT" if inst.structured.get("absent") else inst.structured["value"]

register_task(MyTask())
```

Import it in `src/lctx/tasks/__init__.py`, add it to `TASK_CASES` in
`tests/conftest.py` (a test fails if you forget), and the whole acceptance suite
— solvability, length independence, depth independence, leakage, packing — applies
to it automatically.

---

## Tests

```bash
pytest                      # ~355 tests, ~25s, fully offline
```

| Acceptance criterion | Test |
|---|---|
| Generators are solvable under their scorer | `test_generators.py::test_oracle_solves_every_task` |
| Scorer edge cases (whitespace, casing, absent, partial credit, tolerance) | `test_scorers.py` |
| Token exactness on ≥ 2 tokenizers | `test_packing.py::test_token_exactness` |
| Full pipeline + reporting with no network | `test_pipeline.py`, `test_cli.py::test_smoke_runs_without_network` |
| `lctx smoke` yields a heatmap, a curve, and a summary CSV | `test_cli.py::test_smoke_produces_a_heatmap_a_curve_and_a_summary_csv` |

---

## Known limitations

Stated plainly, because each one bounds what a result means.

- **The bundled filler recycles ~120 sentence templates** with randomized slots.
  Below roughly 32k tokens the repetition is mild; beyond it the filler becomes
  noticeably repetitive, which may make a needle *easier* to spot than it would
  be in real prose. Set `haystack.corpus_path` to a large plain-text file for
  long sweeps, and compare against `source: random_tokens` to bound the effect.
- **Realized depth is measured, never assumed.** At short targets the fixed
  prompt prefix is a large fraction of the context, so a requested depth of 0.0
  realizes around 0.1. The realized value is what gets recorded; use it, not the
  requested one, when interpreting shallow cells.
- **Filler is sliced on token boundaries**, so each needle has at most one
  mid-word artifact at its seam.
- **Anthropic token counting** uses a local BPE proxy calibrated against the
  provider's `count_tokens` endpoint, rather than spending an API call on every
  step of the packer's search. The final count is verified exactly.
- **`no_haystack` collapses** to one cell per (model, task, distractor, seed):
  with no filler there is no length or depth axis to sweep.
- **The simulator (`adapter: mock`) is not a language model.** It produces a known
  degradation curve so the harness can be checked against an answer we already
  have. Never cite its numbers.
- **Probe correlations are descriptive**, on a non-random sample of cells.

## License

MIT
