"""Running attention probes over a (sub)grid.

Probes are expensive — a dense attention matrix is quadratic in context length —
so they run over a *sample* of the grid rather than all of it. Because prompt
construction is fully deterministic, a probe pass regenerates exactly the prompts
a completed run already used, and its metrics join back to that run's trial
records by ``trial_id``. That join is what makes the "does attention mass on the
needle predict recall?" question answerable.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from ..config import ExperimentConfig
from ..haystack.filler import build_filler
from ..haystack.packing import Packer
from ..models.base import ModelAdapter
from ..models.registry import build_adapter
from ..runner.records import TrialRecord, read_jsonl
from ..runner.sweep import TrialPlan, expand_grid
from ..tasks import TaskSpec, get_task
from .attention import ProbeResult, analyze_capture

PROBES_FILE = "probes.jsonl"


def locate_needle_spans(
    adapter: ModelAdapter, messages: Sequence[dict[str, str]], needle_texts: Sequence[str]
) -> list[tuple[int, int]]:
    """Token spans of each needle inside the prompt *as the model renders it*.

    The packer's recorded positions are measured on its own assembly; a local
    model may apply a chat template that shifts them. Re-locating against the
    rendered string keeps the spans aligned with the attention matrix.
    """
    render = getattr(adapter, "render", None)
    text = render(messages) if callable(render) else "\n\n".join(m["content"] for m in messages)
    tok = adapter.tokenizer
    spans: list[tuple[int, int]] = []
    for needle in needle_texts:
        idx = text.find(needle)
        if idx < 0:
            continue
        start = tok.count_tokens(text[:idx])
        spans.append((start, start + tok.count_tokens(needle)))
    return spans


def sample_plans(
    plans: Sequence[TrialPlan],
    max_cells: int,
    seed: int = 0,
    conditions: Sequence[str] = ("standard",),
    max_tokens: int | None = None,
) -> list[TrialPlan]:
    """Pick a spread of cells to probe, biased to cover the length x depth plane."""
    pool = [p for p in plans if p.condition in set(conditions)]
    if max_tokens is not None:
        pool = [p for p in pool if p.target_tokens <= max_tokens]
    if not pool:
        return []
    # One seed per (task, length, depth) cell keeps coverage wide rather than deep.
    by_cell: dict[tuple, TrialPlan] = {}
    for p in sorted(pool, key=lambda x: (x.task, x.target_tokens, x.depth, x.seed)):
        by_cell.setdefault((p.task, p.target_tokens, p.depth), p)
    cells = list(by_cell.values())
    if len(cells) <= max_cells:
        return cells
    rng = random.Random(seed)
    return sorted(
        rng.sample(cells, max_cells), key=lambda x: (x.task, x.target_tokens, x.depth)
    )


def probe_plan(
    adapter: ModelAdapter,
    packer: Packer,
    plan: TrialPlan,
    record: TrialRecord | None = None,
) -> dict[str, Any] | None:
    """Capture and analyse attention for one cell. ``None`` if unsupported."""
    task = get_task(plan.task)
    spec = TaskSpec(
        task=plan.task, seed=plan.seed, target_tokens=plan.target_tokens, depth=plan.depth,
        num_needles=plan.num_needles, distractor_type=plan.distractor_type,
        condition=plan.condition, params=plan.task_params,
    )
    inst = task.generate(spec)
    packed = packer.pack(inst, plan.target_tokens, plan.condition, plan.seed)

    capture = adapter.capture_attention(packed.messages)
    if capture is None:
        return None

    target_texts = [n.text for n in inst.sorted_needles() if n.role == "target"]
    spans = locate_needle_spans(adapter, packed.messages, target_texts)
    result: ProbeResult = analyze_capture(
        capture,
        spans,
        meta={
            "model": adapter.name,
            "task": plan.task,
            "condition": plan.condition,
            "distractor_type": plan.distractor_type,
            "target_tokens": plan.target_tokens,
            "depth": plan.depth,
            "seed": plan.seed,
            "trial_id": plan.tid,
        },
    )
    return {
        "trial_id": plan.tid,
        "model": adapter.name,
        "task": plan.task,
        "condition": plan.condition,
        "distractor_type": plan.distractor_type,
        "target_tokens": plan.target_tokens,
        "depth": plan.depth,
        "seed": plan.seed,
        "actual_tokens": packed.actual_tokens,
        "input_token_count": result.input_token_count,
        "needle_spans": result.needle_spans,
        "mean_needle_mass": result.mean_needle_mass,
        "max_needle_mass": result.max_needle_mass,
        # Recall outcome for the same prompt, when the sweep has already run it.
        "correct": (record.correct if record else None),
        "score": (record.score if record else None),
        "layers": [asdict(m) for m in result.layers],
    }


def run_probes(
    cfg: ExperimentConfig,
    run_dir: Path | str,
    max_cells: int = 24,
    max_tokens: int | None = 16_000,
    conditions: Sequence[str] = ("standard",),
    seed: int = 0,
    progress: Any = None,
) -> Path | None:
    """Probe a sample of cells for every model with ``capture_attention: true``.

    Returns the path to ``probes.jsonl``, or ``None`` when no model supports
    attention capture (the normal case for hosted APIs and fused kernels).
    """
    run_dir = Path(run_dir)
    say = progress or (lambda _msg: None)
    records = {r.trial_id: r for r in read_jsonl(run_dir / "trials.jsonl")}
    all_plans = expand_grid(cfg)
    filler = build_filler(cfg.haystack)
    out_path = run_dir / PROBES_FILE
    wrote = 0

    for mcfg in cfg.models:
        if not mcfg.capture_attention:
            continue
        adapter = build_adapter(mcfg)
        try:
            packer = Packer(adapter.tokenizer, filler, cfg.grid.tolerance)
            plans = sample_plans(
                [p for p in all_plans if p.model == mcfg.name],
                max_cells, seed, conditions, max_tokens,
            )
            say(f"probing {mcfg.name}: {len(plans)} cells")
            for plan in plans:
                row = probe_plan(adapter, packer, plan, records.get(plan.tid))
                if row is None:
                    say(
                        f"  {mcfg.name} does not expose attention weights "
                        "(fused kernel?); load it with attn_implementation: eager"
                    )
                    break
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
                wrote += 1
        finally:
            import asyncio

            asyncio.run(adapter.aclose())

    if not wrote:
        return None
    say(f"wrote {wrote} probe rows to {out_path}")
    return out_path
