"""Acceptance: the full pipeline runs offline, checkpoints, resumes, and reports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lctx.config import ExperimentConfig
from lctx.report.build import build_report
from lctx.runner.records import read_jsonl
from lctx.runner.sweep import Runner, expand_grid


def _run(cfg: ExperimentConfig, run_dir=None, plans=None):
    runner = Runner(cfg, run_dir=run_dir)
    return asyncio.run(runner.run(plans))


def test_full_pipeline_offline(mini_config):
    """ACCEPTANCE: mock adapter drives the whole pipeline with no network."""
    result = _run(mini_config)
    plans = expand_grid(mini_config)

    assert len(result.records) == len(plans)
    assert result.n_errors == 0
    assert all(r.scorer for r in result.records)
    assert (result.run_dir / "trials.jsonl").exists()
    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "manifest.json").exists()


def test_manifest_records_everything_needed_to_reproduce(mini_config):
    """CORE REQUIREMENT #6: config, seeds, and adapter identity are all logged."""
    result = _run(mini_config)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == mini_config.run.seeds
    assert "sim" in manifest["adapters"]
    assert manifest["adapters"]["sim"]["tokenizer"] == "word"
    assert manifest["grid"]["lengths"] == mini_config.grid.lengths
    # The snapshot must round-trip back into a usable config.
    ExperimentConfig.from_yaml(result.run_dir / "config.yaml")


def test_prompts_are_reproducible_from_the_log(mini_config):
    """The same trial_id must rebuild a byte-identical prompt."""
    mini_config.run.store_prompts = "hash"
    result = _run(mini_config)

    from lctx.haystack.filler import build_filler
    from lctx.haystack.packing import Packer
    from lctx.models.registry import build_adapter
    from lctx.runner.records import prompt_hash
    from lctx.tasks import TaskSpec, get_task

    adapter = build_adapter(mini_config.models[0])
    packer = Packer(adapter.tokenizer, build_filler(mini_config.haystack),
                    mini_config.grid.tolerance)
    for rec in result.records[:12]:
        inst = get_task(rec.task).generate(
            TaskSpec(task=rec.task, seed=rec.seed, target_tokens=rec.target_tokens,
                     depth=rec.depth, num_needles=rec.num_needles,
                     distractor_type=rec.distractor_type, condition=rec.condition,
                     params=rec.task_params)
        )
        packed = packer.pack(inst, rec.target_tokens, rec.condition, rec.seed)
        assert prompt_hash(packed.messages) == rec.prompt_sha256


def test_full_prompt_storage_writes_readable_files(mini_config, tmp_path):
    import gzip

    mini_config.run.store_prompts = "full"
    mini_config.grid.lengths = [600]
    result = _run(mini_config)
    rec = next(r for r in result.records if r.prompt_path)
    path = result.run_dir / rec.prompt_path
    assert path.exists()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        messages = json.load(fh)
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"


def test_checkpoint_resume_skips_completed_trials(mini_config):
    """An interrupted sweep must continue, not restart."""
    plans = expand_grid(mini_config)
    first = _run(mini_config, plans=plans[:10])
    assert len(read_jsonl(first.run_dir / "trials.jsonl")) == 10

    second = _run(mini_config, run_dir=first.run_dir, plans=plans)
    assert second.resumed == 10
    assert len(second.records) == len(plans)
    ids = [r.trial_id for r in second.records]
    assert len(ids) == len(set(ids)), "resume duplicated trials"


def test_trial_ids_are_stable_across_runs(mini_config):
    a = {p.tid for p in expand_grid(mini_config)}
    b = {p.tid for p in expand_grid(mini_config)}
    assert a == b and len(a) == len(expand_grid(mini_config))


def test_task_params_participate_in_the_trial_id(mini_config):
    """Two configs differing only in a task param must not collide on resume."""
    from lctx.runner.records import trial_id

    base = dict(model="m", task="aggregation", condition="standard",
                distractor_type="none", target_tokens=1000, depth=0.5, seed=0,
                num_needles=1)
    assert trial_id(**base, params={"op": "sum"}) != trial_id(**base, params={"op": "max"})


def test_errors_are_recorded_without_aborting_the_sweep(mini_config):
    """One failing model must not take the run down."""
    from lctx.models.base import ModelAdapter, TransientError
    from lctx.models.registry import register_adapter
    from lctx.models.tokenizers import WordTokenizer

    class AlwaysFails(ModelAdapter):
        def __init__(self, name="boom", tokenizer=None, **kw):
            super().__init__(name, tokenizer or WordTokenizer())

        async def agenerate(self, messages, max_tokens=256, temperature=0.0,
                            stop=None, trial=None):
            raise TransientError("simulated outage")

    register_adapter("always_fails", lambda **kw: AlwaysFails(**kw))
    mini_config.models.append(
        mini_config.models[0].model_copy(
            update={"name": "broken", "adapter": "always_fails", "params": {}}
        )
    )
    mini_config.run.max_retries = 1
    mini_config.run.retry_base_delay = 0.001

    result = _run(mini_config)
    broken = [r for r in result.records if r.model == "broken"]
    good = [r for r in result.records if r.model == "sim"]
    assert broken and all(r.error for r in broken)
    assert good and not any(r.error for r in good)
    assert all(r.attempts == 2 for r in broken)  # initial attempt plus one retry


def test_report_generates_figures_and_tables(mini_config):
    """ACCEPTANCE: reporting runs in CI with no display and no network."""
    result = _run(mini_config)
    report = build_report(result.run_dir, mini_config.stats)

    assert any(p.name.startswith("heatmap") for p in report.figures)
    assert any(p.name.startswith("curves") for p in report.figures)
    assert all(p.exists() and p.stat().st_size > 1000 for p in report.figures)

    names = {p.name for p in report.tables}
    for expected in ("summary_effective_context.csv", "cells.csv", "cost_latency.csv",
                     "packing_audit.csv", "summary.json"):
        assert expected in names

    payload = json.loads((report.report_dir / "summary.json").read_text())
    assert payload["totals"]["n_trials"] == len(result.records)
    assert payload["totals"]["tolerance_violations"] == 0


def test_oracle_model_scores_perfectly_end_to_end(mini_config):
    """A simulator that always answers correctly must score 1.0 everywhere.

    This closes the loop: generator, packer, adapter, extractor and scorer all
    have to agree for this to pass.
    """
    mini_config.models[0].params = {"behavior": "oracle"}
    result = _run(mini_config)
    assert all(r.correct for r in result.records if r.ok)
    assert all(r.score == 1.0 for r in result.records if r.ok)


def test_context_limit_skips_oversized_cells(mini_config):
    mini_config.models[0].context_limit = 1000
    result = _run(mini_config)
    assert result.skipped > 0
    assert all(r.target_tokens <= 1000 for r in result.records)


def test_cost_and_latency_are_logged(mini_config):
    mini_config.models[0].pricing = type(mini_config.models[0].pricing)(
        input_per_mtok=3.0, output_per_mtok=15.0
    )
    result = _run(mini_config)
    assert result.total_cost > 0
    assert all(r.input_tokens > 0 for r in result.records if r.ok)

    from lctx.report.tables import cost_latency_rows

    row = cost_latency_rows(result.records)[0]
    assert row["total_cost_usd"] == pytest.approx(result.total_cost)
    assert row["p95_latency_s"] >= row["p50_latency_s"]


def test_no_haystack_condition_is_collapsed_to_one_cell(mini_config):
    """The ceiling control has no length or depth axis, so it must not be swept."""
    plans = expand_grid(mini_config)
    ceiling = [p for p in plans if p.condition == "no_haystack"]
    assert ceiling
    assert {p.target_tokens for p in ceiling} == {0}
    assert {p.depth for p in ceiling} == {0.5}


def test_grid_size_is_exactly_the_product_of_its_axes(mini_config):
    plans = expand_grid(mini_config)
    n_models, n_tasks = len(mini_config.models), len(mini_config.tasks)
    n_seeds = mini_config.run.n_trials
    swept = len(mini_config.grid.lengths) * len(mini_config.grid.depths)
    expected = n_models * n_tasks * n_seeds * (swept + 1)  # +1 for the collapsed ceiling
    assert len(plans) == expected


def test_report_survives_a_run_where_everything_errored(mini_config):
    """A totally failed run must still produce the report that explains why."""
    from lctx.models.base import ModelAdapter, TransientError
    from lctx.models.registry import register_adapter
    from lctx.models.tokenizers import WordTokenizer

    class Dead(ModelAdapter):
        def __init__(self, name="dead", tokenizer=None, **kw):
            super().__init__(name, tokenizer or WordTokenizer())

        async def agenerate(self, messages, max_tokens=256, temperature=0.0,
                            stop=None, trial=None):
            raise TransientError("provider down")

    register_adapter("dead_for_report", lambda **kw: Dead(**kw))
    mini_config.models = [
        mini_config.models[0].model_copy(
            update={"name": "dead", "adapter": "dead_for_report", "params": {}}
        )
    ]
    mini_config.run.max_retries = 0
    result = _run(mini_config)
    assert result.n_errors == len(result.records)

    report = build_report(result.run_dir, mini_config.stats)
    assert any("errored" in w for w in report.warnings)
    assert (report.report_dir / "summary.json").exists()


def test_concurrency_bound_covers_the_whole_trial(mini_config):
    """max_concurrency must bound packing too, not just the model call.

    Packing a long prompt allocates real memory; if only the request were
    bounded, every planned trial would pack up front and a large grid at long
    context would exhaust memory before any model was called.
    """
    import threading

    from lctx.models.base import GenerationResult, ModelAdapter, Usage
    from lctx.models.registry import register_adapter
    from lctx.models.tokenizers import WordTokenizer

    state = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    class Counting(ModelAdapter):
        is_simulator = True

        def __init__(self, name="counting", tokenizer=None, **kw):
            super().__init__(name, tokenizer or WordTokenizer())

        async def agenerate(self, messages, max_tokens=256, temperature=0.0,
                            stop=None, trial=None):
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            try:
                await asyncio.sleep(0.005)  # hold the slot open
                return GenerationResult(text="<answer>x</answer>", usage=Usage(1, 1))
            finally:
                with lock:
                    state["in_flight"] -= 1

    register_adapter("counting", lambda **kw: Counting(**kw))
    mini_config.models = [
        mini_config.models[0].model_copy(
            update={"name": "counting", "adapter": "counting", "params": {}}
        )
    ]
    mini_config.run.max_concurrency = 3

    _run(mini_config)
    assert state["peak"] <= 3, f"peak concurrency {state['peak']} exceeded the limit of 3"
    assert state["peak"] > 1, "the sweep did not run concurrently at all"
