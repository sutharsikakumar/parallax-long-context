"""Config-driven grid sweep: async execution, checkpointing, cost and latency logging."""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

from ..config import ExperimentConfig, ModelConfig
from ..haystack.filler import build_filler
from ..haystack.packing import Packer
from ..models.base import ModelAdapter
from ..models.registry import build_adapter
# Import the packages, not the submodules: importing registers the built-ins.
from ..scoring import get_scorer
from ..tasks import TaskSpec, get_task
from .ratelimit import RetryPolicy, TokenBucket, call_with_retry
from .records import TrialRecord, cell_id, prompt_hash, read_jsonl, store_prompt, trial_id

TRIALS_FILE = "trials.jsonl"
MANIFEST_FILE = "manifest.json"
CONFIG_SNAPSHOT = "config.yaml"
PROMPT_DIR = "prompts"


@dataclass(frozen=True)
class TrialPlan:
    """One unit of work in the sweep."""

    model: str
    task: str
    condition: str
    distractor_type: str
    target_tokens: int
    depth: float
    seed: int
    num_needles: int
    task_params: dict[str, Any] = field(default_factory=dict)

    @property
    def tid(self) -> str:
        return trial_id(
            self.model, self.task, self.condition, self.distractor_type,
            self.target_tokens, self.depth, self.seed, self.num_needles, self.task_params,
        )

    @property
    def cid(self) -> str:
        return cell_id(
            self.model, self.task, self.condition, self.distractor_type,
            self.target_tokens, self.depth,
        )


@dataclass
class TrialContext:
    """Read-only metadata handed to the adapter for each call."""

    trial_id: str
    cell_id: str
    task: str
    condition: str
    target_tokens: int
    depth: float
    seed: int
    #: Ground truth, populated *only* for adapters that declare is_simulator.
    simulator_hint: Any = None


def expand_grid(cfg: ExperimentConfig) -> list[TrialPlan]:
    """Enumerate every (model, task, distractor, condition, length, depth, seed) cell.

    Two collapses keep the grid honest rather than merely large:

    * ``no_haystack`` has no filler, so neither length nor depth applies. It runs
      once per (model, task, distractor, seed) and sets the ceiling.
    * Depth is skipped for tasks where a single needle position is meaningless
      only if the task says so; by default every task sees every depth.
    """
    plans: list[TrialPlan] = []
    seeds = cfg.run.seeds

    for model in cfg.models:
        for tc in cfg.tasks:
            params = dict(tc.params)
            num_needles = int(params.pop("num_needles", 1))
            task_obj = get_task(tc.name)
            task_obj.resolve(TaskSpec(task=tc.name, seed=0, params=params))  # validate early

            for distractor in tc.distractor_types:
                for condition in cfg.grid.conditions:
                    if condition == "no_haystack":
                        lengths: Sequence[int] = [0]
                        depths: Sequence[float] = [0.5]
                    else:
                        lengths = cfg.grid.lengths
                        depths = cfg.grid.depths
                    for length in lengths:
                        for depth in depths:
                            for seed in seeds:
                                plans.append(
                                    TrialPlan(
                                        model=model.name,
                                        task=tc.name,
                                        condition=condition,
                                        distractor_type=distractor,
                                        target_tokens=length,
                                        depth=depth,
                                        seed=seed,
                                        num_needles=num_needles,
                                        task_params=params,
                                    )
                                )
    return plans


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


@dataclass
class RunResult:
    run_dir: Path
    records: list[TrialRecord]
    skipped: int
    resumed: int
    elapsed_s: float

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)


class Runner:
    """Executes a sweep described by an :class:`ExperimentConfig`."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        run_dir: Path | str | None = None,
        progress: Callable[[str], None] | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.progress = progress or (lambda msg: None)
        self.config_path = Path(config_path) if config_path else None

        if run_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{stamp}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.run_dir / TRIALS_FILE
        self.prompt_dir = self.run_dir / PROMPT_DIR

        self._write_lock = asyncio.Lock()
        self._done = 0
        self._total = 0
        self._t0 = 0.0

    # -- persistence ------------------------------------------------------

    def _snapshot_config(self) -> None:
        with open(self.run_dir / CONFIG_SNAPSHOT, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.cfg.model_dump(mode="json"), fh, sort_keys=False)
        if self.config_path and self.config_path.exists():
            shutil.copy(self.config_path, self.run_dir / f"original-{self.config_path.name}")

    def _write_manifest(self, adapters: dict[str, dict[str, Any]], extra: dict[str, Any]) -> None:
        manifest = {
            "run_name": self.cfg.run.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "git_commit": _git_commit(),
            "adapters": adapters,
            "grid": self.cfg.grid.model_dump(mode="json"),
            "haystack": self.cfg.haystack.model_dump(mode="json"),
            "stats": self.cfg.stats.model_dump(mode="json"),
            "seeds": self.cfg.run.seeds,
            **extra,
        }
        with open(self.run_dir / MANIFEST_FILE, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    async def _append(self, record: TrialRecord) -> None:
        """Checkpoint a completed trial immediately.

        Per-trial rather than per-cell: a sweep interrupted mid-cell then loses
        nothing, and resume is exact.
        """
        async with self._write_lock:
            with open(self.trials_path, "a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
                fh.flush()
            self._done += 1
            if self._done % 10 == 0 or self._done == self._total:
                rate = self._done / max(1e-9, time.perf_counter() - self._t0)
                self.progress(
                    f"  {self._done}/{self._total} trials  ({rate:.1f}/s)"
                )

    # -- single trial -----------------------------------------------------

    async def _run_trial(
        self,
        plan: TrialPlan,
        adapter: ModelAdapter,
        mcfg: ModelConfig,
        packer: Packer,
        bucket: TokenBucket,
        sem: asyncio.Semaphore,
        policy: RetryPolicy,
    ) -> TrialRecord:
        task = get_task(plan.task)
        spec = TaskSpec(
            task=plan.task,
            seed=plan.seed,
            target_tokens=plan.target_tokens,
            depth=plan.depth,
            num_needles=plan.num_needles,
            distractor_type=plan.distractor_type,
            condition=plan.condition,
            params=plan.task_params,
        )
        inst = task.generate(spec)

        # Packing is CPU-bound and can dominate at long contexts; keep it off the
        # event loop so network calls stay concurrent.
        packed = await asyncio.to_thread(
            packer.pack, inst, plan.target_tokens, plan.condition, plan.seed
        )

        rec = TrialRecord(
            trial_id=plan.tid,
            cell_id=plan.cid,
            run_name=self.cfg.run.name,
            model=plan.model,
            task=plan.task,
            condition=plan.condition,
            distractor_type=plan.distractor_type,
            target_tokens=plan.target_tokens,
            depth=plan.depth,
            seed=plan.seed,
            num_needles=plan.num_needles,
            task_params=plan.task_params,
            actual_tokens=packed.actual_tokens,
            filler_tokens=packed.filler_tokens,
            chat_overhead_tokens=packed.chat_overhead_tokens,
            needle_positions=packed.needle_positions,
            needle_depths_requested=packed.needle_depths_requested,
            needle_depths_actual=packed.needle_depths_actual,
            within_tolerance=packed.within_tolerance,
            tolerance_checked=packed.tolerance_checked,
            pack_iterations=packed.search_iterations,
            scorer=inst.scorer,
            ground_truth=inst.ground_truth,
        )

        if self.cfg.run.store_prompts == "full":
            digest, rel = await asyncio.to_thread(
                store_prompt, self.prompt_dir, plan.tid, packed.messages
            )
            rec.prompt_sha256, rec.prompt_path = digest, f"{PROMPT_DIR}/{rel}"
        elif self.cfg.run.store_prompts == "hash":
            rec.prompt_sha256 = prompt_hash(packed.messages)

        ctx = TrialContext(
            trial_id=plan.tid,
            cell_id=plan.cid,
            task=plan.task,
            condition=plan.condition,
            target_tokens=plan.target_tokens,
            depth=plan.depth,
            seed=plan.seed,
            # Ground truth is handed over only to declared simulators.
            simulator_hint=task.oracle(inst) if getattr(adapter, "is_simulator", False) else None,
        )

        rng = random.Random(plan.tid)
        async with sem:
            await bucket.acquire()
            result = await call_with_retry(
                lambda: adapter.agenerate(
                    packed.messages,
                    max_tokens=mcfg.max_tokens,
                    temperature=mcfg.temperature,
                    stop=mcfg.stop or None,
                    trial=ctx,
                ),
                policy,
                timeout=self.cfg.run.request_timeout,
                rng=rng,
                on_retry=lambda a, e, d: self.progress(
                    f"  retry {a} for {plan.task}@{plan.target_tokens} in {d:.1f}s: {e}"
                ),
            )

        rec.attempts = result.attempts
        if result.error:
            rec.error = result.error
            return rec

        gen = result.value
        rec.raw_output = gen.text
        rec.input_tokens = gen.usage.input_tokens
        rec.output_tokens = gen.usage.output_tokens
        rec.latency_s = gen.latency_s
        rec.cost_usd = adapter.cost(gen.usage)

        scorer = get_scorer(inst.scorer)
        sr = scorer.score(gen.text, inst.ground_truth, **inst.scorer_params)
        rec.score = sr.score
        rec.correct = sr.correct
        rec.binary_scorer = scorer.binary
        rec.parsed_answer = sr.parsed
        rec.score_detail = sr.detail
        return rec

    # -- sweep ------------------------------------------------------------

    async def run(self, plans: Iterable[TrialPlan] | None = None) -> RunResult:
        t_start = time.perf_counter()
        self._t0 = t_start
        all_plans = list(plans) if plans is not None else expand_grid(self.cfg)

        done_ids = {r.trial_id for r in read_jsonl(self.trials_path)}
        resumed = sum(1 for p in all_plans if p.tid in done_ids)
        todo = [p for p in all_plans if p.tid not in done_ids]
        if resumed:
            self.progress(f"resuming: {resumed} trials already checkpointed")

        self._snapshot_config()
        filler = build_filler(self.cfg.haystack)
        policy = RetryPolicy(
            max_retries=self.cfg.run.max_retries,
            base_delay=self.cfg.run.retry_base_delay,
            max_delay=self.cfg.run.retry_max_delay,
        )

        adapters_info: dict[str, dict[str, Any]] = {}
        new_records: list[TrialRecord] = []
        skipped = 0

        self._total = len(todo)
        # Models run one at a time: a local model should not have to share a GPU
        # with another, and hosted rate limits are per-provider anyway.
        for mcfg in self.cfg.models:
            model_plans = [p for p in todo if p.model == mcfg.name]
            if not model_plans:
                continue

            self.progress(f"model {mcfg.name}: {len(model_plans)} trials")
            adapter = build_adapter(mcfg)
            try:
                adapters_info[mcfg.name] = adapter.describe()
                packer = Packer(adapter.tokenizer, filler, self.cfg.grid.tolerance)

                if self.cfg.run.skip_over_context_limit and adapter.context_limit:
                    budget = adapter.context_limit - mcfg.max_tokens
                    keep = [p for p in model_plans if p.target_tokens <= budget]
                    skipped += len(model_plans) - len(keep)
                    if len(keep) != len(model_plans):
                        self.progress(
                            f"  skipping {len(model_plans) - len(keep)} trials over "
                            f"{mcfg.name}'s {adapter.context_limit}-token limit"
                        )
                        self._total -= len(model_plans) - len(keep)
                    model_plans = keep

                bucket = TokenBucket(self.cfg.run.requests_per_minute)
                sem = asyncio.Semaphore(max(1, self.cfg.run.max_concurrency))

                async def one(plan: TrialPlan) -> None:
                    rec = await self._run_trial(
                        plan, adapter, mcfg, packer, bucket, sem, policy
                    )
                    new_records.append(rec)
                    await self._append(rec)

                await asyncio.gather(*(one(p) for p in model_plans))
            finally:
                await adapter.aclose()

        elapsed = time.perf_counter() - t_start
        all_records = read_jsonl(self.trials_path)
        self._write_manifest(
            adapters_info,
            {
                "n_planned": len(all_plans),
                "n_executed": len(new_records),
                "n_resumed": resumed,
                "n_skipped": skipped,
                "n_errors": sum(1 for r in all_records if not r.ok),
                "elapsed_s": elapsed,
                "total_cost_usd": sum(r.cost_usd for r in all_records),
            },
        )
        return RunResult(
            run_dir=self.run_dir,
            records=all_records,
            skipped=skipped,
            resumed=resumed,
            elapsed_s=elapsed,
        )


def run_sweep(
    cfg: ExperimentConfig,
    run_dir: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
    config_path: Path | str | None = None,
) -> RunResult:
    """Blocking entry point for a sweep."""
    runner = Runner(cfg, run_dir=run_dir, progress=progress, config_path=config_path)
    return asyncio.run(runner.run())
