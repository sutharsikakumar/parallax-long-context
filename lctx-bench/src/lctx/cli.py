"""Command-line interface: ``lctx``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import ExperimentConfig
from .report.build import build_report
from .runner.sweep import Runner, expand_grid
from .scoring import all_scorers
from .tasks import all_tasks

SMOKE_CONFIG = Path(__file__).parent / "configs" / "smoke.yaml"


def _say(msg: str) -> None:
    print(msg, flush=True)


def _load(path: str) -> ExperimentConfig:
    try:
        return ExperimentConfig.from_yaml(path)
    except FileNotFoundError:
        raise SystemExit(f"config not found: {path}")
    except Exception as exc:
        raise SystemExit(f"invalid config {path}:\n  {exc}")


def _report(run_dir: Path, cfg: ExperimentConfig | None) -> None:
    result = build_report(run_dir, cfg.stats if cfg else None)
    _say(f"\nreport: {result.report_dir}")
    _say(f"  {len(result.figures)} figures, {len(result.tables)} tables")
    for w in result.warnings:
        _say(f"  ! {w}")

    effs = result.payload.get("effective_context", [])
    if effs:
        _say("\neffective context length (lower CI bound >= threshold):")
        _say(
            f"  {'model':<16} {'task':<16} {'condition':<15} "
            f"{'distractors':<19} {'strict':>9} {'max':>9}"
        )
        for e in effs:
            k = e["key"]
            strict = e["effective_strict"]
            mx = e["effective_max"]
            _say(
                f"  {k['model']:<16} {k['task']:<16} {k['condition']:<15} "
                f"{k['distractor_type']:<19} "
                f"{(str(strict) if strict else '—'):>9} {(str(mx) if mx else '—'):>9}"
            )


# -- commands -------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    if args.n_trials:
        cfg.run.n_trials = args.n_trials
    if args.output:
        cfg.run.output_dir = args.output

    plans = expand_grid(cfg)
    _say(f"lctx-bench {__version__}")
    _say(
        f"grid: {len(plans)} trials "
        f"({len(cfg.models)} models x {len(cfg.tasks)} tasks x "
        f"{len(cfg.grid.lengths)} lengths x {len(cfg.grid.depths)} depths x "
        f"{cfg.run.n_trials} seeds, conditions: {', '.join(cfg.grid.conditions)})"
    )
    if args.dry_run:
        _say("dry run: no models called")
        return 0

    import asyncio

    runner = Runner(cfg, run_dir=args.resume, progress=_say, config_path=args.config)
    result = asyncio.run(runner.run(plans))
    _say(
        f"\ndone in {result.elapsed_s:.1f}s: {len(result.records)} trials, "
        f"{result.n_errors} errors, {result.skipped} skipped, "
        f"${result.total_cost:.4f}"
    )

    if args.probes:
        from .probes.run import run_probes

        run_probes(cfg, result.run_dir, max_cells=args.probe_cells, progress=_say)

    if not args.no_report:
        _report(result.run_dir, cfg)
    return 1 if result.n_errors and args.strict else 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    cfg = _load(str(cfg_path)) if cfg_path.exists() else None
    _report(run_dir, cfg)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Tiny end-to-end grid: runs offline in seconds and produces a full report."""
    cfg = _load(str(SMOKE_CONFIG))
    if args.output:
        cfg.run.output_dir = args.output
    if args.n_trials:
        cfg.run.n_trials = args.n_trials

    import asyncio

    _say(f"lctx-bench {__version__} — smoke test (mock adapter, no network)")
    plans = expand_grid(cfg)
    _say(f"grid: {len(plans)} trials")
    runner = Runner(cfg, progress=_say, config_path=SMOKE_CONFIG)
    result = asyncio.run(runner.run(plans))
    _say(f"\nran {len(result.records)} trials in {result.elapsed_s:.1f}s, "
         f"{result.n_errors} errors")

    report = build_report(result.run_dir, cfg.stats)
    heat = [p for p in report.figures if p.name.startswith("heatmap")]
    curve = [p for p in report.figures if p.name.startswith("curves")]
    csvs = [p for p in report.tables if p.suffix == ".csv"]

    # The acceptance criterion for --smoke: a heatmap, a curve, and a summary CSV.
    missing = []
    if not heat:
        missing.append("heatmap")
    if not curve:
        missing.append("curve")
    if not any(p.name == "summary_effective_context.csv" for p in csvs):
        missing.append("summary CSV")
    if missing:
        _say(f"FAILED: smoke run produced no {', '.join(missing)}")
        return 1

    _say(f"\nsmoke OK — {len(heat)} heatmaps, {len(curve)} curve figures, "
         f"{len(csvs)} CSV tables")
    _say(f"  {report.report_dir}")
    for w in report.warnings:
        _say(f"  ! {w}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    plans = expand_grid(cfg)
    _say(f"config OK: {args.config}")
    _say(f"  models: {', '.join(m.name + ' (' + m.adapter + ')' for m in cfg.models)}")
    _say(f"  tasks:  {', '.join(t.name for t in cfg.tasks)}")
    _say(f"  grid:   {len(plans)} trials, tolerance ±{cfg.grid.tolerance:.1%}")

    # Wilson lower bound at n trials for a perfect cell — the sample size needed
    # to be able to *demonstrate* the effective-context threshold at all.
    from .stats.intervals import wilson_interval

    n = cfg.run.n_trials * len(cfg.grid.depths)
    lo = wilson_interval(n, n, cfg.stats.ci).low
    thr = cfg.stats.effective_context_threshold
    _say(f"  power:  a perfect length (n={n} pooled over depths) has a "
         f"{cfg.stats.ci:.0%} lower bound of {lo:.3f}")
    if lo < thr:
        need = 1
        while wilson_interval(need, need, cfg.stats.ci).low < thr and need < 10_000:
            need += 1
        _say(
            f"  ! that is below the {thr:.2f} threshold, so no length can ever be "
            f"reported as effective. Raise run.n_trials to >= "
            f"{-(-need // max(1, len(cfg.grid.depths)))} (>= {need} pooled trials), "
            f"or lower stats.effective_context_threshold."
        )
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    _say("tasks:")
    for name, t in sorted(all_tasks().items()):
        _say(f"  {name:<20} scorer={t.scorer:<17} {t.description}")
        if t.defaults:
            _say(f"  {'':<20} params: {json.dumps(t.defaults, default=str)}")
    _say("\nscorers:")
    for name, s in sorted(all_scorers().items()):
        _say(f"  {name:<20} {'binary' if s.binary else 'continuous'}")
    _say("\nadapters: mock, echo, anthropic, openai, hf, custom_attn")
    _say("  (or any 'package.module:callable' returning a ModelAdapter)")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from .probes.run import run_probes

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_dir}; probes need the run's config")
    cfg = _load(str(cfg_path))
    path = run_probes(cfg, run_dir, max_cells=args.cells, max_tokens=args.max_tokens,
                      progress=_say)
    if path is None:
        _say(
            "no probe data written: no configured model both sets capture_attention "
            "and exposes attention weights (fused kernels do not; use "
            "attn_implementation: eager)"
        )
        return 0
    _report(run_dir, cfg)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    import asyncio

    from .models.registry import build_adapter
    from .runner.records import read_jsonl
    from .scoring.judge import LLMJudge, validate_judge

    run_dir = Path(args.run_dir)
    cfg = _load(str(run_dir / "config.yaml"))
    if not cfg.judge.enabled or not cfg.judge.model:
        raise SystemExit("judge is not enabled in this run's config (set judge.enabled)")

    records = read_jsonl(run_dir / "trials.jsonl")
    adapter = build_adapter(cfg.model_by_name(cfg.judge.model))
    judge = LLMJudge(adapter)
    try:
        report = asyncio.run(
            validate_judge(
                records, judge,
                fraction=cfg.judge.validation_fraction,
                seed=cfg.judge.validation_seed,
                min_agreement=cfg.judge.min_agreement,
            )
        )
    finally:
        asyncio.run(adapter.aclose())

    out = run_dir / "report" / "judge_agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    _say(f"judge vs deterministic on n={report.n}:")
    _say(f"  agreement {report.agreement:.3f}  kappa {report.cohen_kappa:.3f}")
    _say(f"  judge false positives {report.false_positive_rate:.3f}, "
         f"false negatives {report.false_negative_rate:.3f}")
    _say(f"  -> {report.as_dict()['verdict']}")
    _say(f"  written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lctx",
        description="lctx-bench: long-context evaluation with statistical rigor.",
    )
    p.add_argument("--version", action="version", version=f"lctx-bench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a sweep from a config file")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("-o", "--output", help="override run.output_dir")
    r.add_argument("--resume", help="resume into an existing run directory")
    r.add_argument("-n", "--n-trials", type=int, help="override run.n_trials")
    r.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--probes", action="store_true", help="run attention probes afterwards")
    r.add_argument("--probe-cells", type=int, default=24)
    r.add_argument("--strict", action="store_true", help="exit non-zero if any trial errored")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="(re)build figures and tables for a run")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    s = sub.add_parser("smoke", help="tiny offline end-to-end run (no network)")
    s.add_argument("-o", "--output")
    s.add_argument("-n", "--n-trials", type=int,
                   help="override run.n_trials (smaller runs faster; CI uses this)")
    s.set_defaults(func=cmd_smoke)

    v = sub.add_parser("validate", help="check a config and report statistical power")
    v.add_argument("-c", "--config", required=True)
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("tasks", help="list tasks, scorers, and adapters")
    t.set_defaults(func=cmd_tasks)

    pr = sub.add_parser("probe", help="run attention probes over an existing run")
    pr.add_argument("run_dir")
    pr.add_argument("--cells", type=int, default=24)
    pr.add_argument("--max-tokens", type=int, default=16000)
    pr.set_defaults(func=cmd_probe)

    j = sub.add_parser("judge", help="validate the LLM judge against deterministic scores")
    j.add_argument("run_dir")
    j.set_defaults(func=cmd_judge)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _say("\ninterrupted; completed trials are checkpointed and `--resume` will continue")
        return 130


if __name__ == "__main__":
    sys.exit(main())
