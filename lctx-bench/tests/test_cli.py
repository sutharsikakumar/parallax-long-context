"""Acceptance: the CLI, including `lctx smoke` end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lctx.cli import main


def test_smoke_produces_a_heatmap_a_curve_and_a_summary_csv(tmp_path, capsys):
    """ACCEPTANCE: `lctx smoke` runs end-to-end and emits all three artifacts."""
    # -n keeps CI fast; the shipped config uses enough seeds for real statistics.
    assert main(["smoke", "-o", str(tmp_path), "-n", "2"]) == 0
    out = capsys.readouterr().out
    assert "smoke OK" in out

    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    report = run_dir / "report"
    heatmaps = list(report.glob("heatmap*.png"))
    curves = list(report.glob("curves*.png"))

    assert heatmaps, "no heatmap produced"
    assert curves, "no accuracy-vs-length curve produced"
    assert (report / "summary_effective_context.csv").exists()
    assert all(p.stat().st_size > 1000 for p in heatmaps + curves)

    payload = json.loads((report / "summary.json").read_text())
    assert payload["totals"]["n_trials"] > 0
    assert payload["totals"]["tolerance_violations"] == 0


def test_smoke_runs_without_network(tmp_path, monkeypatch):
    """Nothing in the smoke path may open a socket."""
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("smoke run attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    assert main(["smoke", "-o", str(tmp_path), "-n", "2"]) == 0


def test_tasks_lists_every_task_and_scorer(capsys):
    assert main(["tasks"]) == 0
    out = capsys.readouterr().out
    for name in ("single_needle", "multi_needle", "multi_hop", "aggregation",
                 "selective_copy", "ordering", "negative_retrieval"):
        assert name in out
    for scorer in ("exact_match", "set_f1", "numeric_tol", "absent_detection"):
        assert scorer in out
    assert "custom_attn" in out


def test_validate_reports_statistical_power(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "grid: {lengths: [1000], depths: [0.5]}\n"
        "run: {n_trials: 2}\n"
        "models: [{name: m, adapter: mock}]\n"
        "tasks: [{name: single_needle}]\n"
    )
    assert main(["validate", "-c", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "power" in out
    # n=2 cannot possibly demonstrate an 0.8 threshold; the CLI must say so.
    assert "Raise run.n_trials" in out


def test_validate_rejects_a_bad_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("grid: {lengths: []}\nmodels: []\ntasks: []\n")
    with pytest.raises(SystemExit):
        main(["validate", "-c", str(bad)])


def test_missing_config_exits_cleanly():
    with pytest.raises(SystemExit, match="config not found"):
        main(["validate", "-c", "/nonexistent/path.yaml"])


def test_dry_run_prints_the_grid_without_calling_a_model(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "grid: {lengths: [1000, 2000], depths: [0.0, 1.0]}\n"
        "run: {n_trials: 3}\n"
        "models: [{name: m, adapter: mock}]\n"
        "tasks: [{name: single_needle}]\n"
    )
    assert main(["run", "-c", str(cfg), "--dry-run", "-o", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    # 2 lengths x 2 depths x 3 seeds = 12, plus 3 collapsed no-haystack ceiling cells
    assert "grid: 15 trials" in out
    assert "dry run" in out
    assert not list(tmp_path.glob("*/trials.jsonl"))


def test_run_then_report_rebuilds_from_the_run_directory(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"run: {{n_trials: 2, output_dir: {tmp_path / 'runs'}, store_prompts: none}}\n"
        "grid: {lengths: [800, 1600], depths: [0.0, 0.5, 1.0]}\n"
        "models: [{name: m, adapter: mock, params: {behavior: oracle}}]\n"
        "tasks: [{name: single_needle}]\n"
    )
    assert main(["run", "-c", str(cfg)]) == 0
    run_dir = next((tmp_path / "runs").iterdir())

    # A fresh report from the run directory alone must work.
    (run_dir / "report").rename(run_dir / "report_old")
    assert main(["report", str(run_dir)]) == 0
    assert (run_dir / "report" / "summary.json").exists()
    assert "effective context length" in capsys.readouterr().out


def test_resume_continues_an_interrupted_run(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"run: {{n_trials: 2, output_dir: {tmp_path / 'runs'}, store_prompts: none}}\n"
        "grid: {lengths: [800], depths: [0.5]}\n"
        "models: [{name: m, adapter: mock}]\n"
        "tasks: [{name: single_needle}]\n"
    )
    assert main(["run", "-c", str(cfg), "--no-report"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    before = (run_dir / "trials.jsonl").read_text().count("\n")

    assert main(["run", "-c", str(cfg), "--resume", str(run_dir), "--no-report"]) == 0
    after = (run_dir / "trials.jsonl").read_text().count("\n")
    assert after == before, "resume re-ran already-completed trials"


def test_probe_command_is_a_no_op_without_capable_models(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"run: {{n_trials: 1, output_dir: {tmp_path / 'runs'}, store_prompts: none}}\n"
        "grid: {lengths: [800], depths: [0.5]}\n"
        "models: [{name: m, adapter: mock}]\n"
        "tasks: [{name: single_needle}]\n"
    )
    assert main(["run", "-c", str(cfg), "--no-report"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    assert main(["probe", str(run_dir)]) == 0
    assert "no probe data" in capsys.readouterr().out
