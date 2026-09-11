"""Config validation: bad configs must fail loudly and early."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lctx.config import ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PACKAGED = Path(__file__).resolve().parents[1] / "src" / "lctx" / "configs"


def _base(**over) -> dict:
    cfg = {
        "grid": {"lengths": [1000], "depths": [0.5]},
        "models": [{"name": "m", "adapter": "mock"}],
        "tasks": [{"name": "single_needle"}],
    }
    cfg.update(over)
    return cfg


def test_minimal_config_is_valid():
    cfg = ExperimentConfig.model_validate(_base())
    assert cfg.run.n_trials == 5
    assert cfg.grid.tolerance == 0.02
    assert cfg.stats.effective_context_threshold == 0.8


@pytest.mark.parametrize(
    "grid",
    [
        {"lengths": [], "depths": [0.5]},
        {"lengths": [-1], "depths": [0.5]},
        {"lengths": [1000], "depths": [1.5]},
        {"lengths": [1000], "depths": [-0.1]},
        {"lengths": [1000], "depths": [0.5], "tolerance": 0},
        {"lengths": [1000], "depths": [0.5], "tolerance": 1.5},
    ],
)
def test_invalid_grids_are_rejected(grid):
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(_base(grid=grid))


def test_typos_are_rejected_rather_than_ignored():
    """extra='forbid' everywhere: a misspelled key must not silently do nothing."""
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(_base(grid={"lengths": [1], "depths": [0.5],
                                                    "tolerence": 0.02}))
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(_base(run={"n_trails": 5}))


def test_duplicate_model_names_are_rejected():
    with pytest.raises(Exception, match="unique"):
        ExperimentConfig.model_validate(
            _base(models=[{"name": "m", "adapter": "mock"},
                          {"name": "m", "adapter": "mock"}])
        )


def test_judge_must_name_a_configured_model():
    with pytest.raises(Exception, match="judge.model"):
        ExperimentConfig.model_validate(
            _base(judge={"enabled": True, "model": "nope"})
        )


def test_seeds_follow_from_base_seed_and_n_trials():
    cfg = ExperimentConfig.model_validate(_base(run={"n_trials": 4, "base_seed": 100}))
    assert cfg.run.seeds == [100, 101, 102, 103]


def test_adapters_requiring_a_model_id_say_so():
    from lctx.models.registry import build_adapter

    cfg = ExperimentConfig.model_validate(
        _base(models=[{"name": "a", "adapter": "anthropic"}])
    )
    with pytest.raises(ValueError, match="model_id"):
        build_adapter(cfg.models[0])


def test_unknown_adapter_names_list_the_alternatives():
    from lctx.models.registry import build_adapter

    cfg = ExperimentConfig.model_validate(_base(models=[{"name": "a", "adapter": "nope"}]))
    with pytest.raises(KeyError, match="unknown adapter"):
        build_adapter(cfg.models[0])


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_shipped_configs_are_valid(path):
    """Every example config in the repo must parse and expand into a grid."""
    from lctx.runner.sweep import expand_grid

    cfg = ExperimentConfig.from_yaml(path)
    assert expand_grid(cfg), f"{path.name} expands to an empty grid"


def test_packaged_smoke_config_matches_the_repo_copy():
    """configs/smoke.yaml and the packaged copy must not drift apart."""
    a = yaml.safe_load((CONFIG_DIR / "smoke.yaml").read_text())
    b = yaml.safe_load((PACKAGED / "smoke.yaml").read_text())
    assert a == b


def test_config_round_trips_through_yaml(tmp_path):
    cfg = ExperimentConfig.model_validate(_base())
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert ExperimentConfig.from_yaml(path) == cfg
