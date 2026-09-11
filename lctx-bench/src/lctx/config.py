"""Pydantic configuration schema for lctx-bench experiments.

A single YAML file fully determines a run. Nothing about a model, task, or
grid point is hardcoded anywhere else in the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Condition = Literal["standard", "no_haystack", "shuffled_haystack", "needle_absent"]
DistractorType = Literal["none", "near_duplicate_key", "semantic_lure", "repeated_decoy"]


class TokenizerConfig(BaseModel):
    """How to count tokens for a model.

    ``kind`` selects the implementation:
      - ``word``     : dependency-free reversible tokenizer (offline / CI).
      - ``tiktoken`` : OpenAI BPE, ``encoding`` names the encoding.
      - ``hf``       : ``transformers`` tokenizer, ``name_or_path`` names the repo.
      - ``anthropic``: authoritative ``count_tokens`` API with a calibrated local proxy.
    """

    model_config = {"extra": "forbid"}

    kind: Literal["word", "tiktoken", "hf", "anthropic"] = "word"
    encoding: str = "cl100k_base"
    name_or_path: str | None = None
    # Chat-template overhead is measured empirically, but a model may need a
    # per-message constant when its template is not introspectable.
    per_message_overhead: int = 0
    # For the anthropic kind: how many exact API calls the packer may spend
    # refining a prompt after the calibrated proxy search.
    exact_refine_budget: int = 4


class PricingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class ModelConfig(BaseModel):
    """One model under test. ``adapter`` names a registered ModelAdapter."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    name: str
    adapter: str
    # Provider-side model identifier, passed through to the adapter untouched.
    model_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    context_limit: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    stop: list[str] = Field(default_factory=list)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    # Probe capture is opt-in and only supported by local adapters.
    capture_attention: bool = False


class TaskConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    distractor_types: list[DistractorType] = Field(default_factory=lambda: ["none"])


class HaystackConfig(BaseModel):
    model_config = {"extra": "forbid"}

    source: Literal["corpus", "random_tokens"] = "corpus"
    corpus_path: str | None = None
    # Vocabulary size for the random-token ablation.
    random_vocab_size: int = 4096


class GridConfig(BaseModel):
    model_config = {"extra": "forbid"}

    lengths: list[int]
    depths: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    # The no-haystack ceiling is on by default: it costs one cell per task and
    # without it a degradation curve cannot be attributed to context length at
    # all. The costlier controls (shuffled_haystack, needle_absent) multiply the
    # grid and so stay opt-in.
    conditions: list[Condition] = Field(
        default_factory=lambda: ["standard", "no_haystack"]
    )
    # Fractional tolerance on the realized token count of every prompt.
    tolerance: float = 0.02

    @model_validator(mode="after")
    def _check(self) -> GridConfig:
        if not self.lengths:
            raise ValueError("grid.lengths must be non-empty")
        if any(l <= 0 for l in self.lengths):
            raise ValueError("grid.lengths must be positive")
        if any(not 0.0 <= d <= 1.0 for d in self.depths):
            raise ValueError("grid.depths must lie in [0, 1]")
        if not 0 < self.tolerance < 1:
            raise ValueError("grid.tolerance must lie in (0, 1)")
        return self


class StatsConfig(BaseModel):
    model_config = {"extra": "forbid"}

    ci: float = 0.95
    effective_context_threshold: float = 0.8
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 12345


class JudgeConfig(BaseModel):
    """LLM-judge settings. The judge is a *fallback*, never the default scorer."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    model: str | None = None  # name of a model defined in `models:`
    # Fraction of deterministically-scored trials to also judge, so that
    # judge-vs-deterministic agreement can be reported.
    validation_fraction: float = 0.2
    validation_seed: int = 7
    # Refuse to trust the judge below this agreement rate.
    min_agreement: float = 0.9


class RunConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = "run"
    output_dir: str = "runs"
    n_trials: int = 5
    base_seed: int = 0
    max_concurrency: int = 8
    requests_per_minute: float | None = None
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    request_timeout: float = 600.0
    checkpoint: bool = True
    # full: gzip every prompt to disk; hash: only sha256; none: neither.
    store_prompts: Literal["full", "hash", "none"] = "full"
    # Skip cells whose target length exceeds a model's context_limit.
    skip_over_context_limit: bool = True

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if self.n_trials < 1:
            raise ValueError("run.n_trials must be >= 1")
        return self

    @property
    def seeds(self) -> list[int]:
        return [self.base_seed + i for i in range(self.n_trials)]


class ExperimentConfig(BaseModel):
    model_config = {"extra": "forbid"}

    run: RunConfig = Field(default_factory=RunConfig)
    grid: GridConfig
    haystack: HaystackConfig = Field(default_factory=HaystackConfig)
    models: list[ModelConfig]
    tasks: list[TaskConfig]
    stats: StatsConfig = Field(default_factory=StatsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if not self.models:
            raise ValueError("config.models must be non-empty")
        if not self.tasks:
            raise ValueError("config.tasks must be non-empty")
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        if self.judge.enabled and self.judge.model not in names:
            raise ValueError(
                f"judge.model {self.judge.model!r} is not one of the configured models {names}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top level of a config must be a mapping")
        return cls.model_validate(raw)

    def model_by_name(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)
