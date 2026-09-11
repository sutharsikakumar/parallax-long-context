"""Trial records: the unit of persistence, checkpointing, and analysis."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


def cell_id(
    model: str, task: str, condition: str, distractor_type: str,
    target_tokens: int, depth: float,
) -> str:
    """Stable identifier for a grid cell (all seeds pooled)."""
    key = f"{model}|{task}|{condition}|{distractor_type}|{target_tokens}|{depth:.6f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def trial_id(
    model: str, task: str, condition: str, distractor_type: str,
    target_tokens: int, depth: float, seed: int, num_needles: int, params: dict[str, Any],
) -> str:
    """Stable identifier for one trial.

    Checkpoint resume is keyed on this, so it must cover everything that changes
    what is sent to the model — including task params.
    """
    key = "|".join([
        model, task, condition, distractor_type, str(target_tokens),
        f"{depth:.6f}", str(seed), str(num_needles),
        json.dumps(params, sort_keys=True, default=str),
    ])
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class TrialRecord(BaseModel):
    """One (model, task, condition, length, depth, seed) observation."""

    model_config = {"protected_namespaces": ()}

    trial_id: str
    cell_id: str
    run_name: str

    # -- grid coordinates
    model: str
    task: str
    condition: str = "standard"
    distractor_type: str = "none"
    target_tokens: int = 0
    depth: float = 0.5
    seed: int = 0
    num_needles: int = 1
    task_params: dict[str, Any] = Field(default_factory=dict)

    # -- packing measurements (core requirement #3)
    actual_tokens: int = 0
    filler_tokens: int = 0
    chat_overhead_tokens: int = 0
    needle_positions: list[int] = Field(default_factory=list)
    needle_depths_requested: list[float] = Field(default_factory=list)
    needle_depths_actual: list[float] = Field(default_factory=list)
    within_tolerance: bool = True
    tolerance_checked: bool = True
    pack_iterations: int = 0

    # -- model interaction (core requirement #6)
    prompt_sha256: str = ""
    prompt_path: str | None = None
    raw_output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    attempts: int = 1
    error: str | None = None

    # -- scoring
    scorer: str = ""
    score: float = 0.0
    correct: bool = False
    binary_scorer: bool = True
    parsed_answer: Any = None
    ground_truth: Any = None
    score_detail: dict[str, Any] = Field(default_factory=dict)

    # -- optional judge cross-check
    judge_score: float | None = None
    judge_agrees: bool | None = None

    @property
    def ok(self) -> bool:
        """True when the trial produced a usable observation."""
        return self.error is None


def write_jsonl(path: Path, records: list[TrialRecord], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[TrialRecord]:
    if not path.exists():
        return []
    out: list[TrialRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(TrialRecord.model_validate_json(line))
            except Exception:
                # A partially written final line is expected after an interrupted
                # run; skip it rather than losing the whole checkpoint.
                continue
    return out


def iter_jsonl(path: Path) -> Iterator[TrialRecord]:
    yield from read_jsonl(path)


def store_prompt(dirpath: Path, tid: str, messages: list[dict[str, str]]) -> tuple[str, str]:
    """Gzip a prompt to disk; return ``(sha256, relative_path)``."""
    payload = json.dumps(messages, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    dirpath.mkdir(parents=True, exist_ok=True)
    rel = f"{tid}.json.gz"
    with gzip.open(dirpath / rel, "wt", encoding="utf-8") as fh:
        fh.write(payload)
    return digest, rel


def prompt_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False).encode()
    ).hexdigest()
