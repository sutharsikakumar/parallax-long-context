"""Task contract.

Every task ships three things that must agree with each other:

  1. a **generator** that emits needles, a question, and structured ground truth;
  2. an **oracle** that answers from the structured data alone;
  3. a **scorer** name under which the oracle's answer scores 1.0.

The acceptance test ``test_generators.py::test_oracle_solves_every_task`` runs
(2) through (3) for every task and every parameter combination, so a generator
that produces an unsolvable or mis-keyed instance fails CI.

Invariant (core requirement #1 — length is decoupled from difficulty):
``TaskSpec.target_tokens`` is carried for logging only. A generator MUST NOT let
it influence the needles, the question, or the ground truth; growing the context
may only add filler. ``test_generators.py::test_content_independent_of_length``
enforces this for every task.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

NeedleRole = Literal["target", "distractor", "support"]

ANSWER_TAG_INSTRUCTION = (
    "Reply with the answer only, wrapped in <answer></answer> tags, with no explanation.\n"
    "If the requested information is not present in the document, "
    "reply with exactly <answer>ABSENT</answer>."
)

ABSENT = "ABSENT"


@dataclass(frozen=True)
class TaskSpec:
    """One grid cell's task parameters."""

    task: str
    seed: int
    #: Carried for logging only. Generators must not condition content on it.
    target_tokens: int = 0
    depth: float = 0.5
    num_needles: int = 1
    distractor_type: str = "none"
    condition: str = "standard"
    params: dict[str, Any] = field(default_factory=dict)

    def content_key(self) -> str:
        """Identity of the *content* of an instance.

        Deliberately excludes ``target_tokens`` and ``depth``: the same seed must
        yield the same facts whether they sit at 10k tokens or 200k, at the top of
        the document or the bottom. Length and position are placement variables,
        never content variables.
        """
        payload = "|".join(
            [
                self.task,
                str(self.seed),
                str(self.num_needles),
                self.distractor_type,
                # needle_absent changes the facts; the packing-only conditions do not.
                "absent" if self.condition == "needle_absent" else "present",
                repr(sorted(self.params.items())),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def content_rng(self) -> random.Random:
        return random.Random(int(self.content_key()[:16], 16))

    def placement_rng(self) -> random.Random:
        """Separate stream for placement, so jitter never perturbs the facts."""
        seed = hashlib.sha256(f"place|{self.content_key()}|{self.depth}".encode()).hexdigest()
        return random.Random(int(seed[:16], 16))


@dataclass
class Needle:
    """A snippet to embed in the haystack at a relative position."""

    text: str
    depth: float
    role: NeedleRole = "target"
    label: str = ""

    def __post_init__(self) -> None:
        self.depth = min(1.0, max(0.0, float(self.depth)))


@dataclass
class TaskInstance:
    """A fully-specified, length-agnostic evaluation item."""

    spec: TaskSpec
    needles: list[Needle]
    question: str
    system: str
    ground_truth: Any
    scorer: str
    scorer_params: dict[str, Any] = field(default_factory=dict)
    #: Structured data the oracle reads. Never shown to the model.
    structured: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def target_needles(self) -> list[Needle]:
        return [n for n in self.needles if n.role == "target"]

    def sorted_needles(self) -> list[Needle]:
        return sorted(self.needles, key=lambda n: n.depth)


class Task(ABC):
    """Base class for task generators."""

    #: Registry key, also the value of ``TaskConfig.name``.
    name: str = "base"
    #: Scorer used in the normal (needle-present) case.
    scorer: str = "exact_match"
    #: Parameters a config may override, with defaults.
    defaults: dict[str, Any] = {}
    #: Human-readable description surfaced by ``lctx tasks``.
    description: str = ""
    #: True when the answer appears verbatim in a needle (a retrieved token).
    #: False when it is *computed* from the needles (a sum) or is a sentinel
    #: (ABSENT) — such answers cannot be found by string search in the document,
    #: which changes what leakage checks and judges can assume.
    answer_is_verbatim: bool = True
    #: True when ``depth`` selects *which item* is queried rather than only where
    #: the needle sits. For such tasks the needle set is depth-invariant but the
    #: question and ground truth are not, by design.
    depth_selects_probe: bool = False

    def resolve(self, spec: TaskSpec) -> dict[str, Any]:
        merged = dict(self.defaults)
        unknown = set(spec.params) - set(merged)
        if unknown:
            raise ValueError(
                f"task {self.name!r} got unknown params {sorted(unknown)}; "
                f"known params: {sorted(merged)}"
            )
        merged.update(spec.params)
        return merged

    def generate(self, spec: TaskSpec) -> TaskInstance:
        """Build an instance, applying the shared ``needle_absent`` transformation."""
        inst = self._generate(spec)
        if spec.condition == "needle_absent":
            inst = self.make_absent(inst)
        self._validate(inst)
        return inst

    @abstractmethod
    def _generate(self, spec: TaskSpec) -> TaskInstance:
        """Task-specific generation, always in the needle-present form."""

    @abstractmethod
    def oracle(self, inst: TaskInstance) -> str:
        """Answer the instance from ``inst.structured`` alone."""

    def make_absent(self, inst: TaskInstance) -> TaskInstance:
        """Control condition: strip the target needles; the answer becomes ABSENT.

        Distractors stay, so the document still looks the same shape — only the
        evidence is gone. Tasks whose absent form needs more care override this.
        """
        inst.needles = [n for n in inst.needles if n.role != "target"]
        inst.ground_truth = ABSENT
        inst.scorer = "absent_detection"
        inst.scorer_params = {}
        inst.structured = {**inst.structured, "absent": True}
        return inst

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _validate(inst: TaskInstance) -> None:
        if not inst.question.strip():
            raise ValueError("task produced an empty question")
        for n in inst.needles:
            if not n.text.strip():
                raise ValueError("task produced an empty needle")

    @staticmethod
    def build_system(task_hint: str = "") -> str:
        base = (
            "You are given a long document, followed by a question about it. "
            "Answer using only information contained in the document.\n"
        )
        if task_hint:
            base += task_hint.rstrip() + "\n"
        return base + ANSWER_TAG_INSTRUCTION

    @staticmethod
    def spread_depths(n: int, rng: random.Random, avoid: float | None = None,
                      margin: float = 0.03) -> list[float]:
        """``n`` distinct depths spread across the document, optionally avoiding one."""
        if n <= 0:
            return []
        out: list[float] = []
        for i in range(n):
            d = (i + 0.5) / n
            d = min(1.0, max(0.0, d + rng.uniform(-0.02, 0.02)))
            if avoid is not None and abs(d - avoid) < margin:
                d = min(1.0, max(0.0, d + (margin * 2 if d <= avoid else -margin * 2)))
            out.append(d)
        return out


# -- registry -------------------------------------------------------------

_REGISTRY: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    if task.name in _REGISTRY:
        raise ValueError(f"duplicate task registration: {task.name!r}")
    _REGISTRY[task.name] = task
    return task


def get_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_tasks() -> dict[str, Task]:
    return dict(_REGISTRY)
