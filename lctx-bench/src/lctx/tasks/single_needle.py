"""single_needle: one random key->value fact buried in filler."""

from __future__ import annotations

from .base import Needle, Task, TaskInstance, TaskSpec, register_task
from .primitives import (
    DEFAULT_DISTRACTOR_COUNT,
    build_distractors,
    random_code,
    unique,
)


def _fact(key: str, value: str) -> str:
    return f"The access code for vault {key} is {value}."


class SingleNeedle(Task):
    name = "single_needle"
    scorer = "exact_match"
    description = "Retrieve the value of one randomly generated key. Exact match."
    defaults = {"n_distractors": None}

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        rng = spec.content_rng()
        prng = spec.placement_rng()

        keys: set[str] = set()
        values: set[str] = set()
        key = unique(lambda: random_code(rng), keys)
        value = unique(lambda: random_code(rng), values)

        needles = [Needle(_fact(key, value), spec.depth, "target", f"key:{key}")]

        n_d = p["n_distractors"]
        if n_d is None:
            n_d = DEFAULT_DISTRACTOR_COUNT.get(spec.distractor_type, 0)
        distractors = build_distractors(
            spec.distractor_type, rng, _fact, key, value, n_d,
            value_fn=lambda: random_code(rng), key_fn=lambda: random_code(rng),
            seen_keys=keys, seen_values=values,
        )
        for d_depth, (text, label) in zip(
            self.spread_depths(len(distractors), prng, avoid=spec.depth), distractors
        ):
            needles.append(Needle(text, d_depth, "distractor", label))

        return TaskInstance(
            spec=spec,
            needles=needles,
            question=f"What is the access code for vault {key}?",
            system=self.build_system("Report the code exactly as it appears."),
            ground_truth=value,
            scorer=self.scorer,
            structured={"key": key, "value": value, "distractors": [d[1] for d in distractors]},
        )

    def oracle(self, inst: TaskInstance) -> str:
        if inst.structured.get("absent"):
            return "ABSENT"
        return str(inst.structured["value"])


register_task(SingleNeedle())
