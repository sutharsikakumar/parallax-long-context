"""multi_needle: k independent key->value facts, all of which must be recovered."""

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


class MultiNeedle(Task):
    name = "multi_needle"
    scorer = "set_f1"
    description = "Retrieve the values of k randomly generated keys. Set-F1."
    defaults = {"n_distractors": None}

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        rng = spec.content_rng()
        prng = spec.placement_rng()
        k = max(1, spec.num_needles)

        keys: set[str] = set()
        values: set[str] = set()
        pairs = [
            (unique(lambda: random_code(rng), keys), unique(lambda: random_code(rng), values))
            for _ in range(k)
        ]

        # Targets straddle `depth`: the requested depth is the centre of a band
        # covering the needles, so a single depth axis still means something when
        # several needles must coexist.
        span = 0.6
        if k == 1:
            depths = [spec.depth]
        else:
            lo = max(0.0, min(1.0 - span, spec.depth - span / 2))
            depths = [lo + span * i / (k - 1) for i in range(k)]

        needles = [
            Needle(_fact(kk, vv), d, "target", f"key:{kk}")
            for (kk, vv), d in zip(pairs, depths)
        ]

        n_d = p["n_distractors"]
        if n_d is None:
            n_d = DEFAULT_DISTRACTOR_COUNT.get(spec.distractor_type, 0)
        distractors = build_distractors(
            spec.distractor_type, rng, _fact, pairs[0][0], pairs[0][1], n_d,
            value_fn=lambda: random_code(rng), key_fn=lambda: random_code(rng),
            seen_keys=keys, seen_values=values,
        )
        for d_depth, (text, label) in zip(
            self.spread_depths(len(distractors), prng), distractors
        ):
            needles.append(Needle(text, d_depth, "distractor", label))

        key_list = ", ".join(kk for kk, _ in pairs)
        return TaskInstance(
            spec=spec,
            needles=needles,
            question=(
                f"What are the access codes for vaults {key_list}? "
                "List all of them, separated by commas."
            ),
            system=self.build_system(
                "Report every requested code exactly as it appears, comma-separated."
            ),
            ground_truth=[vv for _, vv in pairs],
            scorer=self.scorer,
            structured={"pairs": pairs},
        )

    def oracle(self, inst: TaskInstance) -> str:
        if inst.structured.get("absent"):
            return "ABSENT"
        return ", ".join(v for _, v in inst.structured["pairs"])


register_task(MultiNeedle())
