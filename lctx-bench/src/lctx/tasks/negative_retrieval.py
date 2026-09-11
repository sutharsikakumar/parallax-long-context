"""negative_retrieval: the queried key is never present; the answer is ABSENT.

This is the false-positive control for the whole suite. A model that scores well
on ``single_needle`` by emitting a plausible-looking code whenever asked will
score zero here, which is exactly the point.
"""

from __future__ import annotations

from .base import Needle, Task, TaskInstance, TaskSpec, register_task
from .primitives import perturb_code, random_code, unique


def _fact(key: str, value: str) -> str:
    return f"The access code for vault {key} is {value}."


class NegativeRetrieval(Task):
    name = "negative_retrieval"
    scorer = "absent_detection"
    description = "Queried key is absent; correct answer is ABSENT. False-positive control."
    answer_is_verbatim = False  # the answer is the ABSENT sentinel
    defaults = {"num_present": 8, "near_miss": True}

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        rng = spec.content_rng()
        prng = spec.placement_rng()
        n_present = max(1, int(p["num_present"]))

        keys: set[str] = set()
        values: set[str] = set()
        # The queried key is drawn first and then never used in any fact.
        missing_key = unique(lambda: random_code(rng), keys)

        present: list[tuple[str, str]] = []
        for i in range(n_present):
            if p["near_miss"] and i == 0:
                k = unique(lambda: perturb_code(rng, missing_key, 1), keys)
            else:
                k = unique(lambda: random_code(rng), keys)
            present.append((k, unique(lambda: random_code(rng), values)))

        # The near-miss sits at the requested depth; the rest spread out.
        depths = [spec.depth] + self.spread_depths(n_present - 1, prng, avoid=spec.depth)
        needles = [
            Needle(_fact(k, v), d, "distractor", f"present:{k}")
            for (k, v), d in zip(present, depths)
        ]

        return TaskInstance(
            spec=spec,
            needles=needles,
            question=f"What is the access code for vault {missing_key}?",
            system=self.build_system("Report the code exactly as it appears."),
            ground_truth="ABSENT",
            scorer=self.scorer,
            structured={
                "missing_key": missing_key,
                "present_keys": [k for k, _ in present],
                # Values a model might copy from the document instead of abstaining.
                "decoy_values": [v for _, v in present],
                "absent": True,
            },
        )

    def make_absent(self, inst: TaskInstance) -> TaskInstance:
        # Already the absent case; the condition is a no-op here.
        return inst

    def oracle(self, inst: TaskInstance) -> str:
        return "ABSENT"


register_task(NegativeRetrieval())
