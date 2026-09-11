"""selective_copy: retrieve every item satisfying a predicate, among distractors."""

from __future__ import annotations

from .base import Needle, Task, TaskInstance, TaskSpec, register_task
from .primitives import random_code, unique

_OTHER_STATUSES = ["CLEARED", "PENDING", "ROUTED", "HELD", "ARCHIVED"]


class SelectiveCopy(Task):
    name = "selective_copy"
    scorer = "set_f1"
    description = "Return all items matching a predicate. Set-F1 (partial credit)."
    defaults = {
        "num_matching": 5,
        "num_nonmatching": 15,
        "match_status": "FLAGGED",
    }

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        rng = spec.content_rng()
        prng = spec.placement_rng()
        n_match = max(1, int(p["num_matching"]))
        n_other = max(0, int(p["num_nonmatching"]))
        status = str(p["match_status"])

        codes: set[str] = set()

        def entry(st: str) -> tuple[str, str]:
            c = unique(lambda: random_code(rng), codes)
            return f"The shipment {c} status is {st}.", c

        matching = [entry(status) for _ in range(n_match)]
        nonmatching = [entry(rng.choice(_OTHER_STATUSES)) for _ in range(n_other)]

        span = 0.8
        lo = max(0.0, min(1.0 - span, spec.depth - span / 2))
        needles = [
            Needle(text, lo + span * i / max(1, n_match - 1), "target", f"match:{code}")
            for i, (text, code) in enumerate(matching)
        ]
        for d, (text, code) in zip(self.spread_depths(len(nonmatching), prng), nonmatching):
            needles.append(Needle(text, d, "distractor", f"other:{code}"))

        return TaskInstance(
            spec=spec,
            needles=needles,
            question=(
                f"List the identifiers of every shipment whose status is {status}. "
                "Separate them with commas."
            ),
            system=self.build_system(
                "List every matching identifier and nothing else, comma-separated."
            ),
            ground_truth=[c for _, c in matching],
            scorer=self.scorer,
            structured={"matching": [c for _, c in matching], "status": status},
            meta={"num_matching": n_match, "num_nonmatching": n_other},
        )

    def oracle(self, inst: TaskInstance) -> str:
        if inst.structured.get("absent"):
            return "ABSENT"
        return ", ".join(inst.structured["matching"])


register_task(SelectiveCopy())
