"""ordering: which item appeared immediately before a given item.

Entries carry no index or timestamp, so the answer is recoverable only from the
relative position of the entries in the document.
"""

from __future__ import annotations

from .base import Needle, Task, TaskInstance, TaskSpec, register_task
from .primitives import random_code, unique


class Ordering(Task):
    name = "ordering"
    scorer = "exact_match"
    description = "Identify the entry immediately preceding a queried entry."
    depth_selects_probe = True  # depth picks which entry is probed
    defaults = {"num_items": 10}

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        rng = spec.content_rng()
        n = max(2, int(p["num_items"]))

        codes: set[str] = set()
        seq = [unique(lambda: random_code(rng), codes) for _ in range(n)]

        # `depth` selects *which* position in the sequence is probed, and the
        # sequence itself spans the document. Index 0 has no predecessor.
        query_idx = int(round(spec.depth * (n - 1)))
        query_idx = min(n - 1, max(1, query_idx))

        lo, hi = 0.03, 0.97
        depths = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        needles = [
            Needle(
                f"The relay log records unit {code}.",
                d,
                "target" if i in (query_idx, query_idx - 1) else "support",
                f"seq:{i}:{code}",
            )
            for i, (code, d) in enumerate(zip(seq, depths))
        ]

        return TaskInstance(
            spec=spec,
            needles=needles,
            question=(
                "The document contains a sequence of relay log entries. "
                f"Which unit is recorded in the entry immediately before unit {seq[query_idx]}?"
            ),
            system=self.build_system(
                "Answer with the identifier of the preceding unit, exactly as it appears."
            ),
            ground_truth=seq[query_idx - 1],
            scorer=self.scorer,
            structured={"sequence": seq, "query_idx": query_idx, "answer": seq[query_idx - 1]},
            meta={"num_items": n, "query_idx": query_idx},
        )

    def make_absent(self, inst: TaskInstance) -> TaskInstance:
        # Remove only the predecessor entry: the queried unit still appears, but
        # nothing precedes it, so ABSENT is the correct answer.
        idx = inst.structured["query_idx"]
        drop = f"seq:{idx - 1}:"
        inst.needles = [n for n in inst.needles if not n.label.startswith(drop)]
        inst.ground_truth = "ABSENT"
        inst.scorer = "absent_detection"
        inst.scorer_params = {}
        inst.structured = {**inst.structured, "absent": True}
        return inst

    def oracle(self, inst: TaskInstance) -> str:
        if inst.structured.get("absent"):
            return "ABSENT"
        return str(inst.structured["answer"])


register_task(Ordering())
