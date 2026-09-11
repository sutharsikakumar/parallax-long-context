"""aggregation: many small numeric items scattered across the document.

Retrieval of a single span is not enough — the model must attend to every item
in the target batch at once, which is why this task is the sharpest test of
diffuse (rather than peaked) attention in the suite.
"""

from __future__ import annotations

from .base import Needle, Task, TaskInstance, TaskSpec, register_task
from .primitives import random_code, random_number, random_word, unique

_OPS = ("sum", "count", "max")


class Aggregation(Task):
    name = "aggregation"
    scorer = "numeric_tol"
    description = "Sum / count / max over numeric items scattered through the document."
    answer_is_verbatim = False  # the answer is a sum/count/max, not a token
    defaults = {
        "num_items": 12,
        "op": "sum",
        "value_lo": 10,
        "value_hi": 999,
        # Items in other batches, which must be excluded from the aggregate.
        "num_offbatch": 12,
        "rel_tol": 0.0,
        "abs_tol": 0.0,
    }

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        op = p["op"]
        if op not in _OPS:
            raise ValueError(f"aggregation.op must be one of {_OPS}, got {op!r}")
        rng = spec.content_rng()
        prng = spec.placement_rng()

        # num_items is a *task* parameter, fixed independently of target_tokens:
        # growing the context must not silently make the arithmetic harder.
        n_items = max(1, int(p["num_items"]))
        n_off = max(0, int(p["num_offbatch"]))

        tags: set[str] = set()
        target_tag = unique(lambda: random_word(rng), tags)
        sensors: set[str] = set()

        def item(tag: str) -> tuple[str, str, int]:
            sid = unique(lambda: random_code(rng, groups=1, glen=3), sensors)
            val = random_number(rng, int(p["value_lo"]), int(p["value_hi"]))
            return f"The {tag} batch meter {sid} recorded {val} units.", sid, val

        target_items = [item(target_tag) for _ in range(n_items)]
        off_items = []
        for _ in range(n_off):
            other = unique(lambda: random_word(rng), tags)
            off_items.append(item(other))

        # Targets straddle `depth` across a wide band; the aggregate is only
        # recoverable if attention reaches all of them.
        span = 0.8
        lo = max(0.0, min(1.0 - span, spec.depth - span / 2))
        needles = [
            Needle(text, lo + span * i / max(1, n_items - 1), "target", f"item:{sid}")
            for i, (text, sid, _) in enumerate(target_items)
        ]
        for d, (text, sid, _) in zip(self.spread_depths(len(off_items), prng), off_items):
            needles.append(Needle(text, d, "distractor", f"offbatch:{sid}"))

        values = [v for _, _, v in target_items]
        answer = {"sum": sum(values), "count": len(values), "max": max(values)}[op]
        verb = {
            "sum": f"What is the total number of units recorded by all {target_tag} batch meters?",
            "count": f"How many {target_tag} batch meters are reported in the document?",
            "max": f"What is the largest number of units recorded by any {target_tag} batch meter?",
        }[op]

        return TaskInstance(
            spec=spec,
            needles=needles,
            question=verb,
            system=self.build_system(
                f"Consider only meters in the {target_tag} batch. Answer with a single integer."
            ),
            ground_truth=answer,
            scorer=self.scorer,
            scorer_params={"rel_tol": float(p["rel_tol"]), "abs_tol": float(p["abs_tol"])},
            structured={"op": op, "tag": target_tag, "values": values, "answer": answer},
            meta={"num_items": n_items, "op": op},
        )

    def oracle(self, inst: TaskInstance) -> str:
        if inst.structured.get("absent"):
            return "ABSENT"
        return str(inst.structured["answer"])


register_task(Aggregation())
