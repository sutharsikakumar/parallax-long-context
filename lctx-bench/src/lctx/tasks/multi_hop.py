"""multi_hop: a chain of assignments that must be followed across positions.

``a = X; b = a; c = b`` — asking for ``c`` requires binding facts that sit far
apart in the document. The links are placed in *shuffled* document order, so the
chain cannot be followed by reading linearly.
"""

from __future__ import annotations

from .base import Needle, Task, TaskInstance, TaskSpec, register_task
from .primitives import (
    DEFAULT_DISTRACTOR_COUNT,
    build_distractors,
    random_code,
    unique,
)


def _root_fact(key: str, value: str) -> str:
    return f"The stored value in register {key} is {value}."


def _link_fact(key: str, src: str) -> str:
    return f"The stored value in register {key} is copied from register {src}."


class MultiHop(Task):
    name = "multi_hop"
    scorer = "exact_match"
    description = "Follow a chain of register assignments scattered across the document."
    defaults = {"chain_length": 3, "n_distractors": None, "shuffle_links": True}

    def _generate(self, spec: TaskSpec) -> TaskInstance:
        p = self.resolve(spec)
        rng = spec.content_rng()
        prng = spec.placement_rng()
        n = max(2, int(p["chain_length"]))

        keys: set[str] = set()
        values: set[str] = set()
        chain = [unique(lambda: random_code(rng, groups=1, glen=4), keys) for _ in range(n)]
        value = unique(lambda: random_code(rng), values)

        # chain[0] holds the literal; chain[i] copies from chain[i-1];
        # the question asks about chain[-1].
        texts = [(_root_fact(chain[0], value), "root")]
        texts += [(_link_fact(chain[i], chain[i - 1]), f"link:{i}") for i in range(1, n)]

        # `depth` positions the value-bearing root; the links spread over the rest.
        link_depths = self.spread_depths(n - 1, prng, avoid=spec.depth)
        if p["shuffle_links"]:
            prng.shuffle(link_depths)
        depths = [spec.depth] + link_depths

        needles = [
            Needle(t, d, "target" if lab == "root" else "support", lab)
            for (t, lab), d in zip(texts, depths)
        ]

        n_d = p["n_distractors"]
        if n_d is None:
            n_d = DEFAULT_DISTRACTOR_COUNT.get(spec.distractor_type, 0)
        distractors = build_distractors(
            spec.distractor_type, rng, _root_fact, chain[0], value, n_d,
            value_fn=lambda: random_code(rng),
            key_fn=lambda: random_code(rng, groups=1, glen=4),
            seen_keys=keys, seen_values=values,
        )
        for d_depth, (text, label) in zip(
            self.spread_depths(len(distractors), prng), distractors
        ):
            needles.append(Needle(text, d_depth, "distractor", label))

        return TaskInstance(
            spec=spec,
            needles=needles,
            question=f"What is the stored value in register {chain[-1]}?",
            system=self.build_system(
                "Registers may copy their value from another register; follow the chain."
            ),
            ground_truth=value,
            scorer=self.scorer,
            structured={"chain": chain, "value": value},
            meta={"chain_length": n},
        )

    def make_absent(self, inst: TaskInstance) -> TaskInstance:
        # Drop the root literal but keep every copy link: the chain is intact yet
        # terminates in nothing, so the value genuinely is not in the document.
        inst.needles = [n for n in inst.needles if n.role != "target"]
        inst.ground_truth = "ABSENT"
        inst.scorer = "absent_detection"
        inst.scorer_params = {}
        inst.structured = {**inst.structured, "absent": True}
        return inst

    def oracle(self, inst: TaskInstance) -> str:
        if inst.structured.get("absent"):
            return "ABSENT"
        return str(inst.structured["value"])


register_task(MultiHop())
