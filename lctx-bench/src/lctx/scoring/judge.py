"""LLM judge — a *fallback* scorer, and only for free-form answers.

The judge is deliberately hard to trust by default. Every judged run also scores
a sample deterministically and reports agreement, so you can decide whether the
judge is measuring the task or measuring itself. If agreement falls below
``judge.min_agreement`` the report says the judge is unreliable and the
deterministic scores remain authoritative.

Reported agreement statistics:

``agreement``
    Raw fraction of sampled trials where judge and deterministic scorer agree.
``cohen_kappa``
    Agreement corrected for chance. Raw agreement is inflated when one label
    dominates — which it does here, since most trials at short context are
    correct — so kappa is the number to read.
``false_positive_rate`` / ``false_negative_rate``
    Where the judge disagrees, in which direction. A judge that is generous
    about near-miss identifiers inflates long-context scores specifically.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models.base import ModelAdapter
from .base import ScoreResult, Scorer, register_scorer
from .extract import canonical, extract_answer

JUDGE_SYSTEM = (
    "You grade answers to a document-retrieval question. "
    "You are given the question, the reference answer, and a candidate answer.\n"
    "Reply with exactly one word: CORRECT if the candidate conveys the same answer "
    "as the reference, otherwise INCORRECT.\n"
    "Formatting, capitalisation, and surrounding words do not matter. "
    "A different identifier, a different number, or a missing item is INCORRECT. "
    "The reference answer ABSENT means the candidate must decline to give a value."
)

JUDGE_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Reference answer:\n{reference}\n\n"
    "Candidate answer:\n{candidate}\n\n"
    "One word, CORRECT or INCORRECT:"
)


class LLMJudge(Scorer):
    """Wraps a :class:`ModelAdapter` as a scorer. Requires an explicit adapter."""

    name = "llm_judge"
    binary = True

    def __init__(self, adapter: ModelAdapter | None = None, max_tokens: int = 8) -> None:
        self.adapter = adapter
        self.max_tokens = max_tokens

    def score(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        """Synchronous scoring, for parity with the deterministic scorers."""
        return asyncio.run(self.ascore(prediction, ground_truth, **params))

    async def ascore(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        if self.adapter is None:
            raise RuntimeError("LLMJudge needs a ModelAdapter; configure judge.model")
        question = params.get("question", "(question not supplied)")
        reference = (
            ", ".join(str(g) for g in ground_truth)
            if isinstance(ground_truth, (list, tuple))
            else str(ground_truth)
        )
        candidate = extract_answer(prediction) or "(empty)"

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_TEMPLATE.format(
                    question=question, reference=reference, candidate=candidate
                ),
            },
        ]
        result = await self.adapter.agenerate(
            messages, max_tokens=self.max_tokens, temperature=0.0
        )
        verdict = canonical(result.text)
        correct = verdict.startswith("correct")
        undecided = not (verdict.startswith("correct") or verdict.startswith("incorrect"))
        return ScoreResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            parsed=candidate,
            detail={"judge_raw": result.text.strip(), "undecided": undecided},
        )


@dataclass
class AgreementReport:
    """Judge-vs-deterministic agreement on a human-labelable sample."""

    n: int
    agreement: float
    cohen_kappa: float
    false_positive_rate: float
    false_negative_rate: float
    n_undecided: int
    min_agreement: float
    #: Sampled trials where the two scorers disagree — the set to hand-label.
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.n > 0 and self.agreement >= self.min_agreement

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "agreement": self.agreement,
            "cohen_kappa": self.cohen_kappa,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "n_undecided": self.n_undecided,
            "min_agreement": self.min_agreement,
            "trustworthy": self.trustworthy,
            "verdict": (
                "judge agrees with the deterministic scorer closely enough to be used "
                "as a fallback for free-form answers"
                if self.trustworthy
                else "judge disagrees too often; treat judge scores as unreliable and "
                "prefer deterministic scoring"
            ),
            "disagreements": self.disagreements[:50],
        }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Chance-corrected agreement between two binary raters."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - expected) < 1e-12:
        # Both raters are constant and identical: agreement carries no information.
        return float("nan")
    return (observed - expected) / (1 - expected)


async def validate_judge(
    records: Sequence[Any],
    judge: LLMJudge,
    fraction: float = 0.2,
    seed: int = 7,
    min_agreement: float = 0.9,
    max_concurrency: int = 4,
) -> AgreementReport:
    """Judge a random sample of deterministically-scored trials and compare.

    Only trials that already have a deterministic score are sampled — the whole
    point is to measure the judge against a scorer that cannot drift.
    """
    usable = [r for r in records if r.ok and r.scorer != "llm_judge" and r.raw_output]
    if not usable:
        return AgreementReport(0, float("nan"), float("nan"), 0.0, 0.0, 0, min_agreement)

    rng = random.Random(seed)
    k = max(1, int(round(len(usable) * fraction)))
    sample = rng.sample(usable, min(k, len(usable)))

    sem = asyncio.Semaphore(max_concurrency)

    async def judge_one(rec: Any) -> tuple[Any, ScoreResult]:
        async with sem:
            sr = await judge.ascore(
                rec.raw_output, rec.ground_truth, question=rec.task_params.get("question", rec.task)
            )
        return rec, sr

    results = await asyncio.gather(*(judge_one(r) for r in sample))

    det = [bool(r.correct) for r, _ in results]
    jud = [bool(sr.correct) for _, sr in results]
    n = len(results)
    agree = sum(1 for x, y in zip(det, jud) if x == y) / n

    fp = sum(1 for d, j in zip(det, jud) if j and not d)
    fn = sum(1 for d, j in zip(det, jud) if d and not j)
    n_neg = sum(1 for d in det if not d) or 1
    n_pos = sum(1 for d in det if d) or 1

    disagreements = [
        {
            "trial_id": r.trial_id,
            "task": r.task,
            "target_tokens": r.target_tokens,
            "ground_truth": r.ground_truth,
            "raw_output": r.raw_output[:400],
            "deterministic": bool(r.correct),
            "judge": bool(sr.correct),
            "judge_raw": sr.detail.get("judge_raw", ""),
        }
        for r, sr in results
        if bool(r.correct) != bool(sr.correct)
    ]

    return AgreementReport(
        n=n,
        agreement=agree,
        cohen_kappa=cohen_kappa(det, jud),
        false_positive_rate=fp / n_neg,
        false_negative_rate=fn / n_pos,
        n_undecided=sum(1 for _, sr in results if sr.detail.get("undecided")),
        min_agreement=min_agreement,
        disagreements=disagreements,
    )


register_scorer(LLMJudge())
