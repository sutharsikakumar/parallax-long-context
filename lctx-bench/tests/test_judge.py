"""The LLM judge must be validated before it is trusted."""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

from lctx.models.base import GenerationResult, ModelAdapter, Usage
from lctx.models.tokenizers import WordTokenizer
from lctx.runner.records import TrialRecord
from lctx.scoring.judge import LLMJudge, cohen_kappa, validate_judge


class ScriptedJudge(ModelAdapter):
    """A judge whose verdicts are fixed in advance, so agreement is computable."""

    is_simulator = True

    def __init__(self, verdicts: dict[str, str], default: str = "CORRECT") -> None:
        super().__init__("scripted", WordTokenizer())
        self.verdicts = verdicts
        self.default = default
        self.calls = 0

    async def agenerate(self, messages, max_tokens=256, temperature=0.0, stop=None,
                        trial=None) -> GenerationResult:
        self.calls += 1
        content = messages[-1]["content"]
        for needle, verdict in self.verdicts.items():
            if needle in content:
                return GenerationResult(text=verdict, usage=Usage(1, 1))
        return GenerationResult(text=self.default, usage=Usage(1, 1))


def _rec(i: int, correct: bool, output: str) -> TrialRecord:
    return TrialRecord(
        trial_id=f"t{i}", cell_id="c", run_name="r", model="m", task="single_needle",
        raw_output=output, ground_truth="GOLD", correct=correct,
        score=float(correct), scorer="exact_match",
    )


def test_judge_parses_verdicts():
    judge = LLMJudge(ScriptedJudge({}, default="CORRECT"))
    assert judge.score("<answer>X</answer>", "X").correct
    judge = LLMJudge(ScriptedJudge({}, default="INCORRECT"))
    assert not judge.score("<answer>X</answer>", "X").correct


def test_judge_flags_an_unparseable_verdict():
    judge = LLMJudge(ScriptedJudge({}, default="I'm not sure about this one"))
    result = judge.score("<answer>X</answer>", "X")
    assert result.detail["undecided"] and not result.correct


def test_judge_requires_an_adapter():
    with pytest.raises(RuntimeError, match="needs a ModelAdapter"):
        LLMJudge(None).score("x", "y")


def test_perfect_agreement_is_trustworthy():
    records = [_rec(i, True, "<answer>GOLD</answer>") for i in range(10)]
    judge = LLMJudge(ScriptedJudge({}, default="CORRECT"))
    report = asyncio.run(validate_judge(records, judge, fraction=1.0, min_agreement=0.9))
    assert report.n == 10
    assert report.agreement == pytest.approx(1.0)
    assert report.trustworthy
    assert "closely enough" in report.as_dict()["verdict"]


def test_a_lenient_judge_is_rejected():
    """A judge that calls every wrong answer correct must not be trusted."""
    records = [_rec(i, False, "<answer>WRONG</answer>") for i in range(10)]
    judge = LLMJudge(ScriptedJudge({}, default="CORRECT"))
    report = asyncio.run(validate_judge(records, judge, fraction=1.0, min_agreement=0.9))
    assert report.agreement == pytest.approx(0.0)
    assert not report.trustworthy
    assert report.false_positive_rate == pytest.approx(1.0)
    assert "unreliable" in report.as_dict()["verdict"]


def test_disagreements_are_returned_for_hand_labelling():
    """The point of validation is to produce a reviewable list, not just a number."""
    records = (
        [_rec(i, True, "<answer>GOLD</answer>") for i in range(8)]
        + [_rec(100 + i, False, "<answer>NEARMISS</answer>") for i in range(2)]
    )
    judge = LLMJudge(ScriptedJudge({}, default="CORRECT"))
    report = asyncio.run(validate_judge(records, judge, fraction=1.0, min_agreement=0.9))
    assert report.agreement == pytest.approx(0.8)
    assert len(report.disagreements) == 2
    for d in report.disagreements:
        assert d["deterministic"] is False and d["judge"] is True
        assert "NEARMISS" in d["raw_output"]


def test_validation_samples_a_fraction_and_is_reproducible():
    records = [_rec(i, i % 2 == 0, "<answer>GOLD</answer>") for i in range(100)]
    judge = LLMJudge(ScriptedJudge({}, default="CORRECT"))
    a = asyncio.run(validate_judge(records, judge, fraction=0.2, seed=7))
    b = asyncio.run(validate_judge(records, judge, fraction=0.2, seed=7))
    assert a.n == b.n == 20
    assert a.agreement == b.agreement


def test_validation_handles_no_scoreable_records():
    judge = LLMJudge(ScriptedJudge({}))
    report = asyncio.run(validate_judge([], judge))
    assert report.n == 0 and not report.trustworthy


def test_cohen_kappa_corrects_for_chance():
    """Raw agreement is inflated when one label dominates; kappa is not."""
    # 90% agreement, but the judge is simply always saying "correct".
    det = [True] * 9 + [False]
    jud = [True] * 10
    assert sum(d == j for d, j in zip(det, jud)) / 10 == pytest.approx(0.9)
    assert cohen_kappa(det, jud) == pytest.approx(0.0, abs=1e-9)

    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    import math

    assert math.isnan(cohen_kappa([True] * 5, [True] * 5))  # no information


def test_judge_is_never_the_default_scorer():
    """Deterministic scorers must remain the default for every shipped task."""
    from lctx.tasks import all_tasks

    assert all(t.scorer != "llm_judge" for t in all_tasks().values())
