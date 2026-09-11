"""Deterministic scorers. These are the default; the LLM judge is a fallback."""

from __future__ import annotations

from typing import Any

from .base import ScoreResult, Scorer, register_scorer
from .extract import canonical, extract_answer, extract_number, is_absent, split_items


class ExactMatch(Scorer):
    name = "exact_match"
    binary = True

    def score(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        pred = extract_answer(prediction)
        c_pred, c_gold = canonical(pred), canonical(ground_truth)
        ok = bool(c_gold) and c_pred == c_gold
        return ScoreResult(
            score=1.0 if ok else 0.0,
            correct=ok,
            parsed=pred,
            detail={
                "canonical_pred": c_pred,
                "canonical_gold": c_gold,
                # A wrong-but-abstaining answer is a different failure from a
                # wrong-but-confident one, and worth separating in the logs.
                "abstained": is_absent(pred),
            },
        )


class SetF1(Scorer):
    name = "set_f1"
    binary = False

    def score(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        pred_items = split_items(extract_answer(prediction))
        gold_items = [canonical(g) for g in (ground_truth or []) if canonical(g)]
        gold = set(gold_items)
        pred = set(pred_items)

        if not gold:
            # Degenerate gold set: only an empty prediction is right.
            ok = not pred
            return ScoreResult(1.0 if ok else 0.0, ok, pred_items,
                               {"precision": float(ok), "recall": float(ok)})

        tp = len(pred & gold)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return ScoreResult(
            score=f1,
            correct=f1 >= 1.0,
            parsed=pred_items,
            detail={
                "precision": precision,
                "recall": recall,
                "true_positives": tp,
                "n_pred": len(pred),
                "n_gold": len(gold),
                "missed": sorted(gold - pred),
                "spurious": sorted(pred - gold),
            },
        )


class NumericTolerance(Scorer):
    name = "numeric_tol"
    binary = True

    def score(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        rel_tol = float(params.get("rel_tol", 0.0))
        abs_tol = float(params.get("abs_tol", 0.0))
        pred = extract_answer(prediction)
        value = extract_number(pred)
        try:
            gold = float(ground_truth)
        except (TypeError, ValueError):
            return ScoreResult(0.0, False, None, {"error": "non-numeric ground truth"})

        if value is None:
            return ScoreResult(0.0, False, None,
                               {"error": "no number in prediction", "abstained": is_absent(pred)})
        err = abs(value - gold)
        tol = max(abs_tol, rel_tol * abs(gold))
        ok = err <= tol
        return ScoreResult(
            score=1.0 if ok else 0.0,
            correct=ok,
            parsed=value,
            detail={
                "abs_error": err,
                "rel_error": err / abs(gold) if gold else float("inf"),
                "tolerance": tol,
            },
        )


class AbsentDetection(Scorer):
    name = "absent_detection"
    binary = True

    def score(self, prediction: str, ground_truth: Any, **params: Any) -> ScoreResult:
        """Credit abstention, and record *what* a false positive copied.

        ``decoys`` (optional) lists values present in the document. If the model
        produced one of them, it did not invent a value — it retrieved the wrong
        one, which is a distinguishable failure mode.
        """
        pred = extract_answer(prediction)
        abstained = is_absent(pred)
        decoys = {canonical(d) for d in params.get("decoys", []) or []}
        c_pred = canonical(pred)
        return ScoreResult(
            score=1.0 if abstained else 0.0,
            correct=abstained,
            parsed=pred,
            detail={
                "abstained": abstained,
                "copied_decoy": (not abstained) and c_pred in decoys,
                "fabricated": (not abstained) and bool(c_pred) and c_pred not in decoys,
            },
        )


register_scorer(ExactMatch())
register_scorer(SetF1())
register_scorer(NumericTolerance())
register_scorer(AbsentDetection())
