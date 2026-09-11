"""Scoring. Importing this package registers every built-in scorer."""

from .base import ScoreResult, Scorer, all_scorers, get_scorer, register_scorer  # noqa: F401
from .deterministic import (  # noqa: F401
    AbsentDetection,
    ExactMatch,
    NumericTolerance,
    SetF1,
)
from .extract import canonical, extract_answer, extract_number, is_absent, split_items  # noqa: F401
from .judge import AgreementReport, LLMJudge, validate_judge  # noqa: F401

__all__ = [
    "ScoreResult", "Scorer", "all_scorers", "get_scorer", "register_scorer",
    "AbsentDetection", "ExactMatch", "NumericTolerance", "SetF1",
    "canonical", "extract_answer", "extract_number", "is_absent", "split_items",
    "AgreementReport", "LLMJudge", "validate_judge",
]
