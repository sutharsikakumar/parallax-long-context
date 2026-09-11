"""Acceptance: scorer unit tests, including the edge cases that bite in practice."""

from __future__ import annotations

import pytest

from lctx.scoring import get_scorer
from lctx.scoring.extract import (
    canonical,
    extract_answer,
    extract_number,
    is_absent,
    normalize,
    split_items,
)

EM = get_scorer("exact_match")
F1 = get_scorer("set_f1")
NUM = get_scorer("numeric_tol")
ABS = get_scorer("absent_detection")


# -- extraction -----------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<answer>K7QF-2M9X</answer>", "K7QF-2M9X"),
        ("<answer>  K7QF-2M9X  </answer>", "K7QF-2M9X"),
        ("noise <answer>A</answer> more <answer>B</answer>", "B"),  # last tag wins
        ("<answer>unterminated", "unterminated"),
        ("The answer is: **K7QF-2M9X**.", "**K7QF-2M9X**."),
        ("Answer: 42", "42"),
        ("<think>scratch work</think><answer>X1</answer>", "X1"),
        ("", ""),
        ("   \n  ", ""),
        ("just the value", "just the value"),
    ],
)
def test_extract_answer(raw, expected):
    assert extract_answer(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("K7QF-2M9X", "k7qf2m9x"),
        ("  k7qf 2m9x  ", "k7qf2m9x"),
        ('"K7QF-2M9X."', "k7qf2m9x"),
        ("**K7QF-2M9X**", "k7qf2m9x"),
        ("1,234", "1234"),
    ],
)
def test_canonical_ignores_presentation_only(raw, expected):
    assert canonical(raw) == expected


def test_normalize_preserves_internal_structure():
    assert normalize("  The  Quick   Brown  ") == "the quick brown"
    assert normalize("value.") == "value"


@pytest.mark.parametrize(
    "raw,expected",
    [("ABSENT", True), ("absent", True), ("not present", True), ("Not found.", True),
     ("none", True), ("", True), ("K7QF", False), ("0", False), ("42", False)],
)
def test_is_absent(raw, expected):
    assert is_absent(raw) is expected


def test_zero_is_a_value_not_an_abstention():
    """A numeric answer of 0 must never be read as 'no answer'."""
    assert not is_absent("0")
    assert NUM.score("<answer>0</answer>", 0).correct


# -- exact match ----------------------------------------------------------

@pytest.mark.parametrize(
    "pred,gold,ok",
    [
        ("<answer>K7QF-2M9X</answer>", "K7QF-2M9X", True),
        ("<answer>k7qf-2m9x</answer>", "K7QF-2M9X", True),      # casing
        ("<answer> K7QF-2M9X </answer>", "K7QF-2M9X", True),    # whitespace
        ("<answer>K7QF2M9X</answer>", "K7QF-2M9X", True),       # punctuation
        ("<answer>K7QF-2M9Y</answer>", "K7QF-2M9X", False),     # one char off
        ("<answer>ABSENT</answer>", "K7QF-2M9X", False),
        ("", "K7QF-2M9X", False),
    ],
)
def test_exact_match(pred, gold, ok):
    assert EM.score(pred, gold).correct is ok


def test_exact_match_flags_abstention_separately():
    """A wrong answer that abstains is a different failure from a confident one."""
    assert EM.score("<answer>ABSENT</answer>", "X1").detail["abstained"]
    assert not EM.score("<answer>Y2</answer>", "X1").detail["abstained"]


# -- set F1 ---------------------------------------------------------------

def test_set_f1_exact():
    r = F1.score("<answer>A1, B2, C3</answer>", ["A1", "B2", "C3"])
    assert r.score == pytest.approx(1.0) and r.correct


def test_set_f1_partial_credit():
    r = F1.score("<answer>A1, B2</answer>", ["A1", "B2", "C3"])
    assert r.score == pytest.approx(0.8)           # P=1, R=2/3
    assert not r.correct                            # partial credit is not "correct"
    assert r.detail["missed"] == ["c3"]


def test_set_f1_penalises_spurious_items():
    r = F1.score("<answer>A1, B2, C3, D4</answer>", ["A1", "B2", "C3"])
    assert r.score == pytest.approx(6 / 7)
    assert r.detail["spurious"] == ["d4"]


@pytest.mark.parametrize(
    "pred", ["<answer>A1\nB2\nC3</answer>", "<answer>- A1\n- B2\n- C3</answer>",
             "<answer>A1; B2; C3</answer>", "<answer>A1, B2 and C3</answer>",
             "<answer>1. A1 2. B2 3. C3</answer>"],
)
def test_set_f1_accepts_any_list_formatting(pred):
    assert F1.score(pred, ["A1", "B2", "C3"]).score == pytest.approx(1.0)


def test_set_f1_deduplicates_repeats():
    """Repeating a correct item must not inflate precision."""
    r = F1.score("<answer>A1, A1, A1</answer>", ["A1", "B2"])
    assert r.detail["n_pred"] == 1
    assert r.score == pytest.approx(2 / 3)


def test_set_f1_empty_prediction():
    r = F1.score("<answer>ABSENT</answer>", ["A1", "B2"])
    assert r.score == 0.0 and not r.correct


def test_set_f1_empty_gold_requires_empty_prediction():
    assert F1.score("<answer>ABSENT</answer>", []).correct
    assert not F1.score("<answer>A1</answer>", []).correct


# -- numeric tolerance ----------------------------------------------------

@pytest.mark.parametrize(
    "pred,gold,kw,ok",
    [
        ("<answer>1747</answer>", 1747, {}, True),
        ("<answer>1,747</answer>", 1747, {}, True),          # thousands separator
        ("<answer>1747 units</answer>", 1747, {}, True),     # trailing unit
        ("<answer>$1747</answer>", 1747, {}, True),
        ("<answer>1748</answer>", 1747, {}, False),          # exact by default
        ("<answer>1748</answer>", 1747, {"abs_tol": 1}, True),
        ("<answer>1760</answer>", 1747, {"rel_tol": 0.01}, True),
        ("<answer>1800</answer>", 1747, {"rel_tol": 0.01}, False),
        ("<answer>-5</answer>", -5, {}, True),
        ("<answer>3.5</answer>", 3.5, {}, True),
        ("<answer>no idea</answer>", 1747, {}, False),
    ],
)
def test_numeric_tolerance(pred, gold, kw, ok):
    assert NUM.score(pred, gold, **kw).correct is ok


def test_numeric_missing_number_is_reported_not_crashed():
    r = NUM.score("<answer>ABSENT</answer>", 42)
    assert not r.correct and r.detail["error"] == "no number in prediction"


def test_extract_number_takes_the_first():
    assert extract_number("between 10 and 20") == 10.0
    assert extract_number("no digits here") is None


# -- absent detection -----------------------------------------------------

@pytest.mark.parametrize(
    "pred,ok", [("<answer>ABSENT</answer>", True), ("<answer>not present</answer>", True),
                ("<answer>none</answer>", True), ("<answer>XY-12</answer>", False)],
)
def test_absent_detection(pred, ok):
    assert ABS.score(pred, "ABSENT").correct is ok


def test_absent_detection_distinguishes_copying_from_fabricating():
    """Which way a model fails matters: copied a decoy, or invented a value."""
    copied = ABS.score("<answer>XY-12</answer>", "ABSENT", decoys=["XY-12"])
    assert copied.detail["copied_decoy"] and not copied.detail["fabricated"]

    made_up = ABS.score("<answer>ZZ-99</answer>", "ABSENT", decoys=["XY-12"])
    assert made_up.detail["fabricated"] and not made_up.detail["copied_decoy"]


def test_scorers_never_raise_on_junk_output():
    """A malformed completion must score zero, not abort a sweep."""
    junk = ["", "   ", "<answer></answer>", "\x00\x01", "a" * 5000, "<answer>" * 50]
    for bad in junk:
        assert EM.score(bad, "X1").score == 0.0
        assert F1.score(bad, ["X1"]).score == 0.0
        assert NUM.score(bad, 1).score == 0.0
        assert ABS.score(bad, "ABSENT").score in (0.0, 1.0)


def test_binary_flags_are_declared_correctly():
    assert EM.binary and NUM.binary and ABS.binary
    assert not F1.binary
