"""Acceptance: token-exact packing (core requirement #3).

Every packed prompt must land within tolerance of its target token count, under
*each model's own tokenizer*, with chat-template overhead included.
"""

from __future__ import annotations

import pytest
from conftest import TASK_CASES

from lctx.haystack.filler import CorpusFiller, RandomTokenFiller
from lctx.haystack.packing import Packer
from lctx.models.tokenizers import WordTokenizer
from lctx.tasks import TaskSpec, get_task

TARGETS = [500, 1_000, 4_000, 16_000]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
TOLERANCE = 0.02


def _tokenizers():
    """At least two genuinely different tokenizers, as the acceptance test requires."""
    toks = [WordTokenizer()]
    try:
        from lctx.models.tokenizers import TiktokenTokenizer

        toks.append(TiktokenTokenizer("cl100k_base"))
        toks.append(TiktokenTokenizer("o200k_base"))
    except ImportError:  # pragma: no cover
        pass
    return toks


TOKENIZERS = _tokenizers()
TOKENIZER_IDS = [t.name for t in TOKENIZERS]


@pytest.mark.parametrize("tok", TOKENIZERS, ids=TOKENIZER_IDS)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("depth", DEPTHS)
def test_token_exactness(tok, target, depth):
    """ACCEPTANCE: packed prompts hit target_tokens within tolerance."""
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    task = get_task("single_needle")
    inst = task.generate(TaskSpec(task="single_needle", seed=1, target_tokens=target,
                                  depth=depth))
    packed = packer.pack(inst, target, "standard", seed=1)

    assert packed.within_tolerance, (
        f"{tok.name} target={target} depth={depth}: got {packed.actual_tokens} "
        f"({packed.error_frac:+.2%})"
    )
    # Re-measure independently of the packer's own bookkeeping.
    assert tok.count_message_tokens(packed.messages) == packed.actual_tokens
    assert abs(packed.actual_tokens - target) <= TOLERANCE * target


def test_two_tokenizers_are_actually_different():
    """The exactness test is only meaningful if the tokenizers disagree."""
    assert len(TOKENIZERS) >= 2, "need at least two tokenizers for this acceptance test"
    text = "The access code for vault K7QF-2M9X is 4TPL-9WD2."
    counts = {t.name: t.count_tokens(text) for t in TOKENIZERS}
    assert len(set(counts.values())) > 1, f"tokenizers agree exactly: {counts}"


@pytest.mark.parametrize("name,params", TASK_CASES)
def test_every_task_packs_to_target(name, params):
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    task = get_task(name)
    for target in (1_000, 8_000):
        inst = task.generate(TaskSpec(task=name, seed=2, target_tokens=target, depth=0.5,
                                      num_needles=3, params=dict(params)))
        packed = packer.pack(inst, target, "standard", seed=2)
        assert packed.within_tolerance, (
            f"{name} target={target}: {packed.actual_tokens} ({packed.error_frac:+.2%})"
        )


@pytest.mark.parametrize("filler", [CorpusFiller(), RandomTokenFiller(vocab_size=256)])
def test_both_filler_sources_pack_exactly(filler):
    tok = WordTokenizer()
    packer = Packer(tok, filler, tolerance=TOLERANCE)
    inst = get_task("single_needle").generate(
        TaskSpec(task="single_needle", seed=4, target_tokens=4000, depth=0.5)
    )
    packed = packer.pack(inst, 4000, "standard", seed=4)
    assert packed.within_tolerance


def test_chat_overhead_is_measured_and_nonzero():
    """Overhead must be counted in the total, not assumed away."""
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    inst = get_task("single_needle").generate(
        TaskSpec(task="single_needle", seed=5, target_tokens=2000, depth=0.5)
    )
    packed = packer.pack(inst, 2000, "standard", seed=5)
    assert packed.chat_overhead_tokens > 0
    content = sum(tok.count_tokens(m["content"]) for m in packed.messages)
    assert packed.actual_tokens == content + packed.chat_overhead_tokens


def test_needle_lands_near_the_requested_depth():
    """Realized depth must track the requested depth, and is always recorded."""
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    target = 32_000
    realized = []
    for depth in DEPTHS:
        inst = get_task("single_needle").generate(
            TaskSpec(task="single_needle", seed=6, target_tokens=target, depth=depth)
        )
        packed = packer.pack(inst, target, "standard", seed=6)
        realized.append(packed.needle_depths_actual[0])
        assert abs(packed.needle_depths_actual[0] - depth) < 0.05
    assert realized == sorted(realized), "depth ordering was not preserved"


def test_no_haystack_condition_has_no_filler():
    """The ceiling control is the task alone; the length target does not apply."""
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    inst = get_task("single_needle").generate(
        TaskSpec(task="single_needle", seed=7, target_tokens=16_000, depth=0.5,
                 condition="no_haystack")
    )
    packed = packer.pack(inst, 16_000, "no_haystack", seed=7)
    assert packed.filler_tokens == 0
    assert packed.actual_tokens < 200
    assert packed.tolerance_checked is False  # target length is not applicable


def test_shuffled_haystack_changes_filler_but_not_the_task():
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    inst = get_task("single_needle").generate(
        TaskSpec(task="single_needle", seed=8, target_tokens=4000, depth=0.5)
    )
    normal = packer.pack(inst, 4000, "standard", seed=8)
    shuffled = packer.pack(inst, 4000, "shuffled_haystack", seed=8)
    assert normal.messages[1]["content"] != shuffled.messages[1]["content"]
    assert shuffled.within_tolerance
    # The needle itself is untouched by the control.
    assert inst.needles[0].text in shuffled.messages[1]["content"]


def test_packing_is_deterministic():
    """CORE REQUIREMENT #6: the same seed must rebuild the identical prompt."""
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    inst = get_task("single_needle").generate(
        TaskSpec(task="single_needle", seed=9, target_tokens=4000, depth=0.5)
    )
    a = packer.pack(inst, 4000, "standard", seed=9)
    b = Packer(WordTokenizer(), CorpusFiller(), tolerance=TOLERANCE).pack(
        inst, 4000, "standard", seed=9
    )
    assert a.messages == b.messages
    assert a.actual_tokens == b.actual_tokens


def test_needle_appears_exactly_once():
    """A needle duplicated by a packing bug would make retrieval trivially easier."""
    tok = WordTokenizer()
    packer = Packer(tok, CorpusFiller(), tolerance=TOLERANCE)
    inst = get_task("single_needle").generate(
        TaskSpec(task="single_needle", seed=10, target_tokens=8000, depth=0.3)
    )
    packed = packer.pack(inst, 8000, "standard", seed=10)
    assert packed.messages[1]["content"].count(inst.needles[0].text) == 1
