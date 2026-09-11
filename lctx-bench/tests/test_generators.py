"""Acceptance: every generator is solvable, and length never changes the task."""

from __future__ import annotations

import pytest
from conftest import CONDITIONS, DISTRACTORS, TASK_CASES

from lctx.haystack.packing import Packer
from lctx.scoring import get_scorer
from lctx.tasks import TaskSpec, all_tasks, get_task

SEEDS = [0, 1, 2, 7, 42]
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _spec(name: str, params: dict, **kw) -> TaskSpec:
    return TaskSpec(task=name, params=dict(params), **kw)


@pytest.mark.parametrize("name,params", TASK_CASES)
@pytest.mark.parametrize("distractor", DISTRACTORS)
@pytest.mark.parametrize("condition", ["standard", "needle_absent"])
def test_oracle_solves_every_task(name, params, distractor, condition):
    """ACCEPTANCE: a trivial oracle over the structured data scores 1.0.

    This is the guard against a generator that emits an unanswerable item, keys
    the ground truth to the wrong field, or pairs a task with a scorer that
    cannot express its answer.
    """
    task = get_task(name)
    scorer = None
    for seed in SEEDS:
        for depth in DEPTHS:
            inst = task.generate(
                _spec(name, params, seed=seed, depth=depth, num_needles=3,
                      distractor_type=distractor, condition=condition)
            )
            answer = task.oracle(inst)
            scorer = get_scorer(inst.scorer)
            result = scorer.score(f"<answer>{answer}</answer>", inst.ground_truth,
                                  **inst.scorer_params)
            assert result.score == pytest.approx(1.0), (
                f"{name}/{distractor}/{condition} seed={seed} depth={depth}: "
                f"oracle {answer!r} scored {result.score} against {inst.ground_truth!r}"
            )
            assert result.correct
    assert scorer is not None


@pytest.mark.parametrize("name,params", TASK_CASES)
def test_content_independent_of_length(name, params):
    """CORE REQUIREMENT #1: growing the context must not change the task.

    The same seed must produce byte-identical needles, question, and ground truth
    at 1k tokens and at 1M. Only the amount of filler may differ.
    """
    task = get_task(name)
    for seed in SEEDS:
        instances = [
            task.generate(_spec(name, params, seed=seed, depth=0.5, num_needles=3,
                                target_tokens=L))
            for L in (1_000, 32_000, 1_000_000)
        ]
        first = instances[0]
        for other in instances[1:]:
            assert [n.text for n in other.needles] == [n.text for n in first.needles]
            assert other.question == first.question
            assert other.ground_truth == first.ground_truth
            assert other.scorer == first.scorer


@pytest.mark.parametrize("name,params", TASK_CASES)
def test_content_independent_of_depth(name, params):
    """Depth is a placement variable: it moves needles, it does not rewrite them."""
    task = get_task(name)
    by_depth = [
        task.generate(_spec(name, params, seed=3, depth=d, num_needles=3))
        for d in DEPTHS
    ]
    texts = {frozenset(n.text for n in inst.needles) for inst in by_depth}
    assert len(texts) == 1, f"{name}: needle text changed with depth"
    if not task.depth_selects_probe:
        assert len({str(i.ground_truth) for i in by_depth}) == 1
    else:
        # These tasks use depth to choose *which* item is asked about, so the
        # question must move while the document stays fixed.
        assert len({i.question for i in by_depth}) > 1


@pytest.mark.parametrize("name,params", TASK_CASES)
def test_seeds_produce_distinct_content(name, params):
    """CORE REQUIREMENT #2: independent trials must not reuse the same facts."""
    task = get_task(name)
    instances = [
        task.generate(_spec(name, params, seed=s, depth=0.5, num_needles=3))
        for s in range(12)
    ]
    # The document must be freshly generated every trial, always.
    documents = {"\n".join(n.text for n in i.needles) for i in instances}
    assert len(documents) == len(instances), (
        f"{name}: only {len(documents)} distinct documents across 12 seeds"
    )
    # The answer must also vary, except where the answer space is inherently
    # fixed (a count of a fixed number of items, or the ABSENT sentinel).
    if task.answer_is_verbatim:
        truths = {str(i.ground_truth) for i in instances}
        assert len(truths) >= max(2, len(instances) // 2), (
            f"{name}: only {len(truths)} distinct answers across 12 seeds"
        )


@pytest.mark.parametrize("name,params", TASK_CASES)
def test_ground_truth_absent_from_filler(name, params, word_tokenizer, corpus_filler):
    """CORE REQUIREMENT #2: the answer must appear only in a needle, never in filler.

    Guards against a filler corpus that can accidentally supply — or contradict —
    an answer, which would silently corrupt every cell built from it.
    """
    packer = Packer(word_tokenizer, corpus_filler, tolerance=0.02)
    task = get_task(name)
    for seed in (0, 5):
        inst = task.generate(
            _spec(name, params, seed=seed, depth=0.5, num_needles=3, target_tokens=3000)
        )
        packed = packer.pack(inst, 3000, "standard", seed)
        document = packed.messages[1]["content"]
        needle_text = "\n".join(n.text for n in inst.needles)
        # Strip the needles; whatever remains is filler plus the question.
        filler_only = document
        for n in inst.needles:
            filler_only = filler_only.replace(n.text, "")

        golds = inst.ground_truth if isinstance(inst.ground_truth, list) else [inst.ground_truth]
        for gold in golds:
            g = str(gold)
            if g == "ABSENT" or len(g) < 4:
                continue  # sentinel, or too short to be a meaningful collision
            assert g not in filler_only, f"{name}: ground truth {g!r} leaked into filler"
            # Only verbatim answers are expected to appear in the document at
            # all; a sum or a sentinel legitimately appears nowhere.
            if task.answer_is_verbatim and inst.target_needles:
                assert g in needle_text, f"{name}: ground truth {g!r} is in no needle"


def test_needle_absent_removes_the_evidence():
    """The absent control must actually remove the answer from the document."""
    for name, params in TASK_CASES:
        task = get_task(name)
        present = task.generate(_spec(name, params, seed=1, depth=0.5, num_needles=2))
        absent = task.generate(
            _spec(name, params, seed=1, depth=0.5, num_needles=2, condition="needle_absent")
        )
        assert absent.ground_truth == "ABSENT"
        assert absent.scorer == "absent_detection"
        if name == "negative_retrieval":
            continue  # already the absent case; nothing to remove
        golds = present.ground_truth if isinstance(present.ground_truth, list) else [present.ground_truth]
        doc = "\n".join(n.text for n in absent.needles)
        for gold in golds:
            if len(str(gold)) >= 4:
                assert str(gold) not in doc


def test_unknown_task_param_is_rejected():
    """A typo in a config param must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown params"):
        get_task("aggregation").generate(
            TaskSpec(task="aggregation", seed=0, params={"num_itemz": 5})
        )


def test_every_registered_task_is_covered_by_the_suite():
    """Keeps this file honest when a new task is added."""
    covered = {name for name, _ in TASK_CASES}
    assert covered == set(all_tasks()), (
        f"tasks missing from TASK_CASES: {set(all_tasks()) - covered}"
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_all_conditions_generate(condition):
    for name, params in TASK_CASES:
        inst = get_task(name).generate(
            _spec(name, params, seed=2, depth=0.5, num_needles=2, condition=condition)
        )
        assert inst.needles or condition == "needle_absent"
        assert inst.question
