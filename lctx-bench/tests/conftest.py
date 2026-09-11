"""Shared fixtures. Everything here runs offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `pytest` from a checkout without an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lctx.config import ExperimentConfig  # noqa: E402
from lctx.haystack.filler import CorpusFiller, RandomTokenFiller  # noqa: E402
from lctx.models.tokenizers import WordTokenizer  # noqa: E402

#: Every registered task, with a parameter set that exercises its options.
TASK_CASES: list[tuple[str, dict]] = [
    ("single_needle", {}),
    ("multi_needle", {}),
    ("multi_hop", {"chain_length": 2}),
    ("multi_hop", {"chain_length": 5}),
    ("aggregation", {"num_items": 6, "num_offbatch": 4, "op": "sum"}),
    ("aggregation", {"num_items": 6, "num_offbatch": 4, "op": "count"}),
    ("aggregation", {"num_items": 6, "num_offbatch": 4, "op": "max"}),
    ("selective_copy", {"num_matching": 4, "num_nonmatching": 8}),
    ("ordering", {"num_items": 8}),
    ("negative_retrieval", {"num_present": 5}),
]

DISTRACTORS = ["none", "near_duplicate_key", "semantic_lure", "repeated_decoy"]
CONDITIONS = ["standard", "no_haystack", "shuffled_haystack", "needle_absent"]


@pytest.fixture
def word_tokenizer() -> WordTokenizer:
    return WordTokenizer()


@pytest.fixture
def corpus_filler() -> CorpusFiller:
    return CorpusFiller()


@pytest.fixture
def random_filler() -> RandomTokenFiller:
    return RandomTokenFiller(vocab_size=512)


@pytest.fixture
def mini_config(tmp_path: Path) -> ExperimentConfig:
    """A minimal but complete config: two lengths, three depths, mock model."""
    return ExperimentConfig.model_validate(
        {
            "run": {
                "name": "test",
                "output_dir": str(tmp_path / "runs"),
                "n_trials": 3,
                "max_concurrency": 8,
                "store_prompts": "hash",
            },
            "grid": {
                "lengths": [600, 2400],
                "depths": [0.0, 0.5, 1.0],
                "conditions": ["standard", "no_haystack"],
                "tolerance": 0.02,
            },
            "stats": {"bootstrap_resamples": 200},
            "models": [
                {
                    "name": "sim",
                    "adapter": "mock",
                    "tokenizer": {"kind": "word"},
                    "params": {"behavior": "degrading", "half_length": 1500},
                }
            ],
            "tasks": [
                {"name": "single_needle", "distractor_types": ["none"]},
                {"name": "selective_copy",
                 "params": {"num_matching": 3, "num_nonmatching": 5}},
            ],
        }
    )
