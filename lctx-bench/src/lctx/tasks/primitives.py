"""Random content primitives and distractor builders.

Core requirement #2 — zero parametric leakage: every key, value, and entity name
is drawn fresh from the trial's RNG out of an arbitrary symbol alphabet. Nothing
here is answerable from pretraining, and nothing is reused across trials.
"""

from __future__ import annotations

import random
from typing import Callable

_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O, which read ambiguously in codes
_DIGIT = "23456789"                   # no 0/1, same reason
_CODE = _ALPHA + _DIGIT

_ONSET = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "tr", "kl", "vr"]
_NUCLEUS = ["a", "e", "i", "o", "u", "ae", "ei", "ou"]
_CODA = ["", "", "n", "m", "r", "l", "s", "k", "th", "x"]

FactTemplate = Callable[[str, str], str]


def random_code(rng: random.Random, groups: int = 2, glen: int = 4, sep: str = "-") -> str:
    """An opaque identifier such as ``K7QF-2M9X``."""
    return sep.join("".join(rng.choice(_CODE) for _ in range(glen)) for _ in range(groups))


def random_uuid(rng: random.Random) -> str:
    """A UUID4-shaped string drawn from ``rng`` so trials stay reproducible."""
    h = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}"


def random_number(rng: random.Random, lo: int = 100, hi: int = 999_999) -> int:
    return rng.randint(lo, hi)


def random_word(rng: random.Random, syllables: int = 2) -> str:
    """A pronounceable nonsense word, e.g. ``vorlim``. Not a real lexical item."""
    out = "".join(rng.choice(_ONSET) + rng.choice(_NUCLEUS) + rng.choice(_CODA)
                  for _ in range(syllables))
    return out.capitalize()


def unique(fn: Callable[[], str], seen: set[str], limit: int = 1000) -> str:
    """Draw from ``fn`` until the value is new, then record it."""
    for _ in range(limit):
        v = fn()
        if v not in seen:
            seen.add(v)
            return v
    raise RuntimeError("could not draw a unique value; widen the identifier space")


def perturb_code(rng: random.Random, code: str, n_edits: int = 1) -> str:
    """A near-duplicate of ``code``: same shape, one or two characters changed."""
    chars = list(code)
    positions = [i for i, c in enumerate(chars) if c in _CODE]
    if not positions:
        return code
    for _ in range(n_edits):
        i = rng.choice(positions)
        alt = [c for c in _CODE if c != chars[i]]
        chars[i] = rng.choice(alt)
    out = "".join(chars)
    return out if out != code else perturb_code(rng, code, n_edits)


# -- distractors ----------------------------------------------------------

#: Attribute names used by semantic lures: same surface shape as the real fact,
#: different referent. Answering with one of these means the model matched on
#: topic rather than on the exact relation asked about.
LURE_QUALIFIERS = [
    "deprecated", "backup", "provisional", "legacy", "mirrored", "expired", "draft",
]


def build_distractors(
    kind: str,
    rng: random.Random,
    template: FactTemplate,
    true_key: str,
    true_value: str,
    n: int,
    value_fn: Callable[[], str],
    key_fn: Callable[[], str],
    seen_keys: set[str] | None = None,
    seen_values: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(text, label)`` pairs of distractor snippets.

    The four kinds probe different failure modes:

    ``none``
        No distractors — a clean retrieval baseline.
    ``near_duplicate_key``
        Keys one or two characters from the real one. Probes whether retrieval is
        exact or merely approximate-match.
    ``semantic_lure``
        The real key with a *different* attribute, and similar-looking entities
        with the right attribute. Probes relation binding, not just key spotting.
    ``repeated_decoy``
        A single wrong fact repeated throughout. Probes whether frequency in
        context overrides correctness.
    """
    seen_keys = seen_keys if seen_keys is not None else set()
    seen_values = seen_values if seen_values is not None else set()
    seen_keys.add(true_key)
    seen_values.add(true_value)
    out: list[tuple[str, str]] = []

    if kind == "none" or n <= 0:
        return out

    if kind == "near_duplicate_key":
        for _ in range(n):
            k = unique(lambda: perturb_code(rng, true_key, rng.choice([1, 1, 2])), seen_keys)
            v = unique(value_fn, seen_values)
            out.append((template(k, v), f"near_dup:{k}"))
        return out

    if kind == "semantic_lure":
        # Half: the true key, but a qualified (wrong) attribute.
        # Half: a different key carrying the right attribute.
        for i in range(n):
            if i % 2 == 0:
                q = rng.choice(LURE_QUALIFIERS)
                v = unique(value_fn, seen_values)
                base = template(true_key, v)
                # Qualify the attribute itself: "The access code ..." becomes
                # "The deprecated access code ...", which is about a different
                # thing while matching the query on every surface cue.
                if base.startswith("The "):
                    text = "The " + q + " " + base[4:]
                else:
                    text = f"({q.capitalize()}) " + base
                out.append((text, f"lure_attr:{q}"))
            else:
                k = unique(key_fn, seen_keys)
                v = unique(value_fn, seen_values)
                out.append((template(k, v), f"lure_key:{k}"))
        return out

    if kind == "repeated_decoy":
        k = unique(key_fn, seen_keys)
        v = unique(value_fn, seen_values)
        text = template(k, v)
        for _ in range(n):
            out.append((text, f"decoy:{k}"))
        return out

    raise ValueError(f"unknown distractor_type: {kind!r}")


#: How many distractors each kind emits by default.
DEFAULT_DISTRACTOR_COUNT = {
    "none": 0,
    "near_duplicate_key": 4,
    "semantic_lure": 4,
    "repeated_decoy": 6,
}
