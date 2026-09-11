"""Answer extraction and normalization.

Scoring must not punish a model for formatting. It must also not reward a model
for burying a guess inside prose. The rule: prefer the explicit ``<answer>`` tag
the prompt asks for, fall back to a small set of well-defined conventions, and
normalize away only presentation (case, quoting, markdown, separators).
"""

from __future__ import annotations

import re
import unicodedata

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_OPEN_TAG = re.compile(r"<answer>(.*)", re.IGNORECASE | re.DOTALL)
_PREAMBLE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is)?\s*:?\s*", re.IGNORECASE
)
_MARKDOWN = re.compile(r"[*_`]+")
_THINK = re.compile(r"<(think|thinking|scratchpad)>.*?</\1>", re.IGNORECASE | re.DOTALL)

ABSENT_FORMS = {
    "absent", "notpresent", "notfound", "none", "nothing", "na", "nan",
    "noanswer", "notinthedocument", "notindocument", "unknown", "notavailable",
    "nomatch", "doesnotappear", "notmentioned", "noneofthem", "nosuchvalue",
    "nosuchkey", "empty", "null", "nil",
}


def extract_answer(text: str) -> str:
    """Pull the model's answer out of a raw completion."""
    if not text:
        return ""
    # Reasoning traces are never the answer.
    text = _THINK.sub(" ", text)

    matches = _ANSWER_TAG.findall(text)
    if matches:
        # Last complete tag wins: models that restate settle on their final answer.
        return matches[-1].strip()
    unterminated = _OPEN_TAG.search(text)
    if unterminated:
        return unterminated.group(1).strip()

    stripped = text.strip()
    if not stripped:
        return ""
    # "The answer is X" on any line.
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        if _PREAMBLE.match(line):
            return _PREAMBLE.sub("", line).strip()
        return line
    return stripped


def normalize(text: str) -> str:
    """Presentation-insensitive form: case, markdown, quotes, whitespace, trailing stop."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    s = _MARKDOWN.sub("", s)
    s = _PREAMBLE.sub("", s).strip()
    s = s.strip("\"'“”‘’ \t\n\r")
    s = re.sub(r"[.。]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def canonical(text: str) -> str:
    """Comparison form: alphanumerics only.

    Makes ``K7QF-2M9X``, ``k7qf 2m9x`` and ``"K7QF2M9X."`` compare equal, so a
    model is never marked wrong for punctuating an identifier differently.
    """
    return re.sub(r"[^a-z0-9]+", "", normalize(text))


def is_absent(text: str) -> bool:
    """True when the answer is an abstention rather than a value."""
    c = canonical(text)
    if not c:
        return True
    if c in ABSENT_FORMS:
        return True
    # Short phrases such as "not present in the document".
    if len(c) <= 40 and any(
        c.startswith(f) or c.endswith(f) for f in ("notpresent", "notfound", "noaccesscode")
    ):
        return True
    return False


# Also split before an inline enumerator ("1. A 2. B"), which some models emit
# instead of a newline-separated list; without this the whole line reads as one item.
_ITEM_SPLIT = re.compile(r"[,\n;]+|\s+and\s+|\s+(?=\d{1,3}[.)]\s)", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)])\s*")


def split_items(text: str) -> list[str]:
    """Parse a list-shaped answer into normalized, de-duplicated items."""
    if not text or is_absent(text):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in _ITEM_SPLIT.split(text):
        piece = _BULLET.sub("", raw).strip()
        c = canonical(piece)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_number(text: str) -> float | None:
    """First numeric literal in the text, thousands separators tolerated."""
    if text is None:
        return None
    cleaned = re.sub(r"[$€£%]", "", str(text))
    m = _NUMBER.search(cleaned)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None
