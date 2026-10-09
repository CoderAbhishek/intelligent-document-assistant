"""Evaluation metrics. Pure functions over page numbers and text; no I/O or model calls."""

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence

_DASHES = re.compile(r"[‐-―−]")  # hyphens, en/em dashes, minus sign


def ranked_pages(chunk_pages: Sequence[int], k: int) -> list[int]:
    """Distinct pages of the top-``k`` chunks, in first-appearance order."""
    return list(dict.fromkeys(chunk_pages[:k]))


def hit_at_k(chunk_pages: Sequence[int], relevant: Iterable[int], k: int) -> bool:
    """True if any relevant page appears among the top-``k`` chunks."""
    return bool(set(ranked_pages(chunk_pages, k)) & set(relevant))


def coverage_at_k(chunk_pages: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the required pages found in the top-``k`` chunks."""
    required = set(relevant)
    return len(set(ranked_pages(chunk_pages, k)) & required) / len(required)


def reciprocal_rank(chunk_pages: Sequence[int], relevant: Iterable[int]) -> float:
    """1 / rank of the first chunk on a relevant page (0.0 if none was retrieved)."""
    required = set(relevant)
    for rank, page in enumerate(chunk_pages, start=1):
        if page in required:
            return 1 / rank
    return 0.0


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _normalise(text: str) -> str:
    """Lower-case and fold the typographic variants models emit (no-break spaces, dashes)."""
    text = _DASHES.sub("-", unicodedata.normalize("NFKC", text).casefold())
    return " ".join(text.split()).replace(" %", "%")


def rubric_passes(answer: str, criteria: Sequence[Sequence[str]]) -> bool:
    """Every criteria group needs at least one of its alternatives in the answer.

    A case-insensitive substring check on normalised text: a floor for correctness,
    not a semantic judge.
    """
    text = _normalise(answer)
    return all(any(_normalise(alt) in text for alt in group) for group in criteria)


def abstention_confusion(rows: Iterable[Mapping[str, bool]]) -> dict[str, float | int | None]:
    """Confusion counts with *abstaining* as the positive class.

    ``tp``: unanswerable and abstained. ``fp``: answerable but abstained.
    ``fn``: unanswerable but answered. ``tn``: answerable and answered.
    """
    tp = fp = fn = tn = 0
    for row in rows:
        if row["abstained"]:
            tp, fp = (tp + 1, fp) if not row["answerable"] else (tp, fp + 1)
        else:
            fn, tn = (fn + 1, tn) if not row["answerable"] else (fn, tn + 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
