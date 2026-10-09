"""Load the labelled evaluation questions."""

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("dataset.json")


@dataclass(frozen=True)
class EvalItem:
    id: str
    category: str
    question: str
    answerable: bool
    relevant_pages: tuple[int, ...]
    answer_criteria: tuple[tuple[str, ...], ...]
    expected_answer: str
    absent_terms: tuple[str, ...]


def load_items(path: Path = DATASET_PATH) -> list[EvalItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalItem(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            answerable=item["answerable"],
            relevant_pages=tuple(item["relevant_pages"]),
            answer_criteria=tuple(tuple(group) for group in item["answer_criteria"]),
            expected_answer=item["expected_answer"],
            absent_terms=tuple(item.get("absent_terms", ())),
        )
        for item in raw["items"]
    ]
