"""Run the production retrieval and generation functions over the labelled questions.

Rows are plain dicts so they can be shown in a notebook and saved as JSON.
"""

from collections.abc import Iterable, Sequence

from llama_index.core import VectorStoreIndex

from evaluation.dataset import EvalItem
from evaluation.metrics import (
    abstention_confusion,
    coverage_at_k,
    hit_at_k,
    mean,
    ranked_pages,
    reciprocal_rank,
    rubric_passes,
)
from intelligent_document_assistant.generation import GenerationError, answer_question
from intelligent_document_assistant.retrieval import retrieve_passages

Row = dict


def run_retrieval(
    index: VectorStoreIndex, items: Iterable[EvalItem], top_k: int = 5, ks: Sequence[int] = (1, 3, 5)
) -> list[Row]:
    """Retrieve for each question. Metrics are computed for answerable questions only."""
    if max(ks) > top_k:
        raise ValueError("every k must be <= top_k")
    rows = []
    for item in items:
        chunk_pages = [p.page_number for p in retrieve_passages(index, item.question, top_k)]
        row: Row = {
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "answerable": item.answerable,
            "relevant_pages": list(item.relevant_pages),
            "chunk_pages": chunk_pages,
            "rr": reciprocal_rank(chunk_pages, item.relevant_pages) if item.answerable else None,
        }
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(chunk_pages, item.relevant_pages, k) if item.answerable else None
            row[f"coverage@{k}"] = (
                coverage_at_k(chunk_pages, item.relevant_pages, k) if item.answerable else None
            )
        rows.append(row)
    return rows


def summarize_retrieval(rows: Iterable[Row], ks: Sequence[int] = (1, 3, 5)) -> dict:
    """Aggregate over answerable rows: hit rate, coverage, and MRR."""
    answerable = [r for r in rows if r["answerable"]]
    summary: dict = {"n": len(answerable), "mrr": mean([r["rr"] for r in answerable])}
    for k in ks:
        summary[f"hit@{k}"] = mean([float(r[f"hit@{k}"]) for r in answerable])
        summary[f"mean_coverage@{k}"] = mean([r[f"coverage@{k}"] for r in answerable])
        summary[f"full_coverage@{k}"] = mean([float(r[f"coverage@{k}"] == 1.0) for r in answerable])
    return summary


def run_generation(
    index: VectorStoreIndex, items: Iterable[EvalItem], chat_model, top_k: int = 5
) -> list[Row]:
    """Call ``answer_question`` for each item; a failed call is recorded, not raised."""
    rows = []
    for item in items:
        retrieved_pages = ranked_pages(
            [p.page_number for p in retrieve_passages(index, item.question, top_k)], top_k
        )
        row: Row = {
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "answerable": item.answerable,
            "relevant_pages": list(item.relevant_pages),
            "retrieved_pages": retrieved_pages,
            "evidence_retrieved": set(item.relevant_pages) <= set(retrieved_pages) if item.answerable else None,
            "answer": None,
            "abstained": None,
            "cited_pages": [],
            "rubric_pass": None,
            "cited_includes_relevant": None,
            "cited_text_supports": None,
            "error": None,
        }
        try:
            response = answer_question(index, item.question, chat_model, top_k)
        except GenerationError as exc:
            row["error"] = type(exc).__name__
            rows.append(row)
            continue

        row["answer"] = response.answer
        row["abstained"] = response.is_abstention
        row["cited_pages"] = sorted({p.page_number for p in response.cited_passages})
        if item.answerable and not response.is_abstention:
            cited_text = " ".join(p.text for p in response.cited_passages)
            row["rubric_pass"] = rubric_passes(response.answer, item.answer_criteria)
            row["cited_includes_relevant"] = bool(set(row["cited_pages"]) & set(item.relevant_pages))
            # Content proxy only: the rubric facts occur in the cited text, not proof of support.
            row["cited_text_supports"] = rubric_passes(cited_text, item.answer_criteria)
        rows.append(row)
    return rows


def summarize_generation(rows: Iterable[Row]) -> dict:
    """Counts for answers, citations, and abstention (positive class = abstention)."""
    rows = list(rows)
    completed = [r for r in rows if r["error"] is None]
    answered = [r for r in completed if r["answerable"] and not r["abstained"]]
    return {
        "n": len(rows),
        "errors": len(rows) - len(completed),
        "confusion": abstention_confusion(completed),
        "answerable_answered": len(answered),
        "rubric_pass": sum(bool(r["rubric_pass"]) for r in answered),
        "cited_includes_relevant": sum(bool(r["cited_includes_relevant"]) for r in answered),
        "cited_text_supports": sum(bool(r["cited_text_supports"]) for r in answered),
        "answerable_wrongly_abstained_with_evidence": sum(
            r["answerable"] and r["abstained"] and r["evidence_retrieved"] for r in completed
        ),
        "answerable_wrongly_abstained_without_evidence": sum(
            r["answerable"] and r["abstained"] and not r["evidence_retrieved"] for r in completed
        ),
    }


def markdown_table(rows: Iterable[Row], columns: Sequence[str]) -> str:
    """Render rows as a GitHub-flavoured markdown table."""

    def cell(value) -> str:
        return "-" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)
