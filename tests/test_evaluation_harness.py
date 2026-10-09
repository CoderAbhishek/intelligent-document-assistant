import pytest
from chat_stubs import FailingChatModel, model_replying, reply

from evaluation.corpus import build_pdf
from evaluation.dataset import EvalItem, load_items
from evaluation.harness import (
    markdown_table,
    run_generation,
    run_retrieval,
    summarize_generation,
    summarize_retrieval,
)
from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import index_document


@pytest.fixture(scope="module")
def index(tmp_path_factory, corpus_embed_model):
    document = load_pdf(build_pdf(tmp_path_factory.mktemp("harness")))
    return index_document(document, corpus_embed_model)


def item(id: str, question: str, pages=(), criteria=(), answerable=True) -> EvalItem:
    return EvalItem(id, "direct", question, answerable, tuple(pages), tuple(criteria), "", ())


def test_retrieval_rows_record_ranked_pages_and_page_level_hits(index):
    items = [item("q", "What is the quay length of the South Basin terminal?", pages=[5])]

    (row,) = run_retrieval(index, items, top_k=5, ks=(1, 3, 5))

    assert row["relevant_pages"] == [5]
    assert len(row["chunk_pages"]) == 5
    assert row["hit@1"] is True and row["rr"] == 1.0
    assert row["coverage@5"] == 1.0


def test_hits_never_decrease_as_k_grows(index):
    rows = run_retrieval(index, [i for i in load_items() if i.answerable], top_k=5, ks=(1, 3, 5))

    for row in rows:
        assert row["hit@1"] <= row["hit@3"] <= row["hit@5"], row["id"]
        assert row["coverage@1"] <= row["coverage@3"] <= row["coverage@5"], row["id"]


def test_unanswerable_questions_have_no_retrieval_metrics(index):
    rows = run_retrieval(index, [item("u", "What is the capital of France?", answerable=False)], top_k=3, ks=(1, 3))

    assert rows[0]["rr"] is None and rows[0]["hit@3"] is None
    assert len(rows[0]["chunk_pages"]) == 3  # retrieval still returns the nearest passages


def test_retrieval_summary_averages_answerable_rows_only():
    rows = [
        {"answerable": True, "relevant_pages": [1], "hit@1": True, "hit@3": True, "coverage@1": 1.0, "coverage@3": 1.0, "rr": 1.0, "category": "direct"},
        {"answerable": True, "relevant_pages": [2, 3], "hit@1": False, "hit@3": True, "coverage@1": 0.0, "coverage@3": 0.5, "rr": 0.5, "category": "multi_page"},
        {"answerable": False, "relevant_pages": [], "hit@1": None, "hit@3": None, "coverage@1": None, "coverage@3": None, "rr": None, "category": "absent"},
    ]

    summary = summarize_retrieval(rows, ks=(1, 3))

    assert summary["n"] == 2
    assert summary["hit@1"] == 0.5 and summary["hit@3"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["full_coverage@3"] == 0.5  # only the first item has every required page by k=3


def test_generation_row_checks_answer_rubric_and_cited_pages(index):
    items = [item("q", "What is the quay length of the South Basin terminal?", pages=[5], criteria=[["980"]])]
    model = model_replying(reply(True, "The quay is 980 metres long.", [1]))

    (row,) = run_generation(index, items, model, top_k=3)

    assert row["abstained"] is False
    assert row["rubric_pass"] is True
    assert row["cited_pages"] == [5]
    assert row["cited_includes_relevant"] is True and row["cited_text_supports"] is True
    assert row["evidence_retrieved"] is True
    assert row["error"] is None


def test_an_answer_with_the_wrong_citation_is_flagged_even_if_the_text_passes_the_rubric(index):
    items = [item("q", "What is the quay length of the South Basin terminal?", pages=[5], criteria=[["980"]])]
    # passage 2 is not the South Basin page, so the claim is not supported by what was cited
    model = model_replying(reply(True, "980 metres.", [2]))

    (row,) = run_generation(index, items, model, top_k=3)

    assert row["rubric_pass"] is True
    assert row["cited_includes_relevant"] is False
    assert row["cited_text_supports"] is False


def test_generation_failure_is_blamed_on_retrieval_when_evidence_was_not_retrieved(index):
    items = [item("q", "What is the quay length of the South Basin terminal?", pages=[10], criteria=[["120"]])]

    (row,) = run_generation(index, items, model_replying(reply(False, "Not stated.")), top_k=1)

    assert row["abstained"] is True
    assert row["evidence_retrieved"] is False  # page 10 was not among the retrieved pages


def test_a_failing_model_call_is_recorded_and_the_run_continues(index):
    items = [item("a", "Quay length of South Basin?", pages=[5]), item("b", "Rail share at North Quay?", pages=[4])]

    rows = run_generation(index, items, FailingChatModel(responses=["unused"]), top_k=2)

    assert [r["error"] for r in rows] == ["ModelCallError", "ModelCallError"]
    assert all(r["abstained"] is None for r in rows)


def test_unverifiable_citation_is_recorded_as_an_output_error(index):
    rows = run_generation(index, [item("a", "Quay length of South Basin?", pages=[5])], model_replying(reply(True, "x", [99])), top_k=2)

    assert rows[0]["error"] == "ModelOutputError"


def test_generation_summary_reports_abstention_confusion_and_errors():
    rows = [
        {"answerable": True, "abstained": False, "rubric_pass": True, "cited_includes_relevant": True, "cited_text_supports": True, "evidence_retrieved": True, "error": None},
        {"answerable": True, "abstained": True, "rubric_pass": None, "cited_includes_relevant": None, "cited_text_supports": None, "evidence_retrieved": True, "error": None},
        {"answerable": False, "abstained": True, "rubric_pass": None, "cited_includes_relevant": None, "cited_text_supports": None, "evidence_retrieved": None, "error": None},
        {"answerable": False, "abstained": False, "rubric_pass": None, "cited_includes_relevant": None, "cited_text_supports": None, "evidence_retrieved": None, "error": None},
        {"answerable": True, "abstained": None, "rubric_pass": None, "cited_includes_relevant": None, "cited_text_supports": None, "evidence_retrieved": True, "error": "ModelCallError"},
    ]

    summary = summarize_generation(rows)

    assert summary["errors"] == 1
    assert summary["confusion"]["tp"] == 1 and summary["confusion"]["fp"] == 1
    assert summary["confusion"]["fn"] == 1 and summary["confusion"]["tn"] == 1
    assert summary["answerable_answered"] == 1
    assert summary["rubric_pass"] == 1
    assert summary["cited_includes_relevant"] == 1
    assert summary["answerable_wrongly_abstained_with_evidence"] == 1


def test_markdown_table_renders_header_and_escapes_pipes():
    table = markdown_table([{"a": 1, "b": "x|y"}], ["a", "b"])

    assert table.splitlines()[0] == "| a | b |"
    assert table.splitlines()[1] == "|---|---|"
    assert table.splitlines()[2] == "| 1 | x\\|y |"
