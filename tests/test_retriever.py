import pytest
from llama_index.core import VectorStoreIndex

from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import (
    InvalidQueryError,
    RetrievedPassage,
    index_document,
    retrieve_passages,
)


@pytest.fixture
def index(company_pdf, keyword_embed_model):
    return index_document(load_pdf(company_pdf), keyword_embed_model)


def test_top_result_is_the_matching_passage_with_its_provenance(index):
    (top,) = retrieve_passages(index, "Who founded it, and what was the garage in Lisbon?", top_k=1)

    assert isinstance(top, RetrievedPassage)
    assert top.file_name == "altair.pdf"
    assert top.page_number == 1
    assert "founded in 1998" in top.text
    assert isinstance(top.score, float)


@pytest.mark.parametrize(
    ("query", "expected_page"),
    [
        ("headquarters Rotterdam shipyard", 2),
        ("Heron robot and Wren drone", 3),
        ("employed people in Singapore", 4),
    ],
)
def test_each_fact_maps_back_to_its_original_page(index, query, expected_page):
    top = retrieve_passages(index, query, top_k=1)[0]

    assert top.page_number == expected_page


def test_top_k_limits_the_number_of_passages(index):
    assert len(retrieve_passages(index, "Heron Rotterdam Lisbon", top_k=2)) == 2


def test_top_k_larger_than_the_index_returns_everything(index):
    assert len(retrieve_passages(index, "Heron Rotterdam Lisbon", top_k=50)) == 4


def test_results_are_ordered_by_descending_score(index):
    scores = [p.score for p in retrieve_passages(index, "Heron robot Rotterdam", top_k=4)]

    assert scores == sorted(scores, reverse=True)


def test_top_k_can_change_without_rebuilding_the_index(company_pdf, keyword_embed_model):
    index = index_document(load_pdf(company_pdf), keyword_embed_model)
    embed_calls_after_build = keyword_embed_model.text_embed_calls

    retrieve_passages(index, "Heron", top_k=1)
    retrieve_passages(index, "Heron", top_k=3)

    assert keyword_embed_model.text_embed_calls == embed_calls_after_build


def test_query_can_retrieve_from_more_than_one_page(index):
    pages = {p.page_number for p in retrieve_passages(index, "Lisbon garage and Rotterdam shipyard", top_k=2)}

    assert pages == {1, 2}


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_blank_query_raises(index, query):
    with pytest.raises(InvalidQueryError, match="empty"):
        retrieve_passages(index, query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(index, top_k):
    with pytest.raises(ValueError, match="top_k"):
        retrieve_passages(index, "company", top_k=top_k)


def test_empty_index_returns_no_passages(keyword_embed_model):
    empty_index = VectorStoreIndex(nodes=[], embed_model=keyword_embed_model)

    assert retrieve_passages(empty_index, "company") == []
