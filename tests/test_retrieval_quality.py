"""Retrieval checks with the real embedding model.

Run with ``uv run pytest -m integration``. The first run downloads the model
(~130 MB) from Hugging Face; later runs use the local cache.
"""

import pytest
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from transformers import AutoTokenizer

from intelligent_document_assistant.ingestion import PAGE_SEPARATOR, load_pdf
from intelligent_document_assistant.retrieval import (
    EMBEDDING_MODEL_NAME,
    chunk_document,
    create_embed_model,
    index_document,
    retrieve_passages,
)
from intelligent_document_assistant.retrieval.indexing import EMBEDDING_MODEL_REVISION

pytestmark = pytest.mark.integration

PARAPHRASED_QUESTIONS = [
    ("When did the firm start operating?", 1),
    ("Where is the head office situated?", 2),
    ("What machines does the business sell?", 3),
]


@pytest.fixture(scope="session")
def embed_model():
    return create_embed_model()


@pytest.fixture
def index(company_pdf, embed_model):
    return index_document(load_pdf(company_pdf), embed_model)


def test_embedding_model_is_the_configured_one(embed_model):
    assert embed_model.model_name == EMBEDDING_MODEL_NAME
    assert len(embed_model.get_text_embedding("a short sentence")) == 384


@pytest.mark.parametrize(("question", "expected_page"), PARAPHRASED_QUESTIONS)
def test_paraphrased_question_retrieves_the_right_page_first(index, question, expected_page):
    top = retrieve_passages(index, question, top_k=1)[0]

    assert top.page_number == expected_page
    assert top.file_name == "altair.pdf"


def test_question_spanning_two_facts_retrieves_both_pages(index):
    question = "Where did the company begin and where is it based today?"

    pages = {p.page_number for p in retrieve_passages(index, question, top_k=2)}

    assert pages == {1, 2}


def test_ambiguous_question_returns_passages_from_several_pages(index):
    passages = retrieve_passages(index, "Tell me about the company.", top_k=3)

    assert len({p.page_number for p in passages}) == 3


@pytest.mark.parametrize("question", [q for q, _ in PARAPHRASED_QUESTIONS])
def test_unrelated_question_scores_lower_than_a_relevant_one(index, question):
    unrelated = retrieve_passages(index, "What is the recipe for chocolate cake?", top_k=4)
    relevant = retrieve_passages(index, question, top_k=4)

    # Retrieval has no "no answer" outcome: the nearest passages always come back.
    assert len(unrelated) == 4
    assert unrelated[0].score < relevant[0].score


def test_default_chunks_fit_the_embedding_models_input_limit(embed_model):
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION
    )
    # Numbers and hyphenated terms split into many word-pieces: a worst-case-ish page.
    page = " ".join(
        f"Section {i}.{j}: the thermo-mechanical coefficient was recalibrated to {i * j * 3.14159:.4f} MPa."
        for i in range(1, 9)
        for j in range(1, 9)
    )
    nodes = chunk_document(Document(text=page, metadata={"file_name": "dense.pdf"}))

    sizes = [len(tokenizer(n.get_content(MetadataMode.EMBED))["input_ids"]) for n in nodes]

    assert len(nodes) > 1
    assert max(sizes) <= embed_model.max_length
