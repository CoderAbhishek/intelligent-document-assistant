"""Answer generation against the live Groq API.

Opt-in: ``uv run --env-file .env pytest -m integration tests/test_generation_live.py``.
Needs GROQ_API_KEY and network access, and sends the fictional test passages to Groq.
Retrieval uses the offline keyword embedding, so no embedding model is downloaded.
"""

import os

import pytest

from intelligent_document_assistant.generation import answer_question, create_chat_model
from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import index_document

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


@pytest.fixture
def index(company_pdf, keyword_embed_model):
    return index_document(load_pdf(company_pdf), keyword_embed_model)


def test_answerable_question_is_answered_and_cites_the_right_page(index):
    response = answer_question(
        index, "In what year was the company founded, and where was the garage?", create_chat_model()
    )

    assert response.is_abstention is False
    assert "1998" in response.answer
    assert 1 in {p.page_number for p in response.cited_passages}


def test_question_the_evidence_cannot_answer_is_an_abstention(index):
    response = answer_question(index, "What colour is the Heron robot?", create_chat_model())

    assert response.is_abstention is True
    assert response.cited_passages == ()
