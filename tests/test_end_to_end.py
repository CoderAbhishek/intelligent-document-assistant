"""Upload-to-answer regression checks through the public APIs, with stand-in models."""

import pytest
from chat_stubs import FailingChatModel, model_replying, reply

from evaluation.corpus import build_pdf
from intelligent_document_assistant.app.documents import build_document_index
from intelligent_document_assistant.generation import ModelCallError, answer_question

QUAY_QUESTION = "What is the quay length of the South Basin terminal?"


@pytest.fixture(scope="module")
def corpus_bytes(tmp_path_factory):
    return build_pdf(tmp_path_factory.mktemp("e2e")).read_bytes()


def test_uploaded_name_and_original_page_survive_to_the_cited_source(corpus_bytes, corpus_embed_model):
    document = build_document_index(corpus_bytes, "Meridian 2024 report.pdf", corpus_embed_model)

    response = answer_question(
        document.index, QUAY_QUESTION, model_replying(reply(True, "980 metres.", [1])), top_k=3
    )

    (cited,) = response.cited_passages
    assert (cited.file_name, cited.page_number) == ("Meridian 2024 report.pdf", 5)
    assert "980 metres" in cited.text
    assert {p.file_name for p in response.retrieved_passages} == {"Meridian 2024 report.pdf"}


def test_a_replacement_document_never_returns_the_previous_documents_passages(
    corpus_bytes, company_pdf, corpus_embed_model
):
    first = build_document_index(corpus_bytes, "first.pdf", corpus_embed_model)
    second = build_document_index(company_pdf.read_bytes(), "second.pdf", corpus_embed_model)

    response = answer_question(second.index, QUAY_QUESTION, model_replying(reply(False, "n/a")), top_k=5)

    assert {p.file_name for p in response.retrieved_passages} == {"second.pdf"}
    assert first.digest != second.digest


def test_an_abstention_is_a_response_and_a_provider_failure_is_an_error(corpus_bytes, corpus_embed_model):
    document = build_document_index(corpus_bytes, "report.pdf", corpus_embed_model)

    abstention = answer_question(document.index, "What is the capital of France?", model_replying(reply(False, "Not in the document.")))
    with pytest.raises(ModelCallError):
        answer_question(document.index, QUAY_QUESTION, FailingChatModel(responses=["unused"]))

    assert abstention.is_abstention is True and abstention.cited_passages == ()
