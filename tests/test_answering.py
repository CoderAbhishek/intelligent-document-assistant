import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from llama_index.core import VectorStoreIndex
from pydantic import Field

from intelligent_document_assistant.generation import (
    GenerationError,
    ModelCallError,
    ModelOutputError,
    answer_question,
)
from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import (
    InvalidQueryError,
    index_document,
    retrieve_passages,
)

FOUNDING_QUESTION = "Who founded it, and what was the garage in Lisbon?"


class RecordingChatModel(FakeListChatModel):
    """Replies with canned text and records the prompts it was given."""

    prompts: list[str] = Field(default_factory=list)

    def _call(self, messages, *args, **kwargs):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return super()._call(messages, *args, **kwargs)


class FailingChatModel(RecordingChatModel):
    def _call(self, messages, *args, **kwargs):
        super()._call(messages, *args, **kwargs)
        raise RuntimeError("provider exploded, key=sk-secret-123")


def reply(answerable: bool, answer: str, citations: list[int] | None = None) -> str:
    return json.dumps({"answerable": answerable, "answer": answer, "citations": citations or []})


def model_replying(text: str) -> RecordingChatModel:
    return RecordingChatModel(responses=[text])


@pytest.fixture
def index(company_pdf, keyword_embed_model):
    return index_document(load_pdf(company_pdf), keyword_embed_model)


def test_answerable_question_returns_answer_with_cited_source(index):
    model = model_replying(reply(True, "It was founded in 1998 in a Lisbon garage.", [1]))

    response = answer_question(index, FOUNDING_QUESTION, model, top_k=3)

    assert response.answer == "It was founded in 1998 in a Lisbon garage."
    assert response.is_abstention is False
    (cited,) = response.cited_passages
    assert (cited.file_name, cited.page_number) == ("altair.pdf", 1)
    assert "founded in 1998" in cited.text


def test_response_keeps_every_retrieved_passage_unchanged(index):
    model = model_replying(reply(True, "answer", [1]))

    response = answer_question(index, FOUNDING_QUESTION, model, top_k=3)

    assert response.retrieved_passages == tuple(retrieve_passages(index, FOUNDING_QUESTION, top_k=3))
    assert len(response.retrieved_passages) == 3


def test_prompt_contains_question_and_labelled_evidence(index):
    model = model_replying(reply(True, "answer", [1]))

    answer_question(index, FOUNDING_QUESTION, model, top_k=2)

    (prompt,) = model.prompts
    assert FOUNDING_QUESTION in prompt
    assert 'file="altair.pdf" page="1"' in prompt
    assert "founded in 1998" in prompt


def test_insufficient_evidence_is_reported_as_abstention(index):
    model = model_replying(reply(False, "The document does not say.", [2]))

    response = answer_question(index, "What colour is the Heron robot?", model)

    assert response.is_abstention is True
    assert response.answer == "The document does not say."
    assert response.cited_passages == ()  # an abstention cites nothing
    assert response.retrieved_passages  # but the closest passages remain available


def test_abstention_without_explanation_gets_a_default_message(index):
    response = answer_question(index, FOUNDING_QUESTION, model_replying(reply(False, "  ")))

    assert response.is_abstention is True
    assert "does not contain enough information" in response.answer


def test_json_inside_a_code_fence_is_accepted(index):
    fenced = "```json\n" + reply(True, "fenced", [1]) + "\n```"

    response = answer_question(index, FOUNDING_QUESTION, model_replying(fenced))

    assert response.answer == "fenced"


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_blank_question_is_rejected_before_calling_the_model(index, question):
    model = model_replying(reply(True, "x"))

    with pytest.raises(InvalidQueryError):
        answer_question(index, question, model)

    assert model.prompts == []


def test_empty_index_abstains_without_calling_the_model(keyword_embed_model):
    empty_index = VectorStoreIndex(nodes=[], embed_model=keyword_embed_model)
    model = model_replying(reply(True, "should never be used"))

    response = answer_question(empty_index, FOUNDING_QUESTION, model)

    assert response.is_abstention is True
    assert response.retrieved_passages == () and response.cited_passages == ()
    assert model.prompts == []


def test_provider_failure_becomes_model_call_error_without_leaking_details(index):
    model = FailingChatModel(responses=["unused"])

    with pytest.raises(ModelCallError) as excinfo:
        answer_question(index, FOUNDING_QUESTION, model)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "sk-secret-123" not in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_output",
    [
        "I think the answer is 1998.",
        '{"answer": "no answerable field"}',
        '{"answerable": "maybe", "answer": "x", "citations": []}',
        reply(True, "cites a passage that was never retrieved", [99]),
        reply(True, "cites passage zero", [0]),
        reply(True, "   ", [1]),
    ],
)
def test_malformed_or_unverifiable_output_raises_model_output_error(index, bad_output):
    with pytest.raises(ModelOutputError):
        answer_question(index, FOUNDING_QUESTION, model_replying(bad_output), top_k=3)


def test_all_generation_errors_share_one_base_class():
    assert issubclass(ModelCallError, GenerationError)
    assert issubclass(ModelOutputError, GenerationError)
