import pytest
from langchain_groq import ChatGroq

from intelligent_document_assistant.generation import MissingCredentialsError, create_chat_model
from intelligent_document_assistant.generation.model import DEFAULT_GROQ_MODEL


def test_missing_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="GROQ_API_KEY"):
        create_chat_model()


def test_blank_api_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    with pytest.raises(MissingCredentialsError):
        create_chat_model()


def test_model_is_built_from_environment_without_exposing_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")

    model = create_chat_model()

    assert isinstance(model, ChatGroq)
    assert model.model_name == "some/other-model"
    assert model.temperature < 1e-6  # ChatGroq turns 0 into 1e-08
    assert "gsk_test_not_a_real_key" not in repr(model)


def test_model_name_defaults_when_not_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert create_chat_model().model_name == DEFAULT_GROQ_MODEL
