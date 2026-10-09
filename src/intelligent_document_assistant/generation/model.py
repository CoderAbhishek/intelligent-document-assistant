"""Construct the chat model from environment configuration."""

import os

from langchain_core.language_models import BaseChatModel

from intelligent_document_assistant.generation.errors import MissingCredentialsError

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 30


def create_chat_model() -> BaseChatModel:
    """Build the Groq chat model from ``GROQ_API_KEY`` and optional ``GROQ_MODEL``."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise MissingCredentialsError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file."
        )

    # Imported lazily: the import takes several seconds.
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL,
        api_key=api_key,
        temperature=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
