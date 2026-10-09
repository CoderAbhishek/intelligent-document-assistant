"""Stub chat models shared by the answer-generation and UI tests."""

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field


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
