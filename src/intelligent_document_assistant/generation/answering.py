"""Answer a question from retrieved passages with a LangChain chat model."""

from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field

from intelligent_document_assistant.generation.errors import ModelCallError, ModelOutputError
from intelligent_document_assistant.generation.prompt import ANSWER_PROMPT, format_evidence
from intelligent_document_assistant.retrieval import (
    RetrievedPassage,
    retrieve_passages,
)
from intelligent_document_assistant.retrieval.retriever import DEFAULT_TOP_K

NO_ANSWER_MESSAGE = "The document does not contain enough information to answer this question."


class _ModelAnswer(BaseModel):
    answerable: bool = Field(
        description="true only if the evidence contains the information needed to answer"
    )
    answer: str = Field(description="the answer, or a short note on what the evidence lacks")
    citations: list[int] = Field(
        default_factory=list, description="numbers of the passages the answer relies on"
    )


_parser = PydanticOutputParser(pydantic_object=_ModelAnswer)
_prompt = ANSWER_PROMPT.partial(format_instructions=_parser.get_format_instructions())


@dataclass(frozen=True)
class AnswerResponse:
    """An answer plus its sources.

    ``retrieved_passages`` is everything the model was shown. ``cited_passages`` is the
    subset the model named by number; a citation shows the model's claim, not proof that
    the passage supports the answer.
    """

    answer: str
    is_abstention: bool
    retrieved_passages: tuple[RetrievedPassage, ...]
    cited_passages: tuple[RetrievedPassage, ...]


def answer_question(
    index: VectorStoreIndex,
    question: str,
    chat_model: BaseChatModel,
    top_k: int = DEFAULT_TOP_K,
) -> AnswerResponse:
    """Retrieve passages for ``question`` and have ``chat_model`` answer from them only.

    Raises ``InvalidQueryError`` for a blank question, ``ModelCallError`` if the request
    fails, and ``ModelOutputError`` if the reply is malformed or cites unknown passages.
    With no retrieved passages it abstains without calling the model.
    """
    passages = retrieve_passages(index, question, top_k)
    if not passages:
        return AnswerResponse(NO_ANSWER_MESSAGE, True, (), ())

    parsed = _generate(chat_model, question.strip(), passages)
    return _to_response(parsed, passages)


def _generate(
    chat_model: BaseChatModel, question: str, passages: list[RetrievedPassage]
) -> _ModelAnswer:
    chain = _prompt | chat_model | _parser
    try:
        return chain.invoke({"question": question, "evidence": format_evidence(passages)})
    except OutputParserException as exc:
        raise ModelOutputError("The model's reply was not in the expected format") from exc
    except Exception as exc:
        # Only the exception type is reported: provider messages can echo request details.
        raise ModelCallError(f"The language model request failed ({type(exc).__name__})") from exc


def _to_response(parsed: _ModelAnswer, passages: list[RetrievedPassage]) -> AnswerResponse:
    answer = parsed.answer.strip()
    if not parsed.answerable:
        return AnswerResponse(answer or NO_ANSWER_MESSAGE, True, tuple(passages), ())

    if not answer:
        raise ModelOutputError("The model marked the question answerable but gave no answer")
    unknown = [n for n in parsed.citations if not 1 <= n <= len(passages)]
    if unknown:
        raise ModelOutputError(f"The model cited passages that were not provided: {unknown}")

    cited = tuple(passages[n - 1] for n in dict.fromkeys(parsed.citations))
    return AnswerResponse(answer, False, tuple(passages), cited)
