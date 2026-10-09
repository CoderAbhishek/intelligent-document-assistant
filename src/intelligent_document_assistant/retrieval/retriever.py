"""Retrieve the passages most similar to a query from a vector index."""

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from intelligent_document_assistant.retrieval.chunking import FILE_NAME_KEY, PAGE_NUMBER_KEY
from intelligent_document_assistant.retrieval.errors import InvalidQueryError

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievedPassage:
    """A retrieved chunk with its source. ``score`` ranks passages for one query.

    It is a similarity score, not a probability, and is not comparable across queries.
    """

    text: str
    file_name: str
    page_number: int
    score: float | None


def retrieve_passages(
    index: VectorStoreIndex, query: str, top_k: int = DEFAULT_TOP_K
) -> list[RetrievedPassage]:
    """Return up to ``top_k`` passages, most similar first.

    Always returns the nearest passages, even when none is truly relevant.
    Raises ``InvalidQueryError`` for a blank query and ``ValueError`` for ``top_k < 1``.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not query.strip():
        raise InvalidQueryError("The query is empty")

    results = index.as_retriever(similarity_top_k=top_k).retrieve(query.strip())
    return [_to_passage(result) for result in results]


def _to_passage(result: NodeWithScore) -> RetrievedPassage:
    return RetrievedPassage(
        text=result.node.get_content(),
        file_name=result.node.metadata[FILE_NAME_KEY],
        page_number=result.node.metadata[PAGE_NUMBER_KEY],
        score=result.score,
    )
