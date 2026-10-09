class RetrievalError(Exception):
    """Base class for indexing and retrieval failures."""


class NoIndexableContentError(RetrievalError):
    """There is no text to chunk or index."""


class InvalidQueryError(RetrievalError):
    """The query is empty or whitespace-only."""
