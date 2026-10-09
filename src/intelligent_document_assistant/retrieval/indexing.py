"""Build an in-memory vector index from chunked documents."""

from collections.abc import Sequence

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode

from intelligent_document_assistant.retrieval.chunking import chunk_document
from intelligent_document_assistant.retrieval.errors import NoIndexableContentError

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Pinned so a change upstream cannot silently alter the vectors.
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def create_embed_model() -> BaseEmbedding:
    """Load the local embedding model (downloaded from Hugging Face on first use)."""
    # Imported lazily: it pulls in torch, which callers injecting their own model avoid.
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION)


def build_index(nodes: Sequence[BaseNode], embed_model: BaseEmbedding) -> VectorStoreIndex:
    """Embed ``nodes`` and store them, with their text and metadata, in memory."""
    if not nodes:
        raise NoIndexableContentError("Cannot build an index from zero chunks")
    return VectorStoreIndex(nodes=list(nodes), embed_model=embed_model)


def index_document(document: Document, embed_model: BaseEmbedding) -> VectorStoreIndex:
    """Chunk a ``load_pdf`` document page by page and index the chunks."""
    return build_index(chunk_document(document), embed_model)
