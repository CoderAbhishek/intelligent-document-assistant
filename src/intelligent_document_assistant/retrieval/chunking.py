"""Turn an ingested PDF ``Document`` into page-aware chunks for indexing."""

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from intelligent_document_assistant.ingestion import PAGE_SEPARATOR
from intelligent_document_assistant.retrieval.errors import NoIndexableContentError

FILE_NAME_KEY = "file_name"
PAGE_NUMBER_KEY = "page_number"

# Measured in tokens of LlamaIndex's default tokenizer (tiktoken cl100k_base), not
# characters and not the embedding model's tokenizer. 256 leaves ample room under
# the 512-token input limit of the embedding model.
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32

_METADATA_KEYS = [FILE_NAME_KEY, PAGE_NUMBER_KEY]


def split_into_pages(document: Document) -> list[Document]:
    """Split a ``load_pdf`` document into one ``Document`` per non-blank page.

    Pages are numbered from 1; blank pages are skipped without shifting numbers.
    """
    file_name = document.metadata[FILE_NAME_KEY]
    return [
        Document(
            text=text,
            metadata={FILE_NAME_KEY: file_name, PAGE_NUMBER_KEY: page_number},
            # Provenance is for the application, not for embedding or prompt text.
            # Metadata left in either counts against chunk_size and can overflow it.
            excluded_embed_metadata_keys=_METADATA_KEYS,
            excluded_llm_metadata_keys=_METADATA_KEYS,
        )
        for page_number, text in enumerate(document.text.split(PAGE_SEPARATOR), start=1)
        if text.strip()
    ]


def chunk_document(
    document: Document,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[BaseNode]:
    """Split a document into chunks that each belong to exactly one page.

    Raises ``NoIndexableContentError`` if the document has no text.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(split_into_pages(document))
    if not nodes:
        file_name = document.metadata.get(FILE_NAME_KEY, "document")
        raise NoIndexableContentError(f"There is no text to index in {file_name}")
    return nodes
