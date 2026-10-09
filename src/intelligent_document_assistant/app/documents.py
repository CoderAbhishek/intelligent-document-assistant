"""Turn uploaded PDF bytes into an indexed document. No Streamlit code here."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding

from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import index_document
from intelligent_document_assistant.retrieval.chunking import FILE_NAME_KEY

MAX_NAME_LENGTH = 100
DEFAULT_NAME = "document.pdf"
_TEMP_PDF_NAME = "upload.pdf"


@dataclass(frozen=True)
class IndexedDocument:
    """An uploaded PDF, identified by content hash, with its in-memory index."""

    digest: str
    file_name: str
    page_count: int
    index: VectorStoreIndex


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def display_name(raw_name: str) -> str:
    """Reduce an uploaded file name to a printable basename for display and provenance."""
    basename = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    printable = "".join(ch for ch in basename if ch.isprintable()).strip()
    return printable[:MAX_NAME_LENGTH] or DEFAULT_NAME


def build_document_index(
    data: bytes, raw_name: str, embed_model: BaseEmbedding
) -> IndexedDocument:
    """Ingest and index uploaded PDF bytes.

    Raises the ``PDFIngestionError`` and ``RetrievalError`` subclasses of the pipeline.
    """
    name = display_name(raw_name)

    # A fixed name inside a private temp dir keeps user input out of filesystem paths.
    with tempfile.TemporaryDirectory(prefix="ida_") as temp_dir:
        pdf_path = Path(temp_dir) / _TEMP_PDF_NAME
        pdf_path.write_bytes(data)
        document = load_pdf(pdf_path)

    # The loader recorded the temp name; restore the name the user uploaded.
    document.metadata[FILE_NAME_KEY] = name
    index = index_document(document, embed_model)
    return IndexedDocument(
        digest=file_digest(data),
        file_name=name,
        page_count=document.metadata["page_count"],
        index=index,
    )
