"""Streamlit interface: upload one PDF, ask questions, inspect the sources."""

import logging
import re
from dataclasses import dataclass

import streamlit as st

from intelligent_document_assistant.app.documents import (
    IndexedDocument,
    build_document_index,
    file_digest,
)
from intelligent_document_assistant.generation import (
    AnswerResponse,
    GenerationError,
    MissingCredentialsError,
    answer_question,
    create_chat_model,
)
from intelligent_document_assistant.ingestion import (
    PDFExtractionError,
    PDFFileError,
    PDFIngestionError,
    PDFNoTextError,
    PDFPasswordProtectedError,
)
from intelligent_document_assistant.retrieval import (
    RetrievalError,
    RetrievedPassage,
    create_embed_model,
)

logger = logging.getLogger(__name__)

# Per-browser-session state (st.session_state keys).
DOCUMENT_KEY = "document"  # IndexedDocument for the current upload, or None
RESULT_KEY = "result"  # QAResult for the latest answered question, or None
LOAD_ERROR_KEY = "load_error"  # LoadError for an upload that failed, or None
UPLOADER_NONCE_KEY = "uploader_nonce"  # changing it resets the upload widget

_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]<>!~$#|])")

_INGESTION_MESSAGES = {
    PDFPasswordProtectedError: "This PDF is password-protected. Remove the password and upload it again.",
    PDFNoTextError: (
        "This PDF has no extractable text. It may be a scan; scanned PDFs (which need OCR) "
        "are not supported."
    ),
    PDFExtractionError: "Text could not be read from one of this PDF's pages.",
    PDFFileError: "This file is not a valid, readable PDF.",
}


@dataclass(frozen=True)
class QAResult:
    question: str
    response: AnswerResponse


@dataclass(frozen=True)
class LoadError:
    digest: str
    message: str


# Shared by all sessions: both objects are read-only after construction.
@st.cache_resource(show_spinner="Loading the embedding model (the first run downloads ~130 MB)...")
def get_embed_model():
    return create_embed_model()


@st.cache_resource
def get_chat_model():
    return create_chat_model()


def escape_markdown(text: str) -> str:
    """Make model- or document-supplied text render as plain text in Streamlit markdown.

    Escaping brackets and ``!`` blocks links and images: a remote image URL written by a
    prompt-injected model would otherwise be fetched by the browser.
    """
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text)


def main() -> None:
    st.set_page_config(page_title="Intelligent Document Assistant", page_icon="📄")
    st.title("Intelligent Document Assistant")
    st.caption(
        "Upload a text-based PDF and ask questions about it. Answers are based only on "
        "passages retrieved from the document, shown with their page numbers."
    )

    nonce = st.session_state.get(UPLOADER_NONCE_KEY, 0)
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key=f"uploader-{nonce}")
    _sync_document(uploaded)

    document: IndexedDocument | None = st.session_state.get(DOCUMENT_KEY)
    if document is None:
        if uploaded is None:
            st.info("Upload a text-based PDF to get started.")
        return

    pages = f"{document.page_count} page{'s' if document.page_count != 1 else ''}"
    st.success(f"**{escape_markdown(document.file_name)}** is indexed ({pages}).")
    st.button("Start over with another PDF", key="start_over", on_click=_start_over)

    with st.form("question-form"):
        question = st.text_input("Ask a question about this document")
        submitted = st.form_submit_button("Ask", key="ask")
    if submitted:
        _ask(document, question)

    result: QAResult | None = st.session_state.get(RESULT_KEY)
    if result is not None:
        _render_result(result)


def _sync_document(uploaded) -> None:
    """Make session state match the uploaded file, indexing it only when it is new."""
    if uploaded is None:
        _clear_document_state()
        return

    data = uploaded.getvalue()
    digest = file_digest(data)
    current: IndexedDocument | None = st.session_state.get(DOCUMENT_KEY)
    if current is not None and current.digest == digest:
        return
    failed: LoadError | None = st.session_state.get(LOAD_ERROR_KEY)
    if failed is not None and failed.digest == digest:
        st.error(failed.message)
        return

    _clear_document_state()
    try:
        embed_model = get_embed_model()
    except Exception:
        logger.exception("Could not load the embedding model")
        _fail_load(digest, "The embedding model could not be loaded. The first run needs internet access to download it.")
        return

    try:
        with st.spinner("Reading and indexing the PDF..."):
            st.session_state[DOCUMENT_KEY] = build_document_index(data, uploaded.name, embed_model)
    except (PDFIngestionError, RetrievalError) as exc:
        _fail_load(digest, _ingestion_message(exc))
    except Exception:
        logger.exception("Unexpected error while indexing the uploaded PDF")
        _fail_load(digest, "Something went wrong while processing this PDF. Details are in the server log.")


def _ingestion_message(exc: Exception) -> str:
    # Generic wording per error type: exception text can mention the internal temp file name.
    for error_type, message in _INGESTION_MESSAGES.items():
        if isinstance(exc, error_type):
            return message
    return "This PDF could not be processed."


def _fail_load(digest: str, message: str) -> None:
    st.session_state[LOAD_ERROR_KEY] = LoadError(digest, message)
    st.error(message)


def _clear_document_state() -> None:
    st.session_state[DOCUMENT_KEY] = None
    st.session_state[RESULT_KEY] = None
    st.session_state[LOAD_ERROR_KEY] = None


def _start_over() -> None:
    _clear_document_state()
    st.session_state[UPLOADER_NONCE_KEY] = st.session_state.get(UPLOADER_NONCE_KEY, 0) + 1


def _ask(document: IndexedDocument, question: str) -> None:
    if not question.strip():
        st.warning("Enter a question before asking.")
        return

    st.session_state[RESULT_KEY] = None  # never show an old answer next to a new failure
    try:
        chat_model = get_chat_model()
        with st.spinner("Searching the document and writing an answer..."):
            response = answer_question(document.index, question, chat_model)
    except MissingCredentialsError as exc:
        st.error(f"{exc} Then restart the app.")
    except (GenerationError, RetrievalError) as exc:
        st.error(f"{exc}. Please try again.")
    except Exception:
        logger.exception("Unexpected error while answering a question")
        st.error("Something went wrong while answering. Details are in the server log.")
    else:
        st.session_state[RESULT_KEY] = QAResult(question.strip(), response)


def _render_result(result: QAResult) -> None:
    response = result.response
    st.divider()
    st.markdown(f"**Question:** {escape_markdown(result.question)}")

    if response.is_abstention:
        st.warning(
            f"**Not enough information in the document.**\n\n{escape_markdown(response.answer)}"
        )
    else:
        st.subheader("Answer")
        st.markdown(escape_markdown(response.answer))

    cited = response.cited_passages
    if cited:
        st.subheader("Sources cited by the model")
        for passage in cited:
            with st.expander(_passage_title(passage)):
                st.text(passage.text)

    uncited = [p for p in response.retrieved_passages if p not in cited]
    if uncited:
        if response.is_abstention:
            title, note = "Closest passages retrieved", "The model found that these do not answer the question."
        elif cited:
            title, note = "Other retrieved passages the model did not cite", "Found by similarity search; they may not support the answer."
        else:
            title, note = "Retrieved passages: the model did not cite any", "Found by similarity search; they may not support the answer."
        with st.expander(f"{title} ({len(uncited)})"):
            st.caption(note)
            for passage in uncited:
                st.markdown(f"**{_passage_title(passage)}**")
                st.text(passage.text)


def _passage_title(passage: RetrievedPassage) -> str:
    return f"{escape_markdown(passage.file_name)} — page {passage.page_number}"
