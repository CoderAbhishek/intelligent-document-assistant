# Intelligent Document Assistant

A document question-answering application built on retrieval-augmented generation (RAG). The goal: upload a PDF, ask questions in natural language, and receive answers grounded in the passages retrieved from that document.

> **Project status: work in progress.** PDF ingestion is implemented and tested. Retrieval, answer generation, and the web interface are planned and are **not yet available**. See [Key capabilities](#key-capabilities).

## Problem statement

Finding a specific answer in a long PDF means skimming or keyword searching, which misses passages that use different wording. Asking a general-purpose language model instead risks answers that sound plausible but are not supported by the document. This project aims to combine semantic retrieval with a language model so that answers are based on retrieved passages, shown alongside the excerpts that support them, and withheld when the document lacks sufficient evidence.

## Key capabilities

| Capability | Status |
|---|---|
| PDF validation and text extraction | Implemented |
| LlamaIndex `Document` with source metadata | Implemented |
| Chunking, embeddings, in-memory vector index | Planned |
| Semantic retrieval of relevant passages | Planned |
| Grounded answer generation with source excerpts | Planned |
| Streamlit user interface | Planned |
| Deployment | Planned |

The first version targets a single, text-based PDF at a time.

## How it works

Intended pipeline (only the first stage exists today):

```
PDF upload -> validation + text extraction -> LlamaIndex Document
  -> chunking + embeddings -> vector index -> retrieval
  -> context formatting -> prompt + language model -> answer with supporting excerpts
```

## Architecture and technology stack

| Component | Role | State |
|---|---|---|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracts text from PDF pages | In use |
| [LlamaIndex](https://docs.llamaindex.ai/) | Document representation (in use); chunking, indexing, retrieval (planned) | Partly in use |
| [LangChain](https://python.langchain.com/) | Prompt construction, model call, output parsing | Planned |
| [Streamlit](https://streamlit.io/) | Web interface | Planned |
| Python service layer | Connects ingestion, retrieval, and generation | Planned |

The initial version keeps its index in memory and uses no remote vector database.

Source layout:

```
src/intelligent_document_assistant/
    ingestion/pdf_loader.py   # PDF validation and extraction
tests/
```

## Installation and setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```powershell
git clone https://github.com/CoderAbhishek/intelligent-document-assistant.git
cd intelligent-document-assistant
uv sync
```

`uv sync` creates `.venv` and installs the locked dependencies, including the development group. No API key is needed for the current functionality. `.env.example` lists the variables that later features will use.

## Usage

There is no user interface yet. The ingestion component can be used from Python:

```python
from intelligent_document_assistant.ingestion import load_pdf

document = load_pdf("path/to/file.pdf")
document.metadata  # {'file_name': 'file.pdf', 'page_count': 12, 'pages_with_text': 12}
```

`load_pdf` returns one LlamaIndex `Document` for the whole PDF:

- **Text:** pages in order, joined by a form-feed character (`PAGE_SEPARATOR`). Blank pages stay as empty entries so page positions are preserved.
- **Metadata:** `file_name`, `page_count`, and `pages_with_text`.
- **Errors:** failures raise a subclass of `PDFIngestionError`:

| Exception | Raised when |
|---|---|
| `PDFFileError` | The path is missing or is a directory, the file is empty, not a valid PDF, or has no pages |
| `PDFPasswordProtectedError` | The PDF requires a password to open |
| `PDFNoTextError` | No page has extractable text (scanned, image-only, or blank) |
| `PDFExtractionError` | The PDF opened but reading a page failed |

## Testing

```powershell
uv run pytest
```

The tests generate their own small PDFs and need no network access, API key, or sample files.

## Limitations

- No OCR: scanned or image-only PDFs are rejected. If only some pages lack text, the remaining pages are extracted and `pages_with_text` is lower than `page_count`.
- Password-protected PDFs are rejected; supplying a password is not supported. PDFs that only restrict copying or printing open normally.
- Extraction returns the text PyMuPDF produces. There is no layout analysis, so tables, multi-column pages, and headers or footers are not specially handled.
- Files are accepted based on content, not file extension.
- Only one document at a time is in scope.

## Security and document handling

- Secrets go in a local `.env` file, which is git-ignored. Never commit real keys.
- `*.pdf` is git-ignored to avoid committing private documents; test PDFs are generated in code.
- Ingestion processes documents in memory. Error messages contain the file name only, never its directory or its contents, and document text is not logged.
- Answer generation, once added, will send retrieved passages to an external language model provider. Do not upload documents you are not permitted to share with such a provider.

## Future improvements

- Semantic retrieval with source excerpts and page references.
- Grounded answers that decline when the document lacks evidence.
- Streamlit interface and deployment.
- Possible later extensions: OCR for scanned documents, multiple documents, and advanced retrieval techniques.

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
