# Intelligent Document Assistant

A document question-answering application built on retrieval-augmented generation (RAG). The goal: upload a PDF, ask questions in natural language, and receive answers grounded in the passages retrieved from that document.

> **Project status: work in progress.** PDF ingestion and semantic retrieval are implemented and tested. Answer generation and the web interface are planned and are **not yet available**. See [Key capabilities](#key-capabilities).

## Problem statement

Finding a specific answer in a long PDF means skimming or keyword searching, which misses passages that use different wording. Asking a general-purpose language model instead risks answers that sound plausible but are not supported by the document. This project aims to combine semantic retrieval with a language model so that answers are based on retrieved passages, shown alongside the excerpts that support them, and withheld when the document lacks sufficient evidence.

## Key capabilities

| Capability | Status |
|---|---|
| PDF validation and text extraction | Implemented |
| Page-aware chunking with file and page provenance | Implemented |
| Local embeddings and in-memory vector index | Implemented |
| Semantic retrieval of ranked passages with page references | Implemented |
| Grounded answer generation with source excerpts | Planned |
| Streamlit user interface | Planned |
| Deployment | Planned |

The first version targets a single, text-based PDF at a time.

## How it works

```
PDF -> validation + text extraction -> one Document per page -> chunks (nodes)
    -> embeddings -> in-memory vector index -> semantic retrieval -> ranked passages with page numbers
    -> context formatting -> prompt + language model -> answer with supporting excerpts
```

The first two lines are implemented. The last line is planned.

Each page is chunked on its own, so a chunk never spans two pages and its page number is exact. A fact that straddles a page break is therefore split across two chunks.

## Architecture and technology stack

| Component | Role | State |
|---|---|---|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracts text from PDF pages | In use |
| [LlamaIndex](https://docs.llamaindex.ai/) | Document and node representation, chunking, in-memory vector index, retrieval | In use |
| [Sentence-transformers](https://www.sbert.net/) model `BAAI/bge-small-en-v1.5` | Local text embeddings (384 dimensions), no API key | In use |
| [LangChain](https://python.langchain.com/) | Prompt construction, model call, output parsing | Planned |
| [Streamlit](https://streamlit.io/) | Web interface | Planned |
| Python service layer | Connects ingestion, retrieval, and generation | Planned |

The index lives in memory and no remote vector database is used.

Source layout:

```
src/intelligent_document_assistant/
    ingestion/    # PDF validation and text extraction
    retrieval/    # chunking, embeddings and indexing, retrieval
tests/
```

## Installation and setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```powershell
git clone https://github.com/CoderAbhishek/intelligent-document-assistant.git
cd intelligent-document-assistant
uv sync
```

`uv sync` creates `.venv` and installs the locked dependencies, including the development group. The environment is large (about 1 GB) because the local embedding model runs on PyTorch.

The embedding model is downloaded from Hugging Face the first time it is used (about 130 MB) and cached locally afterwards, so the first run needs network access and later runs work offline. No API key is needed for the current functionality. `.env.example` lists variables that later features will use.

## Usage

There is no user interface yet. The components can be used from Python:

```python
from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import create_embed_model, index_document, retrieve_passages

document = load_pdf("path/to/file.pdf")
index = index_document(document, create_embed_model())

for passage in retrieve_passages(index, "When was the company founded?", top_k=3):
    print(passage.file_name, passage.page_number, passage.score, passage.text)
```

Build the index once and reuse it: `retrieve_passages` embeds only the query, so `top_k` can change per call without re-indexing.

### Ingestion

`load_pdf` returns one LlamaIndex `Document` for the whole PDF with `file_name`, `page_count` and `pages_with_text` metadata. Pages are joined by a form-feed character (`PAGE_SEPARATOR`) and blank pages stay as empty entries so page positions are preserved. Failures raise a subclass of `PDFIngestionError`:

| Exception | Raised when |
|---|---|
| `PDFFileError` | The path is missing or is a directory, the file is empty, not a valid PDF, or has no pages |
| `PDFPasswordProtectedError` | The PDF requires a password to open |
| `PDFNoTextError` | No page has extractable text (scanned, image-only, or blank) |
| `PDFExtractionError` | The PDF opened but reading a page failed |

### Indexing and retrieval

- `index_document` splits the document into one page-level document per non-blank page, chunks each page (256 tokens with 32 tokens of overlap, counted with LlamaIndex's default tokenizer), embeds the chunks, and stores them in a `VectorStoreIndex`.
- Page number and file name are kept as metadata on every chunk but are excluded from the text that gets embedded.
- `retrieve_passages(index, query, top_k=5)` returns `RetrievedPassage` objects (`text`, `file_name`, `page_number`, `score`), most similar first. The score is a cosine similarity used to rank passages for one query. It is not a probability and does not indicate whether a passage is correct or answers the question.
- Retrieval always returns the nearest passages, even when none is relevant. Deciding whether the evidence is sufficient is left to the generation stage.
- Errors derive from `RetrievalError`: `NoIndexableContentError` (nothing to index) and `InvalidQueryError` (blank query). `top_k < 1` raises `ValueError`.

## Testing

```powershell
uv run pytest
```

This runs the fast offline tests. They generate their own small PDFs and use a deterministic stand-in embedding, so they need no network access, model download, or API key.

```powershell
uv run pytest -m integration
```

This runs the retrieval tests against the real embedding model on a small generated document with known facts on separate pages. It downloads the model on first run.

## Limitations

- No OCR: scanned or image-only PDFs are rejected. If only some pages lack text, the remaining pages are extracted and `pages_with_text` is lower than `page_count`.
- Password-protected PDFs are rejected; supplying a password is not supported. PDFs that only restrict copying or printing open normally.
- Extraction returns the text PyMuPDF produces. There is no layout analysis, so tables, multi-column pages, and headers or footers are not specially handled.
- Chunks do not cross page boundaries. Facts split across a page break are retrieved only in part.
- Retrieval quality has been checked only on a small synthetic document. The embedding model is a small English model; non-English documents are not evaluated.
- Chunking parameters and `top_k` are initial defaults, not tuned values.
- The index is in memory and is rebuilt for each document; there is no persistence.
- Only one document at a time is in scope.

## Security and document handling

- Secrets go in a local `.env` file, which is git-ignored. Never commit real keys.
- `*.pdf` is git-ignored to avoid committing private documents; test PDFs are generated in code.
- Ingestion and indexing process documents in memory and run locally. Document text is not sent to any service and is not logged. The only network access is the one-time model download.
- Error messages contain the file name only, never its directory or its contents.
- Answer generation, once added, will send retrieved passages to an external language model provider. Do not upload documents you are not permitted to share with such a provider.

## Future improvements

- Grounded answers with source excerpts that decline when the document lacks evidence.
- Streamlit interface and deployment.
- Possible later extensions: OCR for scanned documents, multiple documents, and advanced retrieval techniques.

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
