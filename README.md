# Intelligent Document Assistant

A document question-answering application built on retrieval-augmented generation (RAG). The goal: upload a PDF, ask questions in natural language, and receive answers grounded in the passages retrieved from that document.

> **Project status: work in progress.** PDF ingestion, semantic retrieval, and grounded answer generation are implemented and tested as a Python library. The web interface is planned and is **not yet available**. See [Key capabilities](#key-capabilities).

## Problem statement

Finding a specific answer in a long PDF means skimming or keyword searching, which misses passages that use different wording. Asking a general-purpose language model instead risks answers that sound plausible but are not supported by the document. This project aims to combine semantic retrieval with a language model so that answers are based on retrieved passages, shown alongside the excerpts that support them, and withheld when the document lacks sufficient evidence.

## Key capabilities

| Capability | Status |
|---|---|
| PDF validation and text extraction | Implemented |
| Page-aware chunking with file and page provenance | Implemented |
| Local embeddings and in-memory vector index | Implemented |
| Semantic retrieval of ranked passages with page references | Implemented |
| Grounded answer generation (LangChain + Groq) with cited passages | Implemented |
| Explicit "not enough evidence" responses | Implemented (prompt-based, see [Limitations](#limitations)) |
| Streamlit user interface | Planned |
| Deployment | Planned |

The first version targets a single, text-based PDF at a time.

## How it works

```
PDF -> validation + text extraction -> one Document per page -> chunks (nodes)
    -> embeddings -> in-memory vector index -> semantic retrieval -> ranked passages with page numbers
    -> context formatting -> prompt + chat model -> parsed answer with cited passages
```

All stages shown are implemented; the Streamlit interface that will sit on top is planned.

Each page is chunked on its own, so a chunk never spans two pages and its page number is exact. A fact that straddles a page break is therefore split across two chunks.

## Architecture and technology stack

| Component | Role | State |
|---|---|---|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracts text from PDF pages | In use |
| [LlamaIndex](https://docs.llamaindex.ai/) | Document and node representation, chunking, in-memory vector index, retrieval | In use |
| [Sentence-transformers](https://www.sbert.net/) model `BAAI/bge-small-en-v1.5` | Local text embeddings (384 dimensions), no API key | In use |
| [LangChain](https://python.langchain.com/) (`langchain-core`, `langchain-groq`) | Prompt template, chat-model call, output parsing | In use |
| [Groq](https://groq.com/) chat model (default `openai/gpt-oss-20b`) | Writes the answer from the supplied passages | In use, requires an API key |
| [Streamlit](https://streamlit.io/) | Web interface | Planned |
| Python library (`answer_question`) | Connects retrieval and generation | In use |

The index lives in memory and no remote vector database is used.

Source layout:

```
src/intelligent_document_assistant/
    ingestion/    # PDF validation and text extraction
    retrieval/    # chunking, embeddings and indexing, retrieval
    generation/   # evidence formatting, prompt, chat model, answer parsing
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

The embedding model is downloaded from Hugging Face the first time it is used (about 130 MB) and cached locally afterwards, so the first run needs network access and later runs work offline.

### Configuration

Answer generation calls the Groq API and needs a key from [console.groq.com](https://console.groq.com/). Copy the template and fill in your own values; `.env` is git-ignored.

```powershell
Copy-Item .env.example .env
```

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required for answer generation |
| `GROQ_MODEL` | Optional; defaults to `openai/gpt-oss-20b` |

The library reads these from the process environment and does not load `.env` itself. To run a script with the file applied:

```powershell
uv run --env-file .env python your_script.py
```

Ingestion and retrieval need no key.

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

To get an answer, pass the index and a chat model to `answer_question` (requires `GROQ_API_KEY`):

```python
from intelligent_document_assistant.generation import answer_question, create_chat_model

response = answer_question(index, "When was the company founded?", create_chat_model(), top_k=5)

print(response.answer)
print(response.is_abstention)  # True when the document does not answer the question
for passage in response.cited_passages:
    print(passage.file_name, passage.page_number)
```

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

### Answer generation

`answer_question(index, question, chat_model, top_k=5)` retrieves passages, labels each with its number, file name and page, and asks the model to answer from them only. It returns an `AnswerResponse`:

| Field | Meaning |
|---|---|
| `answer` | The answer, or an explanation of what the document lacks |
| `is_abstention` | `True` when the model reports that the evidence does not answer the question |
| `retrieved_passages` | Every passage the model was shown |
| `cited_passages` | The passages the model named by number, checked to be among those retrieved |

A citation records which passages the model says it relied on. It does not prove that the passage supports the answer. If nothing is retrieved, the response is an abstention and the model is not called.

Errors derive from `GenerationError`: `MissingCredentialsError`, `ModelCallError` (the request failed; the message names only the exception type), and `ModelOutputError` (malformed reply or a citation of a passage that was not provided). A blank question raises `InvalidQueryError`.

## Testing

```powershell
uv run pytest
```

This runs the fast offline tests. They generate their own small PDFs and use a deterministic stand-in embedding and a stub chat model, so they need no network access, model download, or API key.

```powershell
uv run pytest -m integration
```

This runs the opt-in tests that use real models: retrieval against the real embedding model (downloaded on first run), and answer generation against the live Groq API. The generation tests are skipped unless `GROQ_API_KEY` is set. To supply it from `.env`:

```powershell
uv run --env-file .env pytest -m integration
```

The live tests send small fictional test passages to Groq.

## Limitations

- No OCR: scanned or image-only PDFs are rejected. If only some pages lack text, the remaining pages are extracted and `pages_with_text` is lower than `page_count`.
- Password-protected PDFs are rejected; supplying a password is not supported. PDFs that only restrict copying or printing open normally.
- Extraction returns the text PyMuPDF produces. There is no layout analysis, so tables, multi-column pages, and headers or footers are not specially handled.
- Chunks do not cross page boundaries. Facts split across a page break are retrieved only in part.
- Retrieval quality has been checked only on a small synthetic document. The embedding model is a small English model; non-English documents are not evaluated.
- Chunking parameters and `top_k` are initial defaults, not tuned values.
- "Not enough evidence" is decided by the language model following the prompt, not by a retrieval-score threshold, because similarity scores are not calibrated probabilities. Prompt instructions reduce but do not eliminate unsupported answers or susceptibility to instructions hidden in a document, and the abstention behaviour has been checked on only a few questions.
- Answers are only as good as retrieval: if the right passage is not among the top-k, the model cannot use it.
- Each question is answered independently; there is no conversation memory.
- The index is in memory and is rebuilt for each document; there is no persistence.
- Only one document at a time is in scope.

## Security and document handling

- Secrets go in a local `.env` file, which is git-ignored. Never commit real keys.
- `*.pdf` is git-ignored to avoid committing private documents; test PDFs are generated in code.
- Ingestion and indexing process documents in memory and run locally. Document text is not logged. Network access is limited to the one-time embedding model download and, for answer generation, the Groq API call.
- Error messages contain the file name only, never its directory or its contents.
- Answer generation sends the question and the retrieved passages to the Groq API. Do not use documents you are not permitted to share with that provider. Ingestion, indexing, and retrieval stay local.
- The API key is read from the environment, is never logged, and is not included in error messages.

## Future improvements

- Streamlit interface and deployment.
- Retrieval-quality evaluation on realistic documents, and tuning of chunking and `top_k`.
- Possible later extensions: OCR for scanned documents, multiple documents, and advanced retrieval techniques.

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
