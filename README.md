# Intelligent Document Assistant

A document question-answering application built on retrieval-augmented generation (RAG). The goal: upload a PDF, ask questions in natural language, and receive answers grounded in the passages retrieved from that document.

> **Project status: work in progress.** PDF ingestion, semantic retrieval, grounded answer generation, and a Streamlit web interface are implemented and tested, and run locally. The application has **not been deployed**. See [Key capabilities](#key-capabilities).

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
| Streamlit interface: upload one PDF, ask questions, view answers and sources | Implemented |
| Deployment | Planned |

The first version targets a single, text-based PDF at a time.

## How it works

```
PDF -> validation + text extraction -> one Document per page -> chunks (nodes)
    -> embeddings -> in-memory vector index -> semantic retrieval -> ranked passages with page numbers
    -> context formatting -> prompt + chat model -> parsed answer with cited passages
```

All stages shown are implemented and are driven by the Streamlit interface.

Each page is chunked on its own, so a chunk never spans two pages and its page number is exact. A fact that straddles a page break is therefore split across two chunks.

## Architecture and technology stack

| Component | Role | State |
|---|---|---|
| [PyMuPDF](https://pymupdf.readthedocs.io/) | Extracts text from PDF pages | In use |
| [LlamaIndex](https://docs.llamaindex.ai/) | Document and node representation, chunking, in-memory vector index, retrieval | In use |
| [Sentence-transformers](https://www.sbert.net/) model `BAAI/bge-small-en-v1.5` | Local text embeddings (384 dimensions), no API key | In use |
| [LangChain](https://python.langchain.com/) (`langchain-core`, `langchain-groq`) | Prompt template, chat-model call, output parsing | In use |
| [Groq](https://groq.com/) chat model (default `openai/gpt-oss-20b`) | Writes the answer from the supplied passages | In use, requires an API key |
| [Streamlit](https://streamlit.io/) | Web interface: upload, question input, answer and source display, per-session state | In use |
| Python library (`answer_question`) | Connects retrieval and generation | In use |

The index lives in memory and no remote vector database is used.

Source layout:

```
src/intelligent_document_assistant/
    ingestion/    # PDF validation and text extraction
    retrieval/    # chunking, embeddings and indexing, retrieval
    generation/   # evidence formatting, prompt, chat model, answer parsing
    app/          # Streamlit interface and upload handling
evaluation/       # labelled dataset, synthetic corpus, metrics (not part of the package)
notebooks/        # evaluation notebook
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

The application reads these from the process environment and does not load `.env` itself; `uv run --env-file .env ...` applies the file for a single command. For a hosted deployment, supply the variables through the platform's environment or secrets configuration instead of a file.

Ingestion, indexing, and retrieval need no key. Without `GROQ_API_KEY` the app still indexes a PDF, and asking a question shows a setup message.

## Usage

### Web app

```powershell
uv run --env-file .env streamlit run src/intelligent_document_assistant/app/streamlit_app.py --server.address localhost
```

Open the printed local URL, then:

1. Upload a text-based PDF. The first upload in a server process also loads the embedding model, which takes some time (and downloads it on the very first run).
2. Ask a question. The answer is shown with the passages the model cited, each labelled with the uploaded file name and page number. Other retrieved passages are listed separately and labelled as not cited.
3. If the document does not answer the question, the app shows a "Not enough information in the document" notice instead of an answer.
4. Upload a different PDF, or use **Start over**, to replace the document. The previous index, answer, and sources are discarded.

The index is built once per upload and kept in the browser session's state, so further questions about the same PDF do not re-index it. File watching is disabled in `.streamlit/config.toml`, so restart the app after editing code.

### Python library

The components can also be used from Python:

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

This runs the fast offline tests, including the Streamlit interface tests (run with Streamlit's `AppTest` against the real app script). They generate their own small PDFs and use a deterministic stand-in embedding and a stub chat model, so they need no network access, model download, or API key.

```powershell
uv run pytest -m integration
```

This runs the opt-in tests that use real models: retrieval against the real embedding model (downloaded on first run), and answer generation against the live Groq API. The generation tests are skipped unless `GROQ_API_KEY` is set. To supply it from `.env`:

```powershell
uv run --env-file .env pytest -m integration
```

The live tests send small fictional test passages to Groq.

## Evaluation

`notebooks/evaluation.ipynb` measures the pipeline on a labelled dataset, scoring retrieval, answers, citations, and abstention separately. It calls the production functions; `evaluation/` only adds the dataset, the metrics, and the loops.

```powershell
# retrieval only: local, no API calls (the embedding model is downloaded on first use)
uv run --group notebook jupyter execute notebooks/evaluation.ipynb --output executed.ipynb
# adds live generation: sends the questions and retrieved passages to Groq
$env:EVAL_RUN_LIVE = "1"; uv run --env-file .env --group notebook jupyter execute notebooks/evaluation.ipynb --output executed.ipynb
```

The executed copy is saved as the git-ignored `notebooks/executed.ipynb` and raw results as `evaluation/results/latest.json`; the committed notebook has no outputs.

**Dataset.** 27 questions (18 answerable, 9 not) about a synthetic 10-page report on a fictional company, covering direct, paraphrased, multi-page, explicit-negative ("the company does not publish guidance"), absent, unrelated, and ambiguous questions. Page labels and expected facts are checked against the text extracted from the generated PDF by the test suite. The questions were written by the same author as the document and nothing was tuned on them, so the results are **exploratory**, not a held-out benchmark, and say little about real-world PDFs.

**Measured results** (top 5 chunks retrieved; generation with `openai/gpt-oss-20b` on 2026-10-09, two runs):

| Stage | Result |
|---|---|
| Retrieval, 18 answerable questions | Hit@1 0.83, Hit@3 0.94, Hit@5 0.94, MRR 0.89. One question (a paraphrase using "staff" where the text says "employed ... people") did not retrieve its page in the top 5. |
| Answers | 16 of the 18 answerable questions were answered, and all 16 were correct on manual review. Of the other two, one abstained because the evidence was not retrieved, and one abstained although the evidence was retrieved. |
| Citations | For all 16 answered questions the cited pages included a required page. A cited page is a reference to a passage the model was shown, not proof that the passage supports the claim. |
| Abstention (positive = abstained) | 8 correct abstentions, 2 answerable questions refused, 1 unanswerable question answered, 16 correct answers. The decisions were identical across both runs. |

The unanswerable question that was answered ("Which of the two terminals is better?") received a comparison that misstated one figure, which shows why a valid citation does not guarantee a supported claim. Each page of the test document is a single chunk, so the evaluation does not exercise chunk boundaries or pages split into several chunks.

## Limitations

- No OCR: scanned or image-only PDFs are rejected. If only some pages lack text, the remaining pages are extracted and `pages_with_text` is lower than `page_count`.
- Password-protected PDFs are rejected; supplying a password is not supported. PDFs that only restrict copying or printing open normally.
- Extraction returns the text PyMuPDF produces. There is no layout analysis, so tables, multi-column pages, and headers or footers are not specially handled.
- Chunks do not cross page boundaries. Facts split across a page break are retrieved only in part.
- Retrieval and answer quality have been measured only on a small synthetic document (see [Evaluation](#evaluation)). The embedding model is a small English model; non-English documents are not evaluated.
- Retrieval matches meaning loosely: a paraphrase that shares no words with the passage can miss it.
- Chunking parameters and `top_k` are initial defaults, not tuned values.
- "Not enough evidence" is decided by the language model following the prompt, not by a retrieval-score threshold, because similarity scores are not calibrated probabilities. Prompt instructions reduce but do not eliminate unsupported answers or susceptibility to instructions hidden in a document, and the abstention behaviour has been checked on only a few questions.
- Answers are only as good as retrieval: if the right passage is not among the top-k, the model cannot use it.
- Each question is answered independently; there is no conversation memory.
- The index is in memory and belongs to the browser session: refreshing the page or starting over discards it and the PDF must be uploaded again. There is no persistence.
- Only one document per session is supported. Uploads are subject to Streamlit's default 200 MB limit, and large PDFs take proportionally longer to index.
- The embedding model is shared by all sessions of a server process. Concurrent multi-user use has not been load-tested.
- Loading the embedding model needs free memory (the install is about 1 GB). On a 7 GB laptop with many applications open it failed intermittently with memory errors (a `MemoryError`, and Windows error 1455, "paging file is too small"). In a one-off local run on that machine, indexing 800 chunks (a 400-page PDF) took about 50 seconds; this is not a benchmark.

## Security and document handling

- Secrets go in a local `.env` file, which is git-ignored. Never commit real keys.
- `*.pdf` is git-ignored to avoid committing private documents; test PDFs are generated in code.
- Ingestion and indexing process documents in memory and run locally. Document text is not logged. Network access is limited to the one-time embedding model download and, for answer generation, the Groq API call.
- An uploaded PDF is written to a private temporary directory only while it is read, then deleted. The uploaded file name is shown for provenance but is never used as a filesystem path.
- Error messages never contain the document's text, its location on disk, or credentials. Unexpected errors are written to the server log, not to the page.
- Answers and passages are displayed as plain text, so markup produced by the model (for example a remote image link) is not rendered.
- The run command above binds the app to `localhost` so it is not exposed on your network.
- The app has no authentication or rate limiting: anyone who can reach a running instance can use the configured Groq key and send documents through it. Run it locally, or put it behind access control, before exposing it.
- Answer generation sends the question and the retrieved passages to the Groq API. Do not use documents you are not permitted to share with that provider. Ingestion, indexing, and retrieval stay local.
- The API key is read from the environment, is never logged, and is not included in error messages.

## Future improvements

- A hosted deployment.
- Retrieval-quality evaluation on realistic documents, and tuning of chunking and `top_k`.
- Possible later extensions: OCR for scanned documents, multiple documents, and advanced retrieval techniques.

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
