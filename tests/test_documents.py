from pathlib import Path

import pytest

from intelligent_document_assistant.app import documents
from intelligent_document_assistant.app.documents import (
    build_document_index,
    display_name,
    file_digest,
)
from intelligent_document_assistant.ingestion import PDFFileError, PDFNoTextError
from intelligent_document_assistant.retrieval import retrieve_passages


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("My Report (final).pdf", "My Report (final).pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\scan.pdf", "scan.pdf"),
        ("bad\x00na\nme.pdf", "badname.pdf"),
        ("", "document.pdf"),
        ("   ", "document.pdf"),
    ],
)
def test_display_name_keeps_only_a_safe_basename(raw, expected):
    assert display_name(raw) == expected


def test_display_name_is_length_limited():
    assert len(display_name("x" * 500 + ".pdf")) <= 100


def test_digest_depends_on_content_not_name():
    assert file_digest(b"abc") == file_digest(b"abc")
    assert file_digest(b"abc") != file_digest(b"abd")


def test_original_file_name_is_the_source_shown_in_retrieval(company_pdf, keyword_embed_model):
    document = build_document_index(
        company_pdf.read_bytes(), "Annual Report (v2).pdf", keyword_embed_model
    )

    (passage,) = retrieve_passages(document.index, "Heron robot", top_k=1)

    assert document.file_name == "Annual Report (v2).pdf"
    assert passage.file_name == "Annual Report (v2).pdf"
    assert passage.page_number == 3


def test_indexed_document_describes_the_upload(company_pdf, keyword_embed_model):
    data = company_pdf.read_bytes()

    document = build_document_index(data, "altair.pdf", keyword_embed_model)

    assert document.page_count == 4
    assert document.digest == file_digest(data)


def test_hostile_file_name_cannot_choose_where_the_upload_is_written(
    company_pdf, keyword_embed_model, monkeypatch
):
    seen: list[Path] = []
    real_load_pdf = documents.load_pdf
    monkeypatch.setattr(documents, "load_pdf", lambda p: (seen.append(Path(p)), real_load_pdf(p))[1])

    document = build_document_index(
        company_pdf.read_bytes(), "../../outside.pdf", keyword_embed_model
    )

    assert document.file_name == "outside.pdf"
    assert seen[0].name == "upload.pdf"  # fixed name inside a private temp dir


def test_temp_directory_is_removed_after_success(company_pdf, keyword_embed_model, monkeypatch):
    seen: list[Path] = []
    real_load_pdf = documents.load_pdf
    monkeypatch.setattr(documents, "load_pdf", lambda p: (seen.append(Path(p)), real_load_pdf(p))[1])

    document = build_document_index(company_pdf.read_bytes(), "a.pdf", keyword_embed_model)

    assert not seen[0].parent.exists()
    # the index is self-contained: it still answers with the temp file gone
    assert retrieve_passages(document.index, "Rotterdam shipyard", top_k=1)[0].page_number == 2


def test_temp_directory_is_removed_when_ingestion_fails(keyword_embed_model, monkeypatch):
    seen: list[Path] = []
    real_load_pdf = documents.load_pdf

    def spying_load_pdf(path):
        seen.append(Path(path))
        return real_load_pdf(path)

    monkeypatch.setattr(documents, "load_pdf", spying_load_pdf)

    with pytest.raises(PDFFileError):
        build_document_index(b"this is not a pdf", "fake.pdf", keyword_embed_model)

    assert seen and not seen[0].parent.exists()


def test_image_only_pdf_is_reported_as_no_text(make_pdf, keyword_embed_model):
    scanned = make_pdf([None], image_pages={0})

    with pytest.raises(PDFNoTextError):
        build_document_index(scanned.read_bytes(), "scan.pdf", keyword_embed_model)
