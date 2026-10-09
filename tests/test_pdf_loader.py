import pymupdf
import pytest
from llama_index.core import Document

from intelligent_document_assistant.ingestion import (
    PAGE_SEPARATOR,
    PDFExtractionError,
    PDFFileError,
    PDFIngestionError,
    PDFNoTextError,
    PDFPasswordProtectedError,
    load_pdf,
)


def test_returns_llamaindex_document(make_pdf):
    path = make_pdf(["alpha", "bravo"])

    document = load_pdf(path)

    assert isinstance(document, Document)


def test_pages_are_extracted_in_order(make_pdf):
    path = make_pdf(["first page", "second page", "third page"])

    document = load_pdf(path)

    pages = document.text.split(PAGE_SEPARATOR)
    assert pages == ["first page", "second page", "third page"]


def test_metadata_describes_source_file(make_pdf):
    path = make_pdf(["one", "two", "three"], name="report.pdf")

    document = load_pdf(path)

    assert document.metadata == {
        "file_name": "report.pdf",
        "page_count": 3,
        "pages_with_text": 3,
    }


def test_accepts_str_paths(make_pdf):
    path = make_pdf(["alpha"])

    document = load_pdf(str(path))

    assert document.text == "alpha"


def test_missing_path_raises_file_error(tmp_path):
    with pytest.raises(PDFFileError, match="not found"):
        load_pdf(tmp_path / "does_not_exist.pdf")


def test_directory_raises_file_error(tmp_path):
    with pytest.raises(PDFFileError, match="not a file"):
        load_pdf(tmp_path)


def test_empty_file_raises_file_error(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(PDFFileError, match="empty"):
        load_pdf(path)


def test_non_pdf_content_raises_file_error(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_text("this is plain text, not a PDF")

    with pytest.raises(PDFFileError, match="valid PDF"):
        load_pdf(path)


def test_other_document_format_is_not_accepted_as_pdf(tmp_path):
    # PyMuPDF opens images too; they must still be rejected.
    path = tmp_path / "scan.pdf"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10))
    path.write_bytes(pixmap.tobytes("png"))

    with pytest.raises(PDFFileError):
        load_pdf(path)


def test_truncated_pdf_with_no_pages_raises_file_error(tmp_path):
    # PyMuPDF opens this file but reports zero pages.
    path = tmp_path / "truncated.pdf"
    path.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\n")

    with pytest.raises(PDFFileError, match="no pages"):
        load_pdf(path)


def test_password_protected_pdf_raises_dedicated_error(make_pdf):
    path = make_pdf(["secret"], user_password="hunter2")

    with pytest.raises(PDFPasswordProtectedError, match="password"):
        load_pdf(path)


def test_error_message_names_file_but_not_its_directory(tmp_path):
    path = tmp_path / "private_dir" / "missing.pdf"

    with pytest.raises(PDFFileError) as excinfo:
        load_pdf(path)

    assert "missing.pdf" in str(excinfo.value)
    assert "private_dir" not in str(excinfo.value)


def test_all_specific_errors_share_one_base_class():
    for error in (
        PDFFileError,
        PDFPasswordProtectedError,
        PDFNoTextError,
        PDFExtractionError,
    ):
        assert issubclass(error, PDFIngestionError)


def test_mixed_pages_keep_positions_and_count_pages_with_text(make_pdf):
    # Page 2 is image-only, like a scanned page.
    path = make_pdf(["intro", None, "conclusion"], image_pages={1})

    document = load_pdf(path)

    assert document.text.split(PAGE_SEPARATOR) == ["intro", "", "conclusion"]
    assert document.metadata["page_count"] == 3
    assert document.metadata["pages_with_text"] == 2


def test_blank_pdf_raises_no_text_error(make_pdf):
    path = make_pdf([None, None])

    with pytest.raises(PDFNoTextError, match="no extractable text"):
        load_pdf(path)


def test_image_only_pdf_raises_no_text_error_and_mentions_scanning(make_pdf):
    path = make_pdf([None, None], image_pages={0, 1})

    with pytest.raises(PDFNoTextError, match="scanned"):
        load_pdf(path)


def test_owner_password_only_pdf_is_readable(tmp_path):
    # An owner password restricts permissions but does not block opening.
    path = tmp_path / "restricted.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "readable")
        doc.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="",
        )

    assert load_pdf(path).text == "readable"


def test_extraction_failure_on_a_page_raises_extraction_error(make_pdf, monkeypatch):
    path = make_pdf(["page one", "page two", "page three"])
    real_get_text = pymupdf.Page.get_text

    def flaky_get_text(self, *args, **kwargs):
        if self.number == 1:
            raise RuntimeError("simulated MuPDF failure")
        return real_get_text(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", flaky_get_text)

    with pytest.raises(PDFExtractionError, match="page 2") as excinfo:
        load_pdf(path)

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "page one" not in str(excinfo.value)


def test_failed_load_does_not_keep_the_file_locked(tmp_path):
    # On Windows an open handle blocks deletion; callers delete uploads right after a failure.
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")

    with pytest.raises(PDFFileError) as excinfo:
        load_pdf(path)

    path.unlink()
    assert excinfo.value is not None
