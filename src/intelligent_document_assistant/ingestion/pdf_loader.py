"""Load a text-based PDF into a LlamaIndex ``Document``."""

import os
from pathlib import Path

import pymupdf
from llama_index.core import Document

# Form feed is the conventional plain-text page break. Blank pages stay as empty
# strings, so ``text.split(PAGE_SEPARATOR)[i]`` is always page ``i + 1``.
PAGE_SEPARATOR = "\f"


class PDFIngestionError(Exception):
    """Base class for all PDF ingestion failures."""


class PDFFileError(PDFIngestionError):
    """The path is missing, not a file, empty, not a PDF, or corrupt."""


class PDFPasswordProtectedError(PDFIngestionError):
    """The PDF needs a password to open."""


class PDFNoTextError(PDFIngestionError):
    """The PDF has no extractable text (for example a scanned, image-only PDF)."""


class PDFExtractionError(PDFIngestionError):
    """The PDF opened, but reading text from one of its pages failed."""


def load_pdf(path: str | os.PathLike[str]) -> Document:
    """Return the text of the PDF at ``path`` as a single ``Document``.

    Raises a ``PDFIngestionError`` subclass if the file cannot be read as a
    text-based PDF. Error messages name the file but never its directory or
    its contents.
    """
    pdf_path = Path(path)
    _check_is_readable_file(pdf_path)

    with _open_pdf(pdf_path) as pdf:
        pages = _extract_pages(pdf, pdf_path.name)

    if not any(pages):
        raise PDFNoTextError(
            f"PDF has no extractable text: {pdf_path.name}. It may be a scanned or "
            "image-only document; scanned PDFs (which need OCR) are not supported."
        )

    return Document(
        text=PAGE_SEPARATOR.join(pages),
        metadata={
            "file_name": pdf_path.name,
            "page_count": len(pages),
            "pages_with_text": sum(1 for text in pages if text),
        },
    )


def _check_is_readable_file(path: Path) -> None:
    if not path.exists():
        raise PDFFileError(f"File not found: {path.name}")
    if not path.is_file():
        raise PDFFileError(f"Path is not a file: {path.name}")
    if path.stat().st_size == 0:
        raise PDFFileError(f"File is empty: {path.name}")


def _extract_pages(pdf: pymupdf.Document, file_name: str) -> list[str]:
    pages: list[str] = []
    for page in pdf:
        # A PDF can open successfully yet still fail on an individual page.
        try:
            pages.append(page.get_text().strip())
        except Exception as exc:
            raise PDFExtractionError(
                f"Failed to extract text from page {page.number + 1} of {file_name}"
            ) from exc
    return pages


def _open_pdf(path: Path) -> pymupdf.Document:
    try:
        # Opening from bytes avoids a file handle that a failed open would leave locked on Windows.
        pdf = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
    except (RuntimeError, OSError) as exc:  # PyMuPDF errors subclass RuntimeError
        raise PDFFileError(f"File is not a valid PDF or cannot be read: {path.name}") from exc

    # PyMuPDF also opens images, XPS, EPUB and others; accept only real PDFs.
    if not pdf.is_pdf:
        pdf.close()
        raise PDFFileError(f"File is not a valid PDF: {path.name}")
    if pdf.needs_pass:
        pdf.close()
        raise PDFPasswordProtectedError(
            f"PDF is password-protected and cannot be read: {path.name}"
        )
    # A truncated PDF can open successfully with zero pages.
    if pdf.page_count == 0:
        pdf.close()
        raise PDFFileError(f"PDF has no pages (it may be corrupt or truncated): {path.name}")
    return pdf
