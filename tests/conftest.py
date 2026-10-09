"""Shared fixtures. Test PDFs are generated in code, not stored in the repo."""

from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import pymupdf
import pytest

MakePdf = Callable[..., Path]


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Factory for small PDFs in a temp dir.

    ``pages`` holds one text (or ``None`` for blank) per page. ``image_pages``
    adds an image to those page indexes; an image-only page mimics a scan.
    ``user_password`` encrypts the PDF.
    """

    def _make(
        pages: Sequence[str | None],
        *,
        image_pages: Collection[int] = (),
        name: str = "sample.pdf",
        user_password: str | None = None,
    ) -> Path:
        path = tmp_path / name
        with pymupdf.open() as doc:
            for index, text in enumerate(pages):
                page = doc.new_page()
                if text:
                    page.insert_text((72, 72), text)
                if index in image_pages:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
                    page.insert_image(pymupdf.Rect(72, 100, 172, 200), pixmap=pixmap)
            if user_password is None:
                doc.save(path)
            else:
                doc.save(
                    path,
                    encryption=pymupdf.PDF_ENCRYPT_AES_256,
                    user_pw=user_password,
                    owner_pw=user_password + "-owner",
                )
        return path

    return _make
