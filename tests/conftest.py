"""Shared fixtures. Test PDFs are generated in code, not stored in the repo."""

import math
import os
import re
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

# Must be set before huggingface_hub is imported: its Windows symlink warning
# would otherwise fail runs that use ``-W error``.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pymupdf
import pytest
from llama_index.core.embeddings import BaseEmbedding

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
                if text and page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), text) < 0:
                    raise ValueError("Test text does not fit on one page")
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


COMPANY_PAGES = [
    "Altair Robotics was founded in 1998 by two engineers working in a rented garage "
    "in Lisbon. The first prototype was a robotic arm for sorting parcels.",
    "The company headquarters is a converted shipyard warehouse in Rotterdam. "
    "The campus also contains a test hall where new machines are trialled.",
    "Altair sells two product lines: the Heron warehouse picking robot and the Wren "
    "inspection drone. Heron units are rented to logistics customers.",
    "In its latest year the company employed 240 people and opened a small office "
    "in Singapore to provide customer support in Asia.",
]


@pytest.fixture
def company_pdf(make_pdf) -> Path:
    """A four-page PDF about a fictional company, one distinct fact set per page."""
    return make_pdf(COMPANY_PAGES, name="altair.pdf")


class KeywordEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding over a fixed vocabulary.

    Lets index and retriever behaviour be tested offline; it only matches
    shared words, so it says nothing about semantic quality.
    """

    vocabulary: list[str]
    text_embed_calls: int = 0

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vector = [float(words.count(word)) for word in self.vocabulary]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        self.text_embed_calls += 1
        return self._embed(text)


@pytest.fixture
def keyword_embed_model() -> KeywordEmbedding:
    return KeywordEmbedding(
        vocabulary=[
            "founded", "garage", "lisbon", "headquarters", "rotterdam", "shipyard",
            "heron", "wren", "robot", "drone", "employed", "singapore", "people",
        ]
    )
