import pytest
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import get_tokenizer

from intelligent_document_assistant.ingestion import PAGE_SEPARATOR
from intelligent_document_assistant.retrieval import (
    NoIndexableContentError,
    chunk_document,
    split_into_pages,
)


def make_document(*pages: str, file_name: str = "report.pdf") -> Document:
    return Document(
        text=PAGE_SEPARATOR.join(pages),
        metadata={"file_name": file_name, "page_count": len(pages), "pages_with_text": len(pages)},
    )


def long_page(label: str, sentences: int = 60) -> str:
    return " ".join(f"The {label} report covers item {i} in detail." for i in range(sentences))


def test_split_into_pages_creates_one_document_per_page():
    pages = split_into_pages(make_document("alpha", "bravo", "charlie"))

    assert [page.text for page in pages] == ["alpha", "bravo", "charlie"]
    assert [page.metadata["page_number"] for page in pages] == [1, 2, 3]


def test_page_metadata_is_only_file_name_and_page_number():
    (page,) = split_into_pages(make_document("alpha", file_name="annual.pdf"))

    assert page.metadata == {"file_name": "annual.pdf", "page_number": 1}


def test_blank_pages_are_skipped_but_numbering_is_preserved():
    pages = split_into_pages(make_document("first", "", "   ", "fourth"))

    assert [page.metadata["page_number"] for page in pages] == [1, 4]


def test_chunks_keep_file_name_and_page_number():
    nodes = chunk_document(make_document(long_page("alpha"), long_page("bravo")))

    assert {(n.metadata["file_name"], n.metadata["page_number"]) for n in nodes} == {
        ("report.pdf", 1),
        ("report.pdf", 2),
    }


def test_long_page_is_split_into_several_chunks_within_size_limit():
    tokenize = get_tokenizer()

    nodes = chunk_document(make_document(long_page("alpha")), chunk_size=64, chunk_overlap=8)

    assert len(nodes) > 1
    assert all(len(tokenize(n.get_content(MetadataMode.NONE))) <= 64 for n in nodes)


def test_a_chunk_never_mixes_text_from_two_pages():
    nodes = chunk_document(make_document(long_page("alpha"), long_page("bravo")), chunk_size=64)

    for node in nodes:
        expected_page = 1 if "alpha" in node.text else 2
        assert node.metadata["page_number"] == expected_page
        assert not ("alpha" in node.text and "bravo" in node.text)


def test_page_metadata_does_not_leak_into_embedding_text():
    (node,) = chunk_document(make_document("a short page about turbines"))

    assert node.get_content(MetadataMode.EMBED) == node.text
    assert node.metadata["page_number"] == 1  # still available to the application


@pytest.mark.parametrize("text", ["", "   \n ", PAGE_SEPARATOR, f" {PAGE_SEPARATOR}\n"])
def test_document_without_text_raises(text):
    with pytest.raises(NoIndexableContentError, match="no text"):
        chunk_document(Document(text=text, metadata={"file_name": "blank.pdf"}))


def test_very_long_file_name_does_not_break_chunking():
    file_name = "-".join(f"part{i}" for i in range(200)) + ".pdf"

    nodes = chunk_document(make_document(long_page("alpha"), file_name=file_name))

    assert nodes[0].metadata["file_name"] == file_name
