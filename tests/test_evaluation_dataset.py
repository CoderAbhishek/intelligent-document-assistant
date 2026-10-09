import pytest

from evaluation.corpus import PAGES, build_pdf
from evaluation.dataset import load_items
from intelligent_document_assistant.ingestion import PAGE_SEPARATOR, load_pdf

EXPECTED_CATEGORIES = {
    "direct",
    "paraphrase",
    "multi_page",
    "explicit_negative",
    "absent",
    "unrelated",
    "ambiguous",
}


@pytest.fixture(scope="module")
def extracted_pages(tmp_path_factory) -> list[str]:
    """Page texts as the production loader extracts them from the generated PDF."""
    document = load_pdf(build_pdf(tmp_path_factory.mktemp("corpus")))
    return [" ".join(page.split()).lower() for page in document.text.split(PAGE_SEPARATOR)]


@pytest.fixture(scope="module")
def items():
    return load_items()


def test_generated_pdf_extracts_to_the_source_pages(extracted_pages):
    assert len(extracted_pages) == len(PAGES) == 10
    assert extracted_pages == [" ".join(p.split()).lower() for p in PAGES]


def test_dataset_covers_every_category_and_has_unique_items(items):
    assert {item.category for item in items} == EXPECTED_CATEGORIES
    assert len({item.id for item in items}) == len(items)
    assert len({item.question for item in items}) == len(items)
    assert any(i.answerable for i in items) and any(not i.answerable for i in items)


def test_labels_are_internally_consistent(items):
    for item in items:
        if item.answerable:
            assert item.relevant_pages and item.answer_criteria, item.id
            assert all(1 <= page <= len(PAGES) for page in item.relevant_pages), item.id
        else:
            assert not item.relevant_pages and not item.answer_criteria, item.id
            assert item.absent_terms, item.id


def test_every_answerable_label_is_grounded_in_the_relevant_pages(items, extracted_pages):
    for item in (i for i in items if i.answerable):
        relevant_text = {p: extracted_pages[p - 1] for p in item.relevant_pages}
        for group in item.answer_criteria:
            assert any(
                alt.lower() in text for alt in group for text in relevant_text.values()
            ), f"{item.id}: none of {group} found on pages {item.relevant_pages}"
        for page, text in relevant_text.items():
            assert any(
                alt.lower() in text for group in item.answer_criteria for alt in group
            ), f"{item.id}: page {page} supports none of the criteria"


def test_unanswerable_items_ask_about_things_the_document_never_mentions(items, extracted_pages):
    corpus = " ".join(extracted_pages)
    for item in (i for i in items if not i.answerable):
        for term in item.absent_terms:
            assert term.lower() not in corpus, f"{item.id}: '{term}' occurs in the document"
