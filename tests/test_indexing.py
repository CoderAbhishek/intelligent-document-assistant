import pytest
from llama_index.core import VectorStoreIndex

from intelligent_document_assistant.ingestion import load_pdf
from intelligent_document_assistant.retrieval import (
    NoIndexableContentError,
    build_index,
    chunk_document,
    index_document,
)


def test_build_index_stores_every_node(company_pdf, keyword_embed_model):
    nodes = chunk_document(load_pdf(company_pdf))

    index = build_index(nodes, keyword_embed_model)

    assert isinstance(index, VectorStoreIndex)
    assert len(index.docstore.docs) == len(nodes) == 4


def test_index_keeps_text_and_page_metadata_beside_the_vectors(company_pdf, keyword_embed_model):
    index = index_document(load_pdf(company_pdf), keyword_embed_model)

    stored = {node.metadata["page_number"]: node for node in index.docstore.docs.values()}
    assert "Rotterdam" in stored[2].text
    assert stored[2].metadata["file_name"] == "altair.pdf"


def test_each_chunk_is_embedded_once_at_build_time(company_pdf, keyword_embed_model):
    index_document(load_pdf(company_pdf), keyword_embed_model)

    assert keyword_embed_model.text_embed_calls == 4


def test_build_index_rejects_empty_node_list(keyword_embed_model):
    with pytest.raises(NoIndexableContentError):
        build_index([], keyword_embed_model)
