"""Tests for the first-class ``whoosh.langchain`` integration.

The pure-Whoosh :class:`~whoosh.langchain.WhooshSearch` core is always
exercised so the "use Whoosh as a LangChain retriever" story keeps working code
behind it. The thin ``BaseRetriever`` adapter is only tested where the optional
``langchain-core`` dependency is installed (mirroring tests/test_mcp.py's
importorskip pattern).
"""

import pytest

from whoosh.langchain import WhooshSearch, make_whoosh_retriever


@pytest.fixture()
def core():
    return WhooshSearch.from_texts(
        texts=[
            "Whoosh is a pure-Python full-text search library.",
            "BM25 ranks documents by term rarity and length.",
            "The billing service rejected the charge with error code ERR_2043.",
        ],
        ids=["a", "b", "c"],
        metadatas=[{"src": "readme"}, {"src": "docs"}, {"src": "logs"}],
    )


def test_search_finds_rare_literal_token(core):
    hits = core.search("ERR_2043 payment failure", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].id == "c"
    assert hits[0].score > 0


def test_search_returns_metadata_and_text(core):
    hits = core.search("pure python search", k=1)
    assert hits[0].id == "a"
    assert hits[0].metadata == {"src": "readme"}
    assert "Whoosh" in hits[0].text


def test_empty_query_returns_no_hits(core):
    assert core.search("") == []
    assert core.search("   ") == []


def test_from_texts_defaults_ids_and_metadatas():
    c = WhooshSearch.from_texts(texts=["alpha term", "beta term"])
    hits = c.search("alpha", k=5)
    assert hits[0].id == "0"
    assert hits[0].metadata == {}


def test_from_texts_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        WhooshSearch.from_texts(texts=["one", "two"], ids=["only-one"])


def test_open_dir_round_trips(tmp_path):
    path = str(tmp_path / "ix")
    WhooshSearch.from_texts(
        texts=["persisted document about whoosh"], ids=["x"], path=path
    )
    reopened = WhooshSearch.open_dir(path)
    hits = reopened.search("whoosh", k=1)
    assert hits and hits[0].id == "x"


def test_langchain_adapter_returns_documents(core):
    # Only runs where the optional 'langchain-core' package is installed.
    pytest.importorskip("langchain_core")

    retriever = make_whoosh_retriever(core, k=2)
    docs = retriever.invoke("ERR_2043 payment failure")
    assert docs
    assert docs[0].metadata["id"] == "c"
    assert "score" in docs[0].metadata
    assert isinstance(docs[0].page_content, str)
