"""Tests for the first-class ``whoosh.llamaindex`` integration.

The pure-Whoosh :class:`~whoosh.retrieval.WhooshSearch` core is always exercised
so the "use Whoosh as a LlamaIndex retriever" story keeps working code behind
it. The thin ``BaseRetriever`` adapter is only tested where the optional
``llama-index-core`` dependency is installed (mirroring tests/test_mcp.py's
importorskip pattern).
"""

import pytest

from whoosh.llamaindex import WhooshSearch, make_whoosh_llamaindex_retriever
from whoosh.retrieval import WhooshSearch as CoreSearch


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


def test_module_reexports_shared_core():
    # whoosh.llamaindex should expose the same core as whoosh.retrieval,
    # without needing the LangChain module imported.
    assert WhooshSearch is CoreSearch


def test_search_finds_rare_literal_token(core):
    hits = core.search("ERR_2043 payment failure", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].id == "c"
    assert hits[0].score > 0


def test_llamaindex_adapter_returns_nodes(core):
    # Only runs where the optional 'llama-index-core' package is installed.
    pytest.importorskip("llama_index.core")

    retriever = make_whoosh_llamaindex_retriever(core, k=2)
    nodes = retriever.retrieve("ERR_2043 payment failure")
    assert nodes
    top = nodes[0]
    assert top.node.metadata["id"] == "c"
    assert top.score is not None
    assert isinstance(top.node.text, str)
