"""
Use Whoosh as a LangChain retriever (lexical / BM25).

LangChain pipelines usually reach for a *vector* store, but dense retrieval has
a well-known blind spot: it can quietly miss the *exact* tokens that matter most
(product SKUs, function names, error codes like ``ERR_2043``, gene symbols,
ticket IDs). A lexical BM25 retriever is the classic complement -- and Whoosh
gives you one in pure Python, with no server, no native wheels, and an index
that is just a folder on disk.

This module ships a drop-in ``langchain_core.retrievers.BaseRetriever`` so you
can wire Whoosh into any LangChain chain, ``EnsembleRetriever`` (for hybrid
search), or LangGraph agent exactly like any other retriever::

    from whoosh.langchain import WhooshSearch, make_whoosh_retriever

    core = WhooshSearch.from_texts(
        texts=["Whoosh is a pure-Python search library.", "BM25 ranks by term rarity."],
        ids=["a", "b"],
        metadatas=[{"src": "readme"}, {"src": "docs"}],
    )
    retriever = make_whoosh_retriever(core, k=4)
    docs = retriever.invoke("pure python search")   # -> list[Document]

Design notes
------------
* All the real work lives in :class:`WhooshSearch`, which depends only on Whoosh
  and the standard library, so it is fully unit-testable without LangChain.
* The LangChain adapter is intentionally thin and is built lazily by
  :func:`make_whoosh_retriever`, so importing this module never requires
  ``langchain-core``. Install the optional dependency with
  ``pip install "whoosh3[langchain]"`` (or ``pip install langchain-core``).
* For true *hybrid* search, drop this retriever and your vector retriever into
  LangChain's ``EnsembleRetriever``; it does Reciprocal Rank Fusion for you.
"""

from __future__ import annotations

from typing import Any

# The BM25 core lives in :mod:`whoosh.retrieval` so it can be shared by every
# framework adapter (LangChain, LlamaIndex, ...) and unit-tested without any of
# them installed. Re-exported here so the long-standing
# ``from whoosh.langchain import WhooshSearch, Hit`` import keeps working.
from whoosh.retrieval import Hit, WhooshSearch

__all__ = ["Hit", "WhooshSearch", "make_whoosh_retriever"]


def make_whoosh_retriever(core: WhooshSearch, k: int = 4):
    """Build a LangChain ``BaseRetriever`` backed by a :class:`WhooshSearch`.

    ``langchain-core`` is imported lazily, so importing :mod:`whoosh.langchain`
    never requires it. Raises a clear ``ImportError`` (not an obscure failure
    deep inside a chain) if the optional dependency is missing.
    """
    try:
        from langchain_core.documents import Document  # noqa: PLC0415
        from langchain_core.retrievers import BaseRetriever  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "make_whoosh_retriever needs langchain-core. "
            'Install it with:  pip install "whoosh3[langchain]"'
        ) from exc

    class WhooshRetriever(BaseRetriever):
        """A LangChain retriever that ranks documents with Whoosh BM25."""

        core: Any  # a WhooshSearch instance (Any keeps pydantic v1/v2 happy)
        k: int = 4

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            return [
                Document(
                    page_content=hit.text,
                    metadata={"id": hit.id, "score": hit.score, **hit.metadata},
                )
                for hit in self.core.search(query, self.k)
            ]

    return WhooshRetriever(core=core, k=k)
