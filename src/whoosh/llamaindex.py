"""
Use Whoosh as a LlamaIndex retriever (lexical / BM25).

LlamaIndex leans on vector indices by default, but dense retrieval has a
well-known blind spot: it can quietly miss the *exact* tokens that matter most
(product SKUs, function names, error codes like ``ERR_2043``, gene symbols,
ticket IDs). A lexical BM25 retriever is the classic complement -- and Whoosh
gives you one in pure Python, with no server, no native wheels, and an index
that is just a folder on disk.

This module ships a drop-in ``llama_index.core.retrievers.BaseRetriever`` so you
can wire Whoosh into any LlamaIndex query engine or a
``QueryFusionRetriever`` (for hybrid search) exactly like any other retriever::

    from whoosh.llamaindex import WhooshSearch, make_whoosh_llamaindex_retriever

    core = WhooshSearch.from_texts(
        texts=["Whoosh is a pure-Python search library.", "BM25 ranks by term rarity."],
        ids=["a", "b"],
        metadatas=[{"src": "readme"}, {"src": "docs"}],
    )
    retriever = make_whoosh_llamaindex_retriever(core, k=4)
    nodes = retriever.retrieve("pure python search")   # -> list[NodeWithScore]

Design notes
------------
* All the real work lives in :class:`~whoosh.retrieval.WhooshSearch`, which
  depends only on Whoosh and the standard library, so it is fully unit-testable
  without LlamaIndex.
* The LlamaIndex adapter is intentionally thin and is built lazily by
  :func:`make_whoosh_llamaindex_retriever`, so importing this module never
  requires ``llama-index-core``. Install the optional dependency with
  ``pip install "whoosh3[llamaindex]"`` (or ``pip install llama-index-core``).
* For true *hybrid* search, drop this retriever and your vector retriever into
  LlamaIndex's ``QueryFusionRetriever``; it does Reciprocal Rank Fusion for you.
"""

from __future__ import annotations

# The BM25 core is shared with the other framework adapters and re-exported here
# so ``from whoosh.llamaindex import WhooshSearch, Hit`` works without importing
# the LangChain module.
from whoosh.retrieval import Hit, WhooshSearch

__all__ = ["Hit", "WhooshSearch", "make_whoosh_llamaindex_retriever"]


def make_whoosh_llamaindex_retriever(core: WhooshSearch, k: int = 4):
    """Build a LlamaIndex ``BaseRetriever`` backed by a :class:`WhooshSearch`.

    ``llama-index-core`` is imported lazily, so importing
    :mod:`whoosh.llamaindex` never requires it. Raises a clear ``ImportError``
    (not an obscure failure deep inside a query engine) if the optional
    dependency is missing.
    """
    try:
        from llama_index.core.retrievers import BaseRetriever  # noqa: PLC0415
        from llama_index.core.schema import (  # noqa: PLC0415
            NodeWithScore,
            QueryBundle,
            TextNode,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "make_whoosh_llamaindex_retriever needs llama-index-core. "
            'Install it with:  pip install "whoosh3[llamaindex]"'
        ) from exc

    class WhooshLlamaIndexRetriever(BaseRetriever):
        """A LlamaIndex retriever that ranks nodes with Whoosh BM25."""

        def __init__(self, core: WhooshSearch, k: int = 4):
            self._core = core
            self._k = k
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            return [
                NodeWithScore(
                    node=TextNode(
                        text=hit.text,
                        id_=hit.id,
                        metadata={"id": hit.id, **hit.metadata},
                    ),
                    score=hit.score,
                )
                for hit in self._core.search(query_bundle.query_str, self._k)
            ]

    return WhooshLlamaIndexRetriever(core=core, k=k)
