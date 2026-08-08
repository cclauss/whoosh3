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

from typing import TYPE_CHECKING, Any

from whoosh import scoring
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.filedb.filestore import FileStorage, RamStorage
from whoosh.qparser import MultifieldParser, OrGroup

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Hit", "WhooshSearch", "make_whoosh_retriever"]


class Hit:
    """One retrieval result: a stored id, the chunk text, its metadata and score.

    A plain class (not a dataclass) so this module loads cleanly whether it is
    imported normally or via ``importlib`` in a test harness.
    """

    __slots__ = ("id", "text", "metadata", "score")

    def __init__(self, id: str, text: str, metadata: dict, score: float):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.score = score

    def __repr__(self) -> str:
        return f"Hit(id={self.id!r}, score={self.score:.3f}, text={self.text[:40]!r})"


class WhooshSearch:
    """A tiny BM25 search core over a set of text chunks.

    Depends only on Whoosh + the standard library. This is the piece you unit
    test; the LangChain adapter just maps :class:`Hit` -> ``Document``.
    """

    def __init__(self, ix):
        self.ix = ix

    @classmethod
    def _schema(cls) -> Schema:
        # `meta` is STORED-only (round-tripped as JSON) so arbitrary per-chunk
        # metadata survives without needing a field per key.
        return Schema(
            id=ID(stored=True, unique=True),
            text=TEXT(stored=True),
            meta=STORED,
        )

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
        path: str | None = None,
    ) -> WhooshSearch:
        """Build an index from parallel lists of texts (and optional ids/metadatas).

        Pass ``path`` to persist the index to a directory; omit it to keep the
        whole thing in memory (handy for tests and notebooks).
        """
        texts = list(texts)
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if not (len(texts) == len(ids) == len(metadatas)):
            raise ValueError("texts, ids and metadatas must have the same length")

        storage = FileStorage(path).create() if path else RamStorage()
        ix = storage.create_index(cls._schema())
        writer = ix.writer()
        for cid, text, meta in zip(ids, texts, metadatas):
            writer.add_document(id=str(cid), text=text, meta=dict(meta))
        writer.commit()
        return cls(ix)

    @classmethod
    def open_dir(cls, path: str) -> WhooshSearch:
        """Open an index previously built with ``from_texts(..., path=...)``."""
        return cls(FileStorage(path).open_index())

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """Return up to ``k`` hits ranked by Whoosh BM25 relevance.

        An OrGroup parser keeps recall high (a chunk matches if it contains
        *any* query term) while BM25 still floats the best matches to the top by
        rewarding rarer, more discriminating terms.
        """
        if not query or not query.strip():
            return []
        with self.ix.searcher(weighting=scoring.BM25F()) as s:
            parser = MultifieldParser(["text"], schema=self.ix.schema, group=OrGroup)
            q = parser.parse(query)
            return [
                Hit(
                    id=hit["id"],
                    text=hit["text"],
                    metadata=dict(hit.fields().get("meta") or {}),
                    score=float(hit.score),
                )
                for hit in s.search(q, limit=k)
            ]


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
