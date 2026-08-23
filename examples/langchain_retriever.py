"""
Runnable demo: use Whoosh as a LangChain retriever (lexical / BM25).

The integration itself is shipped as a first-class module,
:mod:`whoosh.langchain`, so you don't need this file to use it -- just::

    pip install "whoosh3[langchain]"

    from whoosh.langchain import WhooshSearch, make_whoosh_retriever

This script is only a small, self-contained demonstration you can run to see the
lexical retriever beat dense retrieval's blind spot on a rare literal token
(``ERR_2043``). See ``examples/rag_retriever.py`` for why *hybrid* search (drop
this retriever + your vector retriever into LangChain's ``EnsembleRetriever``)
wins, with a dependency-free RRF implementation.

Run it:  python examples/langchain_retriever.py
"""

from __future__ import annotations

from whoosh.langchain import WhooshSearch, make_whoosh_retriever

_DOCS = [
    (
        "d1",
        "Whoosh is a fast, featureful pure-Python full-text indexing and search library.",
    ),
    ("d2", "BM25 scores documents by term frequency and inverse document frequency."),
    (
        "d3",
        "Dense vector retrieval matches on meaning but can miss rare literal tokens.",
    ),
    ("d4", "The billing service rejected the charge with error code ERR_2043."),
    ("d5", "Mitochondria are the powerhouse of the cell and generate ATP."),
]


def _demo() -> None:
    core = WhooshSearch.from_texts(
        texts=[t for _, t in _DOCS],
        ids=[i for i, _ in _DOCS],
        metadatas=[{"src": i} for i, _ in _DOCS],
    )

    query = "ERR_2043 payment failure"
    print(f"Query: {query!r}\n")

    print("WhooshSearch core (no LangChain needed):")
    for hit in core.search(query, k=3):
        print(f"  {hit.id}  score={hit.score:5.2f}  {hit.text[:60]}")

    print("\nAs a LangChain retriever:")
    try:
        retriever = make_whoosh_retriever(core, k=3)
    except ImportError as exc:
        print(f"  (skipped) {exc}")
        return
    for doc in retriever.invoke(query):
        print(
            f"  {doc.metadata['id']}  score={doc.metadata['score']:5.2f}  {doc.page_content[:60]}"
        )


if __name__ == "__main__":
    _demo()
