========================================
Improving recall (finding more matches)
========================================

Overview
========

The most common complaint about a freshly-built search index is not that it
returns *wrong* results — it's that it returns *too few*. A user searches for
``run`` and misses documents that say "running"; they search for ``car`` and
miss the ones that only mention "automobile"; they make a typo and get nothing
at all. This is the **recall** problem: relevant documents exist, but the query
never reaches them.

You do **not** need embeddings or a vector database to fix most of this. Whoosh
ships several complementary recall levers that work on ordinary inverted
indexes and run entirely in pure Python. This guide walks through them, from
cheapest to most involved:

#. :ref:`recall-stemming` — collapse word forms at index time.
#. :ref:`recall-variations` — expand word forms at query time.
#. :ref:`recall-fuzzy` — tolerate typos and spelling variants.
#. :ref:`recall-didyoumean` — suggest a correction when a query misses.
#. :ref:`recall-prf` — expand a query with terms from its own top results.

Each technique below is self-contained and independently useful; you can mix
and match. The examples share this tiny index:

.. code-block:: python

    from whoosh.fields import Schema, TEXT, ID
    from whoosh.analysis import StemmingAnalyzer
    from whoosh import index
    from whoosh.qparser import QueryParser, FuzzyTermPlugin
    from whoosh.query import Variations
    import tempfile

    schema = Schema(id=ID(stored=True),
                    body=TEXT(analyzer=StemmingAnalyzer(), stored=True))
    ix = index.create_in(tempfile.mkdtemp(), schema)
    w = ix.writer()
    docs = [
        ("1", "We are connecting the two networks that were disconnected."),
        ("2", "The connection dropped, so the client reconnected automatically."),
        ("3", "Automobiles and cars share many mechanical parts."),
        ("4", "Kubernetes orchestrates containers across clusters at scale."),
    ]
    for i, t in docs:
        w.add_document(id=i, body=t)
    w.commit()


.. _recall-stemming:

1. Stem at index time
=====================

A :class:`~whoosh.analysis.StemmingAnalyzer` reduces related word forms to a
common root *before* they hit the index, so a query for one form matches all of
them. Because the schema above already uses it, a search for ``connect`` finds
the documents that only say "connecting", "connection" and "reconnected":

.. code-block:: python

    with ix.searcher() as s:
        qp = QueryParser("body", ix.schema)
        print([h["id"] for h in s.search(qp.parse("connect"))])
        # ['1', '2']

Stemming is the single highest-leverage recall change you can make, and it is
essentially free at query time. See :doc:`stemming` for the trade-offs (it is
aggressive and language-specific) and for how to combine it with a
:class:`~whoosh.analysis.Filter` that preserves the original word.

.. note::

    Stemming is a heuristic, not a dictionary lookup. The Porter stemmer maps
    "running" to ``runn`` but "run" to ``run``, so those two do **not** match
    under stemming alone. When you need to bridge irregular forms, reach for
    :ref:`recall-variations` or :ref:`recall-fuzzy` below.


.. _recall-variations:

2. Expand variations at query time
===================================

If you did *not* stem at index time — for example on an existing index you
don't want to rebuild — you can expand a term into its morphological variations
at query time with :class:`whoosh.query.Variations`. It behaves like a
:class:`~whoosh.query.Term` query but matches any indexed word that shares the
same root:

.. code-block:: python

    with ix.searcher() as s:
        print([h["id"] for h in s.search(Variations("body", "run"))])
        # matches documents containing runner / running / runs

The cost is paid per query instead of once at index time, and you keep the
original words in the index (useful for exact-phrase search and highlighting).
Use ``Variations`` when you can't re-index; use stemming when you can.


.. _recall-fuzzy:

3. Tolerate typos with fuzzy terms
==================================

Add the :class:`~whoosh.qparser.FuzzyTermPlugin` and users can append ``~`` to a
term to match within a small edit distance. ``kubernets~1`` (one edit away)
still finds the document about "Kubernetes":

.. code-block:: python

    with ix.searcher() as s:
        qp = QueryParser("body", ix.schema)
        qp.add_plugin(FuzzyTermPlugin())
        print([h["id"] for h in s.search(qp.parse("kubernets~1"))])
        # ['4']

Keep the edit distance small (``~1``, at most ``~2``) — larger values match a
lot of unrelated words and slow queries down. Fuzzy matching is best reserved
for a fallback pass or for fields with lots of proper nouns and product names.


.. _recall-didyoumean:

4. Offer "did you mean ... ?"
=============================

When a query returns nothing, you can suggest a nearby indexed term instead of
showing an empty page. A :class:`~whoosh.spelling.Corrector` built from the
field's own vocabulary turns a typo into a real word:

.. code-block:: python

    with ix.searcher() as s:
        corr = s.corrector("body")
        print(corr.suggest("automobl", limit=3))   # ['automobil']
        print(corr.suggest("connecton", limit=3))  # ['connect']

(The suggestions are stemmed here because the field uses a stemming analyzer;
on a non-stemmed field you'd get the surface forms.) This pairs naturally with
fuzzy search: try the query as typed, and if the result set is empty, show the
top correction as a one-click "search instead for ..." link. See :doc:`spelling`
for word-list backends and tuning.


.. _recall-prf:

5. Expand a query from its own results (pseudo-relevance feedback)
==================================================================

Sometimes the vocabulary gap is conceptual, not morphological: the user types
``car`` but the best documents say "automobile". **Pseudo-relevance feedback**
(PRF) closes that gap without any external thesaurus. It assumes the top few
results are relevant, extracts their most distinctive terms, and uses those to
broaden the search. :meth:`Results.key_terms <whoosh.searching.Results.key_terms>`
does the extraction:

.. code-block:: python

    with ix.searcher() as s:
        qp = QueryParser("body", ix.schema)
        results = s.search(qp.parse("car"))
        expansion = [t for t, score in results.key_terms("body", numterms=5)]
        print(expansion)
        # ['automobil', 'car', 'mani', 'mechan', 'part']

Notice that ``automobil`` surfaced even though the user never typed it. You can
feed these terms back into a broadened :class:`~whoosh.query.Or` query to pull in
documents the original query missed. The closely related
:meth:`Searcher.more_like <whoosh.searching.Searcher.more_like>` does the whole
round-trip for you — given a document, it finds others like it — which is handy
for a "related results" panel.

PRF works best on medium-to-large corpora where the top results are genuinely
on-topic; on a tiny or noisy result set it can drift. Gate it behind a minimum
result count, or only expand when the original query returned few hits.


Putting it together
====================

These levers stack. A pragmatic recall pipeline looks like:

#. Index with a :class:`~whoosh.analysis.StemmingAnalyzer` (broad, cheap recall).
#. Run the query as typed.
#. If the result set is empty or thin, retry with :ref:`fuzzy terms
   <recall-fuzzy>` and surface a :ref:`did-you-mean <recall-didyoumean>`
   suggestion.
#. When you have a healthy result set, optionally apply :ref:`PRF <recall-prf>`
   to reach conceptually-related documents.

.. tip::

    Recall and precision trade off against each other — every technique here
    also lets in some noise. The only reliable way to tune them is to build an
    evaluation set from **real user queries** (not sentences copied out of your
    own documents, which flatters recall) and measure the effect of each lever
    on that set before shipping it.

See also :doc:`stemming`, :doc:`spelling`, :doc:`analysis`, and :doc:`parsing`.
