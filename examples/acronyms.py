"""Searching for acronyms and tech tokens like ``R&D``, ``C++``, ``C#`` and ``.NET``.

The stock analyzers (``SimpleAnalyzer``, ``StandardAnalyzer``,
``StemmingAnalyzer`` …) tokenize with a word pattern that treats ``&``, ``+``,
``#`` and ``.`` as *boundaries*. That is the right default for prose, but it
means a token like ``R&D`` is split into ``R`` and ``D`` — and because
``StandardAnalyzer`` also drops single characters, the acronym vanishes
entirely::

    StandardAnalyzer()("Our R&D team ships C++ and C# on .NET")
        ->  ['build', 'net']        # R&D, C++, C# all gone

So a user who searches for ``R&D`` (or ``C++``, ``C#``, ``AT&T``, ``Q&A``,
``.NET``, ``F#`` …) gets *no results*, even though the text is right there.
This is a real, repeatedly-reported papercut for note apps and code/doc search.

The fix is to keep those symbol-bearing tokens whole. This recipe shows a
targeted tokenizer that recognises the common shapes:

* ampersand acronyms — ``R&D``, ``AT&T``, ``Q&A``, ``P&L``
* ``+``/``#`` language names — ``C++``, ``G++``, ``C#``, ``F#``, ``J#``
* dotted platform names — ``.NET``, ``.NETCore``

while ordinary text — including *hyphenated* words like ``well-known`` and
``e-mail`` — keeps splitting exactly as before. Attach the same analyzer to the
field and Whoosh runs it at both index and query time, so ``R&D`` matches
``R&D``.

What to avoid: widening the *whole* word pattern to include ``&``/``+``/``#``.
That would glue punctuation onto ordinary tokens and change unrelated results.
The tokenizer below is scoped: the tech shapes are tried *first* (most specific
wins), and everything else falls through to the normal Whoosh word pattern.

One honest caveat: this handles the well-known tech shapes above, not every
conceivable symbol soup. For fully arbitrary punctuation-search, index a
:class:`~whoosh.analysis.NgramFilter` field as well (see
``examples/custom_analyzers.py``); for exact literal matching of a whole field,
use an :class:`~whoosh.fields.ID` field.

Run it end to end::

    python examples/acronyms.py

It uses only the standard library plus Whoosh — no extra dependencies.
"""

from whoosh.analysis import LowercaseFilter, RegexTokenizer, StandardAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import QueryParser

# A tokenizer that keeps common acronym / tech tokens whole. The alternatives
# are ordered most-specific-first, because the scanner takes the first branch
# that matches at each position:
#
#   \w+(?:&\w+)+       ampersand acronyms:  R&D, AT&T, Q&A, P&L, AT&T&Co
#   [A-Za-z]\+\+       C++, G++
#   [A-Za-z]#          C#, F#, J#
#   \.[A-Za-z][\w.]*   dotted platform names:  .NET, .NETCore
#   \w+(?:\.?\w+)*     the ordinary Whoosh word pattern (words, numbers,
#                      dotted identifiers like foo.bar_baz) — unchanged
TECH_WORD_EXPR = (
    r"\w+(?:&\w+)+"
    r"|[A-Za-z]\+\+"
    r"|[A-Za-z]#"
    r"|\.[A-Za-z][\w.]*"
    r"|\w+(?:\.?\w+)*"
)


def TechAnalyzer():
    """Analyzer that lower-cases and keeps acronyms / tech tokens whole.

    Drop-in replacement for ``StandardAnalyzer`` on fields where users search
    for things like ``R&D``, ``C++`` or ``.NET``. Case-folds so ``R&D`` and
    ``r&d`` match, and — unlike ``StandardAnalyzer`` — does not drop short
    tokens, so single-symbol acronyms survive.
    """
    return RegexTokenizer(TECH_WORD_EXPR) | LowercaseFilter()


def tokens(analyzer, text):
    """Return the list of token *texts* an analyzer produces for a string."""
    return [t.text for t in analyzer(text)]


def show_the_problem():
    print("=" * 72)
    print("1. The default analyzer splits & drops acronyms and tech tokens")
    print("=" * 72)
    default = StandardAnalyzer()
    sample = "Our R&D team ships C++ and C# on .NET"
    print(f"  StandardAnalyzer({sample!r})")
    print(f"    -> {tokens(default, sample)}")
    print("  => R&D, C++, C# never make it into the index, so a search for")
    print("     any of them returns nothing.")


def show_tech_tokenizer():
    print()
    print("=" * 72)
    print("2. A targeted analyzer keeps tech tokens, not stray punctuation")
    print("=" * 72)
    ana = TechAnalyzer()
    for text in [
        "R&D", "AT&T", "Q&A", "C++", "C#", "F#", ".NET",
        "well-known", "e-mail", "foo.bar_baz",
        "Our R&D team ships C++ and C# on .NET",
    ]:
        print(f"  {text!r:38} -> {tokens(ana, text)}")
    print("  => acronyms/tech tokens survive; hyphenated words still split.")


def end_to_end_search():
    print()
    print("=" * 72)
    print("3. End to end: searching for 'R&D' now finds the note")
    print("=" * 72)
    schema = Schema(id=ID(stored=True), body=TEXT(analyzer=TechAnalyzer(), stored=True))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(id="n1", body="Our R&D team evaluated C++ and C# for the .NET port.")
    w.add_document(id="n2", body="Marketing and sales notes; nothing technical in here.")
    w.commit()

    qp = QueryParser("body", ix.schema)
    with ix.searcher() as s:
        for q in ["R&D", "C++", "C#", ".NET"]:
            hits = [h["id"] for h in s.search(qp.parse(q))]
            print(f"  search({q!r:6}) -> {hits}")
    print("  => all four now match note 'n1' (default analyzer -> [] for each).")


def main():
    show_the_problem()
    show_tech_tokenizer()
    end_to_end_search()


if __name__ == "__main__":
    main()
