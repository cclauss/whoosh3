"""Indexing signed numbers (and why default text analysis drops the sign).

The stock analyzers (``SimpleAnalyzer``, ``StandardAnalyzer``,
``StemmingAnalyzer`` …) all tokenize with a word pattern that treats ``-`` and
``+`` as *boundaries*, because a hyphen is normally an intra-word separator
("well-known", "e-mail", "2024-01-02"). A useful side effect for prose, but it
means a leading sign is silently stripped from numeric text::

    RegexTokenizer()("balance -100 usd")  ->  ['balance', '100', 'usd']

so ``-100`` and ``100`` become the same indexed term. If the sign carries
meaning (prices, deltas, temperatures, offsets) you have two good options,
both shown here end to end:

1. **Use a NUMERIC field** for values you actually compare. This preserves the
   sign *and* unlocks range queries — it is the intended tool for numbers.
2. **Scope the sign to numbers only** in a custom tokenizer, so signed numbers
   survive while ordinary hyphenated words keep splitting as before.

What to avoid: widening the *whole* word pattern to ``[+-]?\\w+(\\.?\\w+)*``.
That fixes numbers at the cost of gluing a stray hyphen onto every token that
follows a split, including ordinary words: "well-known" -> ['well', '-known'],
"e-mail" -> ['e', '-mail']. Recipe 2 keeps hyphenated *words* intact.

One honest caveat about recipe 2: a purely *numeric* hyphen sequence such as an
ISO date ("2024-01-02") still has the sign attached to its later parts
(['2024', '-01', '-02']), because each part on its own looks like a signed
number. That is fine for free-text prices/deltas; for real dates use a
:class:`~whoosh.fields.DATETIME` field (see :doc:`dates`), which stores them as
sortable, range-queryable values rather than text.

Run it end to end::

    python examples/signed_numbers.py

It uses only the standard library plus Whoosh — no extra dependencies.
"""

from whoosh.analysis import RegexTokenizer
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import QueryParser

# A tokenizer that keeps a leading sign only when it is glued to a number.
# The first alternative matches signed integers/decimals; the second is the
# ordinary Whoosh word pattern, so hyphenated words split exactly as before.
SIGNED_NUMBER_TOKENIZER = RegexTokenizer(r"[+-]?\d+(\.\d+)?|\w+(\.?\w+)*")


def tokens(analyzer, text):
    """Return the list of token *texts* an analyzer produces for a string."""
    return [t.text for t in analyzer(text)]


def show_the_problem():
    print("=" * 70)
    print("1. The default word pattern drops leading +/- signs")
    print("=" * 70)
    default = RegexTokenizer()
    for text in ["-5", "+7", "balance -100 usd"]:
        print(f"  {text!r:18} -> {tokens(default, text)}")
    print("  => '-100' and '100' collapse to the same term.")


def show_targeted_tokenizer():
    print()
    print("=" * 70)
    print("2. A targeted tokenizer keeps signed numbers, not stray hyphens")
    print("=" * 70)
    for text in [
        "balance -100 usd",
        "-5.5",
        "+7",
        "well-known",
        "e-mail",
        "2024-01-02",
    ]:
        print(f"  {text!r:18} -> {tokens(SIGNED_NUMBER_TOKENIZER, text)}")
    print("  => signed numbers survive; hyphenated words still split.")
    print("     (numeric hyphen sequences like dates -> use a DATETIME field)")


def numeric_field_index():
    """Index a signed value in a NUMERIC field; return the index."""
    schema = Schema(id=ID(stored=True), bal=NUMERIC(signed=True, stored=True))
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(id="neg", bal=-100)
        w.add_document(id="pos", bal=100)
    return ix


def numeric_field_lookup(ix, expr):
    """Parse ``expr`` against the NUMERIC field and return matching ids."""
    with ix.searcher() as s:
        qp = QueryParser("bal", ix.schema)
        return sorted(h["id"] for h in s.search(qp.parse(expr)))


def text_field_index():
    """Index free text with the signed-number tokenizer; return the index."""
    schema = Schema(
        id=ID(stored=True),
        body=TEXT(analyzer=SIGNED_NUMBER_TOKENIZER, stored=True),
    )
    ix = RamStorage().create_index(schema)
    with ix.writer() as w:
        w.add_document(id="neg", body="balance -100 usd")
        w.add_document(id="pos", body="balance 100 usd")
    return ix


def text_field_lookup(ix, expr):
    """Parse ``expr`` against the TEXT field and return matching ids."""
    with ix.searcher() as s:
        qp = QueryParser("body", ix.schema)
        return sorted(h["id"] for h in s.search(qp.parse(expr)))


def show_real_indexes():
    print()
    print("=" * 70)
    print("3. Both approaches distinguish -100 from 100 in a real index")
    print("=" * 70)
    ix = numeric_field_index()
    print("  NUMERIC field:")
    print("    bal:-100        ->", numeric_field_lookup(ix, "bal:-100"))
    print("    bal:100         ->", numeric_field_lookup(ix, "bal:100"))
    print("    bal:[-200 to 0] ->", numeric_field_lookup(ix, "bal:[-200 to 0]"))

    tix = text_field_index()
    print("  TEXT field (signed-number tokenizer):")
    print("    body:-100       ->", text_field_lookup(tix, "-100"))
    print("    body:100        ->", text_field_lookup(tix, "100"))


if __name__ == "__main__":
    show_the_problem()
    show_targeted_tokenizer()
    show_real_indexes()
