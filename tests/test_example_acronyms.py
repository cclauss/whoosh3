"""Smoke tests for examples/acronyms.py.

Keep the "searching for acronyms and tech tokens" cookbook recipe runnable, so
the docs claim (a targeted analyzer keeps ``R&D``, ``C++``, ``C#``, ``.NET``
whole while ordinary hyphenation still splits) always has working code behind
it.
"""

import importlib.util
import pathlib

import pytest

from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import QueryParser

_EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "examples" / "acronyms.py"


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("acronyms", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_analyzer_drops_acronyms(ex):
    # The problem the recipe exists to solve: StandardAnalyzer splits R&D/C++/C#
    # and drops the pieces, so none of them survive into the index.
    toks = ex.tokens(StandardAnalyzer(), "Our R&D team ships C++ and C# on .NET")
    for gone in ("r&d", "c++", "c#", ".net", "r", "d"):
        assert gone not in toks


def test_tech_analyzer_keeps_acronyms_and_langs(ex):
    ana = ex.TechAnalyzer()
    assert ex.tokens(ana, "R&D") == ["r&d"]
    assert ex.tokens(ana, "AT&T") == ["at&t"]
    assert ex.tokens(ana, "Q&A") == ["q&a"]
    assert ex.tokens(ana, "C++") == ["c++"]
    assert ex.tokens(ana, "C#") == ["c#"]
    assert ex.tokens(ana, "F#") == ["f#"]
    assert ex.tokens(ana, ".NET") == [".net"]


def test_tech_analyzer_does_not_regress_hyphenation(ex):
    # The whole point of scoping the pattern: ordinary hyphenated words and
    # numeric hyphen sequences still split exactly as before.
    ana = ex.TechAnalyzer()
    assert ex.tokens(ana, "well-known") == ["well", "known"]
    assert ex.tokens(ana, "e-mail") == ["e", "mail"]
    assert ex.tokens(ana, "foo.bar_baz") == ["foo.bar_baz"]


def test_tech_analyzer_in_a_sentence(ex):
    ana = ex.TechAnalyzer()
    assert ex.tokens(ana, "Our R&D team ships C++ and C# on .NET") == [
        "our", "r&d", "team", "ships", "c++", "and", "c#", "on", ".net",
    ]


def test_end_to_end_search_finds_acronyms(ex):
    schema = Schema(id=ID(stored=True), body=TEXT(analyzer=ex.TechAnalyzer()))
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(id="n1", body="Our R&D team evaluated C++ and C# for the .NET port.")
    w.add_document(id="n2", body="Marketing and sales notes; nothing technical in here.")
    w.commit()
    qp = QueryParser("body", ix.schema)
    with ix.searcher() as s:
        for q in ("R&D", "C++", "C#", ".NET"):
            hits = [h["id"] for h in s.search(qp.parse(q))]
            assert hits == ["n1"], f"{q!r} -> {hits}"


def test_default_analyzer_end_to_end_misses_acronym(ex):
    # Contrast: with the stock analyzer the same search returns nothing —
    # this is the bug the recipe fixes.
    schema = Schema(id=ID(stored=True), body=TEXT)
    ix = RamStorage().create_index(schema)
    w = ix.writer()
    w.add_document(id="n1", body="Our R&D team evaluated C++.")
    w.commit()
    qp = QueryParser("body", ix.schema)
    with ix.searcher() as s:
        assert [h["id"] for h in s.search(qp.parse("R&D"))] == []


def test_main_runs(ex):
    ex.main()
