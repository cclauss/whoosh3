"""Smoke tests for examples/signed_numbers.py.

Keep the "indexing signed numbers" cookbook recipe runnable, so the docs claim
(NUMERIC fields and a targeted tokenizer both preserve the sign) always has
working code behind it.
"""

import importlib.util
import pathlib

import pytest

from whoosh.analysis import RegexTokenizer

_EXAMPLE = (
    pathlib.Path(__file__).resolve().parent.parent / "examples" / "signed_numbers.py"
)


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("signed_numbers", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_tokenizer_drops_sign(ex):
    assert ex.tokens(RegexTokenizer(), "balance -100 usd") == [
        "balance",
        "100",
        "usd",
    ]


def test_targeted_tokenizer_keeps_signed_numbers(ex):
    assert ex.tokens(ex.SIGNED_NUMBER_TOKENIZER, "balance -100 usd") == [
        "balance",
        "-100",
        "usd",
    ]
    assert ex.tokens(ex.SIGNED_NUMBER_TOKENIZER, "-5.5") == ["-5.5"]
    assert ex.tokens(ex.SIGNED_NUMBER_TOKENIZER, "+7") == ["+7"]


def test_targeted_tokenizer_does_not_regress_hyphenated_words(ex):
    # The whole point: ordinary hyphenated words still split as before,
    # because their parts are not numeric and so never match the signed
    # alternative.
    assert ex.tokens(ex.SIGNED_NUMBER_TOKENIZER, "well-known") == ["well", "known"]
    assert ex.tokens(ex.SIGNED_NUMBER_TOKENIZER, "e-mail") == ["e", "mail"]


def test_targeted_tokenizer_date_caveat(ex):
    # Documented caveat: a purely numeric hyphen sequence (an ISO date) still
    # attaches the sign to its later parts, since each part looks like a
    # signed number. Use a DATETIME field for real dates.
    assert ex.tokens(ex.SIGNED_NUMBER_TOKENIZER, "2024-01-02") == [
        "2024",
        "-01",
        "-02",
    ]


def test_numeric_field_distinguishes_sign(ex):
    ix = ex.numeric_field_index()
    assert ex.numeric_field_lookup(ix, "bal:-100") == ["neg"]
    assert ex.numeric_field_lookup(ix, "bal:100") == ["pos"]
    assert ex.numeric_field_lookup(ix, "bal:[-200 to 0]") == ["neg"]


def test_text_field_with_signed_tokenizer_distinguishes_sign(ex):
    ix = ex.text_field_index()
    assert ex.text_field_lookup(ix, "-100") == ["neg"]
    assert ex.text_field_lookup(ix, "100") == ["pos"]
