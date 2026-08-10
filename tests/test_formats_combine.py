"""Regression tests for ``Format.combine()`` across the format hierarchy.

``combine()`` is called when Whoosh merges the posting values for the same
term coming from different segments (e.g. during ``optimize()``). Each format
decodes the per-segment values, merges them, and re-encodes a single value.

These paths were historically untested and carried three latent bugs that were
surfaced by type-checking and fixed in gh#89 (3.40.0):

* ``Frequency.combine`` called a non-existent ``decode_value`` instead of
  ``decode_frequency`` — it raised ``AttributeError`` the moment two segments
  had to be merged for a frequency-format field.
* ``Characters.combine`` and ``CharacterBoosts.combine`` transposed the
  ``s[pos]`` lookup as ``pos[s]``, so merging character/boost postings raised
  ``TypeError`` instead of combining the start/end char spans.

This module pins the correct round-trip behaviour so those bugs cannot return.
"""

from whoosh.formats import (
    CharacterBoosts,
    Characters,
    Existence,
    Frequency,
    PositionBoosts,
    Positions,
)
from whoosh.system import pack_uint


def test_existence_combine_is_empty():
    assert Existence().combine([b"", b""]) == b""


def test_frequency_combine_sums_frequencies():
    fmt = Frequency()
    combined = fmt.combine([pack_uint(2), pack_uint(3), pack_uint(4)])
    # Merged posting frequency is the sum of the per-segment frequencies.
    assert fmt.decode_frequency(combined) == 9


def test_positions_combine_unions_positions():
    fmt = Positions()
    v1 = fmt.encode([1, 3, 5])
    v2 = fmt.encode([2, 3, 9])
    combined = fmt.combine([v1, v2])
    # Union of positions, sorted and de-duplicated.
    assert fmt.decode_positions(combined) == [1, 2, 3, 5, 9]


def test_characters_combine_merges_char_spans():
    fmt = Characters()
    v1 = fmt.encode([(1, 0, 4), (3, 10, 15)])
    v2 = fmt.encode([(1, 2, 6), (5, 20, 25)])
    combined = fmt.combine([v1, v2])
    # Position 1 appears in both: keep the widest span (min start, max end).
    # Positions 3 and 5 are carried through unchanged.
    assert fmt.decode_characters(combined) == [(1, 0, 6), (3, 10, 15), (5, 20, 25)]


def test_position_boosts_combine_sums_boosts():
    fmt = PositionBoosts()
    v1 = fmt.encode([(1, 0.5), (3, 1.0)])
    v2 = fmt.encode([(1, 0.25)])
    combined = fmt.combine([v1, v2])
    decoded = dict(fmt.decode_position_boosts(combined))
    assert decoded[1] == 0.75
    assert decoded[3] == 1.0


def test_character_boosts_combine_merges_spans_and_sums_boosts():
    fmt = CharacterBoosts()
    v1 = fmt.encode([(1, 0, 4, 0.5), (3, 10, 15, 1.0)])[0]
    v2 = fmt.encode([(1, 2, 6, 0.25)])[0]
    combined = fmt.combine([v1, v2])
    decoded = fmt.decode_character_boosts(combined)
    # Position 1: widest char span and summed boost; position 3 unchanged.
    assert decoded == [(1, 0, 6, 0.75), (3, 10, 15, 1.0)]
