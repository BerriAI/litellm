"""Unit coverage for the poll-cap log parser (no proxy needed, always runs).

The e2e guard's verdict rests entirely on which seeded indices it reads out of the
proxy logs, so the parser is pinned here: it must extract exactly the indices of
its own prefix, ignore other prefixes, and not be fooled by substrings.
"""

from __future__ import annotations

from poll_cap import parse_skipped_indices

PREFIX = "pollcap-abc123"


def test_parses_indices_for_its_own_prefix() -> None:
    text = "\n".join(
        f'{{"message": "Skipping job {PREFIX}-{i:03d} because it is not a valid '
        'unified object id"}'
        for i in (0, 7, 49)
    )
    assert parse_skipped_indices(text, PREFIX) == frozenset({0, 7, 49})


def test_ignores_other_prefixes() -> None:
    text = f"Skipping job pollcap-other99-005 ...\nSkipping job {PREFIX}-012 ..."
    assert parse_skipped_indices(text, PREFIX) == frozenset({12})


def test_empty_when_prefix_absent() -> None:
    assert parse_skipped_indices("nothing here", PREFIX) == frozenset()


def test_deduplicates_repeated_cycles() -> None:
    text = f"{PREFIX}-003 ...\n{PREFIX}-003 ...\n{PREFIX}-004 ..."
    assert parse_skipped_indices(text, PREFIX) == frozenset({3, 4})
