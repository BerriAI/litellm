from typing import Final

import pytest

from litellm.litellm_core_utils.db_truncation import (
    TRUNCATION_MARKER,
    is_truncation_of,
    truncated_for_db,
    was_truncated_for_db,
)

_LONG: Final = "Count the triples. " * 1000
_QUOTED_CUT: Final = truncated_for_db(_LONG, 100)
_NOTES_STORED_ROWS_CARRY: Final = (
    "",
    ". Truncation is a DB storage safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, "
    "etc.). To increase the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env.",
)


@pytest.mark.parametrize("max_length", [20, 99, 125, 126, 180, 2047, 2048, 2049, 10000])
def test_a_value_over_the_limit_keeps_its_ends_and_reads_as_cut_whatever_the_limit_was(max_length: int):
    cut: Final = truncated_for_db(_LONG, max_length)

    assert TRUNCATION_MARKER.split(cut) == [_LONG[: int(max_length * 0.35)], _LONG[-int(max_length * 0.65) :]]
    assert was_truncated_for_db(cut)


@pytest.mark.parametrize("note", _NOTES_STORED_ROWS_CARRY, ids=["no-note", "note"])
@pytest.mark.parametrize(("kept_start", "kept_end"), [(34, 64), (44, 81), (716, 1331)])
def test_a_value_an_earlier_release_stored_cut_still_reads_as_cut(kept_start: int, kept_end: int, note: str):
    skipped: Final = len(_LONG) - kept_start - kept_end
    stored: Final = f"{_LONG[:kept_start]}... (litellm_truncated skipped {skipped} chars{note}) ...{_LONG[-kept_end:]}"

    assert was_truncated_for_db(stored)
    assert is_truncation_of(stored, _LONG)


def test_only_the_marker_this_release_writes_splits_text_the_safeguard_cut():
    assert TRUNCATION_MARKER.split("Ask... (litellm_truncated skipped 9 chars) ...again") == [
        "Ask... (litellm_truncated skipped 9 chars) ...again"
    ]


@pytest.mark.parametrize(
    "marker",
    [
        pytest.param("... (litellm_truncated skipped  chars) ...", id="no-count"),
        pytest.param(f"... (litellm_truncated skipped  chars{_NOTES_STORED_ROWS_CARRY[1]}) ...", id="no-count-note"),
        pytest.param(f"... (litellm_truncated skipped 9 chars{_NOTES_STORED_ROWS_CARRY[1] * 2}) ...", id="two-notes"),
    ],
)
def test_a_marker_no_release_writes_reads_as_text(marker: str):
    assert TRUNCATION_MARKER.search(marker) is None
    assert not was_truncated_for_db(f"{_LONG[:34]}{marker}{_LONG[-64:]}")


@pytest.mark.parametrize(
    "whole",
    [
        pytest.param(f"-{_LONG}", id="another-start"),
        pytest.param(f"{_LONG}-", id="another-end"),
        pytest.param(f"{_LONG[:34]}#{_LONG[35:]}", id="last-kept-start-character"),
        pytest.param(f"{_LONG[:-65]}#{_LONG[-64:]}", id="first-kept-end-character"),
        pytest.param(f"{_LONG}{_LONG}", id="same-ends-but-longer"),
        pytest.param(_LONG[: len(_LONG) // 10], id="same-ends-but-shorter"),
        pytest.param(_LONG[:95], id="ends-that-overlap"),
        pytest.param("", id="empty"),
    ],
)
def test_a_cut_value_is_no_truncation_of_another_value(whole: str):
    assert is_truncation_of(truncated_for_db(_LONG, 100), _LONG)
    assert not is_truncation_of(truncated_for_db(_LONG, 100), whole)


@pytest.mark.parametrize("length", [101, 102, 1000])
def test_a_value_shorter_than_its_cut_text_is_what_storage_cut(length: int):
    whole: Final = "x" * length

    assert is_truncation_of(truncated_for_db(whole, 100), whole)


def test_a_value_shorter_than_what_storage_kept_is_not_what_storage_cut():
    assert not is_truncation_of(truncated_for_db("x" * 105, 100), "x" * 95)


def test_a_whole_value_is_no_truncation_of_itself():
    assert not is_truncation_of(_LONG, _LONG)


def test_a_value_that_quotes_a_cut_one_is_no_truncation():
    assert not is_truncation_of(f"-{_QUOTED_CUT}", f"-{_LONG}")


@pytest.mark.parametrize("value", ["", "Count.", _LONG])
def test_a_value_within_the_limit_is_stored_as_it_is_and_reads_as_whole(value: str):
    assert truncated_for_db(value, len(_LONG)) == value
    assert not was_truncated_for_db(value)


@pytest.mark.parametrize(
    "quoting",
    [
        pytest.param(f"The row holds {_QUOTED_CUT}", id="cut-at-the-end"),
        pytest.param(f"{_QUOTED_CUT} is what the row holds", id="cut-at-the-start"),
        pytest.param(f"The row holds {_QUOTED_CUT} and {_QUOTED_CUT}, twice", id="cut-in-the-middle"),
        pytest.param(f"-{_QUOTED_CUT}", id="one-character-more-before-the-cut"),
        pytest.param(_QUOTED_CUT[1:], id="one-character-less-before-the-cut"),
        pytest.param(_QUOTED_CUT[:-1], id="one-character-less-after-the-cut"),
    ],
)
def test_a_whole_value_that_quotes_a_cut_one_reads_as_whole(quoting: str):
    assert not was_truncated_for_db(quoting)


@pytest.mark.parametrize(
    "quoting",
    [
        pytest.param(f"{_QUOTED_CUT} {_LONG}", id="quoted-before-the-cut"),
        pytest.param(f"{_LONG} {_QUOTED_CUT}", id="quoted-after-the-cut"),
    ],
)
def test_a_cut_value_that_also_quotes_a_cut_one_reads_as_cut(quoting: str):
    assert was_truncated_for_db(truncated_for_db(quoting, 4096))
