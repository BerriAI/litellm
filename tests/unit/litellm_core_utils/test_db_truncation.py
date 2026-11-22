from typing import Final

import pytest

from litellm.litellm_core_utils.db_truncation import TRUNCATION_MARKER, truncated_for_db, was_truncated_for_db

_LONG: Final = "Count the triples. " * 1000
_QUOTED_CUT: Final = truncated_for_db(_LONG, 100)


@pytest.mark.parametrize("max_length", [20, 99, 125, 126, 180, 2047, 2048, 2049, 10000])
def test_a_value_over_the_limit_keeps_its_ends_and_reads_as_cut_whatever_the_limit_was(max_length: int):
    cut: Final = truncated_for_db(_LONG, max_length)

    assert TRUNCATION_MARKER.split(cut) == [_LONG[: int(max_length * 0.35)], _LONG[-int(max_length * 0.65) :]]
    assert was_truncated_for_db(cut)


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
