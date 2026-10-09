from typing import Final

import pytest

from litellm.telemetry.histogram import Histogram

_BOUNDS: Final = (10.0, 100.0)


@pytest.mark.parametrize(
    ("value", "expected_counts"),
    [
        (None, (0, 0, 0)),
        (0.0, (1, 0, 0)),
        (10.0, (1, 0, 0)),
        (10.5, (0, 1, 0)),
        (100.0, (0, 1, 0)),
        (100.5, (0, 0, 1)),
    ],
)
def test_a_value_lands_in_the_first_bucket_whose_bound_it_does_not_exceed(
    value: float | None, expected_counts: tuple[int, ...]
) -> None:
    assert Histogram.of(_BOUNDS, value).counts == expected_counts


def test_merge_adds_bucket_counts() -> None:
    merged: Final = Histogram.of(_BOUNDS, 5.0).merge(Histogram.of(_BOUNDS, 5.0)).merge(Histogram.of(_BOUNDS, 500.0))
    assert merged == Histogram(bounds=_BOUNDS, counts=(2, 0, 1))
