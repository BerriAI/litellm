import math
from typing import Final

import pytest

from litellm.telemetry.histogram import Histogram

_BOUNDS: Final = (10.0, 100.0)


def _histogram_of(*values: float | None) -> Histogram:
    histogram: Final = Histogram.empty(_BOUNDS)
    for value in values:
        histogram.add(value)
    return histogram


@pytest.mark.parametrize(
    ("value", "expected_counts"),
    [
        (None, [0, 0, 0]),
        (0.0, [1, 0, 0]),
        (10.0, [1, 0, 0]),
        (10.5, [0, 1, 0]),
        (100.0, [0, 1, 0]),
        (100.5, [0, 0, 1]),
    ],
)
def test_a_value_lands_in_the_first_bucket_whose_bound_it_does_not_exceed(
    value: float | None, expected_counts: list[int]
) -> None:
    assert _histogram_of(value) == Histogram(bounds=_BOUNDS, counts=expected_counts, invalid=0)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, -math.inf])
def test_a_negative_or_non_finite_value_is_counted_as_invalid_instead_of_bucketed(value: float) -> None:
    assert _histogram_of(value, 5.0) == Histogram(bounds=_BOUNDS, counts=[1, 0, 0], invalid=1)


def test_add_histogram_adds_bucket_and_invalid_counts() -> None:
    total: Final = _histogram_of(5.0, -1.0)
    total.add_histogram(_histogram_of(5.0, 500.0, math.nan))
    assert total == Histogram(bounds=_BOUNDS, counts=[2, 0, 1], invalid=2)
