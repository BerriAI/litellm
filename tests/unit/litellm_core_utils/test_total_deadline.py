from collections.abc import Mapping

import pytest

from litellm.litellm_core_utils.total_deadline import InvalidTotalTimeout, TotalTimeout, total_timeout_from_request


@pytest.mark.parametrize(
    ("header", "body", "expected"),
    [
        (None, {}, None),
        (None, {"total_timeout": 10}, TotalTimeout(10.0, from_body=True)),
        (None, {"total_timeout": "2.5"}, TotalTimeout(2.5, from_body=True)),
        ("4", {}, TotalTimeout(4.0, from_body=False)),
        ("4", {"total_timeout": 10}, TotalTimeout(4.0, from_body=False)),
        ("4", {"total_timeout": "bad"}, TotalTimeout(4.0, from_body=False)),
    ],
    ids=["absent", "body", "body_numeric_string", "header", "header_wins", "header_wins_over_a_bad_body"],
)
def test_total_timeout_is_read_from_the_header_first_then_the_body(
    header: str | None, body: Mapping[str, object], expected: TotalTimeout | None
) -> None:
    assert total_timeout_from_request(header, body) == expected


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -1, "0", "-2", "nan", "inf", "-inf", float("nan"), 10**400, "ten", "", [10], {"s": 10}],
)
def test_a_total_timeout_that_is_not_a_positive_finite_number_of_seconds_is_invalid(value: object) -> None:
    assert isinstance(total_timeout_from_request(None, {"total_timeout": value}), InvalidTotalTimeout)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "ten", ""])
def test_an_invalid_total_timeout_header_is_invalid(value: str) -> None:
    assert isinstance(total_timeout_from_request(value, {}), InvalidTotalTimeout)
