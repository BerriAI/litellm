import pytest

from litellm.router_rust.invoke.deadline import (
    capped_to_deadline,
    outlives_deadline,
    seconds_left,
    with_request_deadline,
)

NOW = 1000.0


def test_a_caller_controlled_timeout_pins_the_deadline_from_now() -> None:
    prepared = with_request_deadline({"model": "m", "timeout": 10, "client_side_timeout": True}, NOW)

    assert seconds_left(prepared, NOW) == 10
    assert seconds_left(prepared, NOW + 7.5) == 2.5
    assert prepared["timeout"] == 10


def test_request_timeout_counts_as_the_callers_budget_too() -> None:
    prepared = with_request_deadline({"model": "m", "request_timeout": 4.0, "client_side_timeout": True}, NOW)

    assert seconds_left(prepared, NOW + 1) == 3.0


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"model": "m", "timeout": 10},
        {"model": "m", "timeout": "10", "client_side_timeout": True},
        {"model": "m", "timeout": True, "client_side_timeout": True},
        {"model": "m", "client_side_timeout": True},
        {"model": "m", "timeout": 10, "client_side_timeout": "yes"},
    ],
    ids=["sdk timeout", "non-numeric timeout", "bool timeout", "no timeout", "forged marker"],
)
def test_no_deadline_without_a_numeric_caller_controlled_timeout(request_kwargs: dict[str, object]) -> None:
    prepared = with_request_deadline(request_kwargs, NOW)

    assert "request_deadline" not in prepared
    assert seconds_left(prepared, NOW) is None
    assert not outlives_deadline(prepared, NOW, 3600)
    assert capped_to_deadline(600, prepared, NOW + 3600) == 600


def test_a_sleep_that_ends_past_the_deadline_outlives_it() -> None:
    prepared = with_request_deadline({"model": "m", "timeout": 10, "client_side_timeout": True}, NOW)

    assert not outlives_deadline(prepared, NOW + 1, 6.5)
    assert outlives_deadline(prepared, NOW + 7, 6.5)
    assert outlives_deadline(prepared, NOW + 4, 6)
    assert not outlives_deadline(prepared, NOW + 9.9)
    assert outlives_deadline(prepared, NOW + 10)
    assert outlives_deadline(prepared, NOW + 11)


def test_an_attempts_timeout_is_cut_to_what_is_left() -> None:
    prepared = with_request_deadline({"model": "m", "timeout": 10, "client_side_timeout": True}, NOW)

    assert capped_to_deadline(10, prepared, NOW) == 10
    assert capped_to_deadline(10, prepared, NOW + 7.5) == 2.5
    assert capped_to_deadline(1, prepared, NOW + 7.5) == 1
    assert capped_to_deadline(10, prepared, NOW + 12) == 0
    assert capped_to_deadline(None, prepared, NOW + 7.5) is None
