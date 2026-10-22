from typing import Final

import pytest

import litellm
from litellm.router_utils.request_priority import (
    InvalidPriority,
    parse_default_priority,
    request_drops_params,
    resolve_request_priority,
    unqueueable_priority_message,
)


@pytest.mark.parametrize(
    ("requested", "default_priority", "expected"),
    [
        (None, None, None),
        (None, 3, 3),
        (2, 3, 2),
        (0, 3, 0),
        (0, None, 0),
        (255, None, 255),
        ("1", 3, 1),
        (" 2 ", None, 2),
        ("-1", 3, -1),
        (1.0, 3, 1),
        (True, None, 1),
    ],
)
def test_an_absent_or_null_priority_takes_the_default_and_an_integer_like_value_wins(
    requested: object, default_priority: int | None, expected: int | None
) -> None:
    assert resolve_request_priority(requested, default_priority, drops_params=lambda: False) == expected
    assert resolve_request_priority(requested, default_priority, drops_params=lambda: True) == expected


@pytest.mark.parametrize("requested", ["high", "", [1], 1.5, {"level": 1}])
def test_a_non_integer_priority_is_rejected_unless_params_are_dropped(requested: object) -> None:
    rejected: Final = resolve_request_priority(requested, 3, drops_params=lambda: False)
    assert isinstance(rejected, InvalidPriority)
    assert rejected.value == requested
    assert rejected.message.startswith("priority must be an integer, got ")
    assert resolve_request_priority(requested, 3, drops_params=lambda: True) == 3
    assert resolve_request_priority(requested, None, drops_params=lambda: True) is None


def test_the_drop_params_decision_is_consulted_only_for_a_non_integer_priority() -> None:
    def _never() -> bool:
        raise AssertionError("drop_params was consulted")

    assert resolve_request_priority(None, 3, drops_params=_never) == 3
    assert resolve_request_priority(2, 3, drops_params=_never) == 2
    assert resolve_request_priority(0, None, drops_params=_never) == 0
    assert resolve_request_priority("1", 3, drops_params=_never) == 1


def test_the_global_drop_params_wins_then_the_request_then_the_router_defaults_then_the_deployments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    assert request_drops_params({}, {}, ()) is False
    assert request_drops_params({"drop_params": True}, {}, ()) is True
    assert request_drops_params({"drop_params": "true"}, {}, ()) is True
    assert request_drops_params({}, {"drop_params": True}, ()) is True
    assert request_drops_params({"drop_params": False}, {"drop_params": True}, ()) is False
    assert request_drops_params({}, {}, ({"drop_params": True},)) is True
    assert request_drops_params({}, {}, ({"drop_params": True}, {"model": "openai/gpt-5.4-nano"})) is True
    assert request_drops_params({}, {}, ({"drop_params": False}, {"model": "openai/gpt-5.4-nano"})) is False
    assert request_drops_params({}, {"drop_params": False}, ({"drop_params": True},)) is False
    monkeypatch.setattr(litellm, "drop_params", True)
    assert request_drops_params({}, {}, ()) is True
    assert request_drops_params({}, {}, ({"model": "openai/gpt-5.4-nano"},)) is True
    assert request_drops_params({"drop_params": False}, {}, ()) is True
    assert request_drops_params({}, {"drop_params": False}, ()) is True
    assert request_drops_params({}, {}, ({"drop_params": False},)) is True
    assert request_drops_params({"drop_params": "not-a-flag"}, {}, ()) is True


def test_a_priority_named_in_additional_drop_params_at_any_level_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    by_name: Final = {"additional_drop_params": ["priority"]}
    assert request_drops_params(by_name, {}, ()) is True
    assert request_drops_params({**by_name, "drop_params": False}, {}, ()) is True
    assert request_drops_params({}, by_name, ()) is True
    assert request_drops_params({}, {"drop_params": False}, (by_name,)) is True
    assert request_drops_params({"additional_drop_params": ["seed"]}, {}, ()) is False
    assert request_drops_params({}, {}, (params for params in ({"drop_params": True},))) is True


def test_a_request_it_cannot_queue_is_told_why() -> None:
    assert "priority is required" in unqueueable_priority_message(None)
    assert unqueueable_priority_message("high").startswith("priority must be an integer, got str 'high'")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, None), (3, 3), (0, 0), ("3", 3), (" 12 ", 12), ("0", 0), ("-1", -1), (2.0, 2)],
    ids=["unset", "int", "zero", "digit-string", "padded-digit-string", "zero-string", "negative-string", "whole-float"],
)
def test_the_default_priority_is_taken_as_an_integer_or_an_integer_like_value(
    configured: object, expected: int | None
) -> None:
    assert parse_default_priority(configured) == expected


@pytest.mark.parametrize("configured", ["high", "", "1.5", 1.5, [3], {"level": 3}, True, False])
def test_any_other_default_priority_is_invalid(configured: object) -> None:
    parsed: Final = parse_default_priority(configured)
    assert isinstance(parsed, InvalidPriority)
    assert parsed.value == configured
    assert parsed.message.startswith("priority must be an integer, got ")
