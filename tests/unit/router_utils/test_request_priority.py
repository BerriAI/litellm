from typing import Final

import pytest

import litellm
from litellm.router_utils.request_priority import InvalidPriority, request_drops_params, resolve_request_priority


@pytest.mark.parametrize(
    ("requested", "default_priority", "expected"),
    [
        (None, None, None),
        (None, 3, 3),
        (2, 3, 2),
        (0, 3, 0),
        (0, None, 0),
        (255, None, 255),
    ],
)
def test_an_absent_or_null_priority_takes_the_default_and_an_integer_wins(
    requested: object, default_priority: int | None, expected: int | None
) -> None:
    assert resolve_request_priority(requested, default_priority, drops_params=lambda: False) == expected
    assert resolve_request_priority(requested, default_priority, drops_params=lambda: True) == expected


@pytest.mark.parametrize("requested", ["1", [1], 1.5, True, {"level": 1}])
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


def test_drop_params_is_read_from_the_request_then_the_router_defaults_then_the_deployments_then_the_global_flag(
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
    assert request_drops_params({"drop_params": False}, {}, ()) is False
    assert request_drops_params({}, {"drop_params": False}, ()) is False
    assert request_drops_params({}, {}, ({"drop_params": False},)) is False
    assert request_drops_params({"drop_params": "not-a-flag"}, {}, ()) is True
