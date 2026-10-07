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
    assert resolve_request_priority(requested, default_priority, drop_params=False) == expected
    assert resolve_request_priority(requested, default_priority, drop_params=True) == expected


@pytest.mark.parametrize("requested", ["1", [1], 1.5, True, {"level": 1}])
def test_a_non_integer_priority_is_rejected_unless_params_are_dropped(requested: object) -> None:
    rejected: Final = resolve_request_priority(requested, 3, drop_params=False)
    assert isinstance(rejected, InvalidPriority)
    assert rejected.value == requested
    assert rejected.message.startswith("priority must be an integer, got ")
    assert resolve_request_priority(requested, 3, drop_params=True) == 3
    assert resolve_request_priority(requested, None, drop_params=True) is None


def test_the_request_drop_params_flag_wins_over_the_router_default_and_the_global_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    assert request_drops_params({}, {}) is False
    assert request_drops_params({"drop_params": True}, {}) is True
    assert request_drops_params({"drop_params": "true"}, {}) is True
    assert request_drops_params({}, {"drop_params": True}) is True
    assert request_drops_params({"drop_params": False}, {"drop_params": True}) is False
    monkeypatch.setattr(litellm, "drop_params", True)
    assert request_drops_params({}, {}) is True
    assert request_drops_params({"drop_params": False}, {}) is False
    assert request_drops_params({}, {"drop_params": False}) is False
    assert request_drops_params({"drop_params": "not-a-flag"}, {}) is True
