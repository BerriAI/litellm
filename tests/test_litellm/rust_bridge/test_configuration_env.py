from __future__ import annotations

import pytest

from litellm.rust_bridge.configuration import (
    _parse_env_bool,  # pyright: ignore[reportPrivateUsage]  # directly test env parsing contract
)


@pytest.mark.parametrize(("value", "expected"), (("1", True), ("0", False), (" 1 ", True), (" 0 ", False)))
def test_parse_env_bool_accepts_binary_values(value: str, expected: bool) -> None:
    assert _parse_env_bool(value) is expected


def test_parse_env_bool_preserves_unset_value() -> None:
    assert _parse_env_bool(None) is None


@pytest.mark.parametrize("value", ("enabled", "true", "false", "yes", "no", "on", "off", ""))
def test_parse_env_bool_rejects_unknown_value(value: str) -> None:
    with pytest.raises(ValueError, match="must be '1' or '0'"):
        _parse_env_bool(value)
