from __future__ import annotations

import pytest
from pydantic import ValidationError

from litellm.rust_bridge.configuration import (
    _parse_env_bool,  # pyright: ignore[reportPrivateUsage]  # directly test env parsing contract
)


@pytest.mark.parametrize("value", ("1", "true", "t", "yes", "y", "on", "TRUE", " yes "))
def test_parse_env_bool_accepts_standard_true_values(value: str) -> None:
    assert _parse_env_bool(value) is True


@pytest.mark.parametrize("value", ("0", "false", "f", "no", "n", "off", "FALSE", " no "))
def test_parse_env_bool_accepts_standard_false_values(value: str) -> None:
    assert _parse_env_bool(value) is False


def test_parse_env_bool_preserves_unset_value() -> None:
    assert _parse_env_bool(None) is None


def test_parse_env_bool_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        _parse_env_bool("enabled")
