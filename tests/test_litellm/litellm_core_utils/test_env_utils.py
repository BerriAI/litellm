"""Tests for litellm.litellm_core_utils.env_utils."""

from typing import Optional

import pytest

from litellm.litellm_core_utils.env_utils import get_env_bool


@pytest.mark.parametrize(
    ("raw", "default", "expected"),
    [
        (None, False, False),
        (None, True, True),
        ("", False, False),
        ("0", False, False),
        ("false", False, False),
        ("FALSE", False, False),
        ("no", False, False),
        ("1", False, True),
        ("true", False, True),
        ("TRUE", False, True),
        ("yes", False, True),
        ("on", False, True),
        ("  on  ", False, True),
    ],
)
def test_get_env_bool_parses_strings(
    monkeypatch: pytest.MonkeyPatch, raw: Optional[str], default: bool, expected: bool
) -> None:
    name = "_TEST_GET_ENV_BOOL_VAR"
    monkeypatch.delenv(name, raising=False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    assert get_env_bool(name, default) is expected


def test_get_env_bool_does_not_treat_string_false_as_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: bool(\"false\") is True in Python; get_env_bool must not do that."""
    name = "_TEST_GET_ENV_BOOL_FALSE_STRING"
    monkeypatch.setenv(name, "false")
    assert get_env_bool(name, False) is False
