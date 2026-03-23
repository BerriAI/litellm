import pytest

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

MODEL = "claude-opus-4-5-20251101"


def test_empty_string_raises_bad_request_error():
    """Empty string reasoning_effort must raise BadRequestError (400), not ValueError (500)."""
    with pytest.raises(litellm.BadRequestError) as exc_info:
        AnthropicConfig._map_reasoning_effort(reasoning_effort="", model=MODEL)
    assert "Invalid reasoning_effort" in str(exc_info.value)


def test_invalid_string_raises_bad_request_error():
    """Unrecognized reasoning_effort value must raise BadRequestError with valid options listed."""
    with pytest.raises(litellm.BadRequestError) as exc_info:
        AnthropicConfig._map_reasoning_effort(reasoning_effort="ultra", model=MODEL)
    error_msg = str(exc_info.value)
    assert "Invalid reasoning_effort" in error_msg
    assert "low" in error_msg
    assert "medium" in error_msg
    assert "high" in error_msg


def test_none_returns_none():
    """None reasoning_effort must return None (disable thinking)."""
    result = AnthropicConfig._map_reasoning_effort(reasoning_effort=None, model=MODEL)
    assert result is None


def test_none_string_returns_none():
    """'none' reasoning_effort must return None (disable thinking)."""
    result = AnthropicConfig._map_reasoning_effort(reasoning_effort="none", model=MODEL)
    assert result is None


@pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal"])
def test_valid_values_return_thinking_param(effort: str):
    """Valid reasoning_effort values must return a non-None AnthropicThinkingParam."""
    result = AnthropicConfig._map_reasoning_effort(reasoning_effort=effort, model=MODEL)
    assert result is not None
