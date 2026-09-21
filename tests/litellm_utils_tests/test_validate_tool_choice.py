import re
from typing import Final

import pytest

import litellm
from litellm.utils import validate_chat_completion_tool_choice

MODEL: Final = "anthropic/claude-haiku-4-5"


def test_validate_tool_choice_none():
    """Test that None is returned as-is."""
    result = validate_chat_completion_tool_choice(None, model=MODEL)
    assert result is None


def test_validate_tool_choice_string():
    """Test that string values are returned as-is."""
    assert validate_chat_completion_tool_choice("auto", model=MODEL) == "auto"
    assert validate_chat_completion_tool_choice("none", model=MODEL) == "none"
    assert validate_chat_completion_tool_choice("required", model=MODEL) == "required"


def test_validate_tool_choice_standard_dict():
    """Test standard OpenAI format with function."""
    tool_choice = {"type": "function", "function": {"name": "my_function"}}
    result = validate_chat_completion_tool_choice(tool_choice, model=MODEL)
    assert result == tool_choice


def test_validate_tool_choice_cursor_format():
    """Cursor IDE format {"type": "auto"} is unwrapped to the bare string."""
    assert validate_chat_completion_tool_choice({"type": "auto"}, model=MODEL) == "auto"
    assert validate_chat_completion_tool_choice({"type": "none"}, model=MODEL) == "none"
    assert validate_chat_completion_tool_choice({"type": "required"}, model=MODEL) == "required"


@pytest.mark.parametrize(
    "tool_choice",
    [
        {},
        {"type": "invalid"},
        {"type": "function"},
        {"name": "lookup_fruit"},
        {"type": "file_search"},
    ],
)
def test_validate_tool_choice_invalid_dict_is_a_400(tool_choice):
    """A dict shape chat completions cannot carry is the caller's mistake: a 400 that names the field, never a 500."""
    with pytest.raises(
        litellm.BadRequestError, match=f"Invalid tool choice, tool_choice={re.escape(str(tool_choice))}\\. Please ensure"
    ) as exc_info:
        validate_chat_completion_tool_choice(tool_choice, model=MODEL)
    assert exc_info.value.status_code == 400
    assert exc_info.value.model == MODEL


@pytest.mark.parametrize("tool_choice", [123, []])
def test_validate_tool_choice_invalid_type_is_a_400(tool_choice):
    """A non-str, non-dict tool_choice is rejected as a 400 that names the type it got."""
    with pytest.raises(
        litellm.BadRequestError, match=f"Got={re.escape(str(type(tool_choice)))}\\. Expecting str, or dict\\."
    ) as exc_info:
        validate_chat_completion_tool_choice(tool_choice, model=MODEL)
    assert exc_info.value.status_code == 400


def test_validate_tool_choice_without_model_is_still_a_400():
    """Callers that predate the model argument keep getting a 400, with an empty model on the error."""
    with pytest.raises(litellm.BadRequestError, match="Invalid tool choice") as exc_info:
        validate_chat_completion_tool_choice({"type": "bogus"})
    assert exc_info.value.status_code == 400
    assert exc_info.value.model == ""
