import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.utils import validate_chat_completion_tool_choice


def test_validate_tool_choice_none():
    """Test that None is returned as-is."""
    result = validate_chat_completion_tool_choice(None)
    assert result is None


def test_validate_tool_choice_string():
    """Test that string values are returned as-is."""
    assert validate_chat_completion_tool_choice("auto") == "auto"
    assert validate_chat_completion_tool_choice("none") == "none"
    assert validate_chat_completion_tool_choice("required") == "required"


def test_validate_tool_choice_standard_dict():
    """Test standard OpenAI format with function."""
    tool_choice = {"type": "function", "function": {"name": "my_function"}}
    result = validate_chat_completion_tool_choice(tool_choice)
    assert result == tool_choice


def test_validate_tool_choice_cursor_format():
    """Test Cursor IDE format: {"type": "auto"} -> {"type": "auto"}."""
    assert validate_chat_completion_tool_choice({"type": "auto"}) == {"type": "auto"}
    assert validate_chat_completion_tool_choice({"type": "none"}) == {"type": "none"}
    assert validate_chat_completion_tool_choice({"type": "required"}) == {"type": "required"}


def test_validate_tool_choice_invalid_dict():
    """Test that invalid dict formats raise exceptions."""
    # Missing both type and function
    with pytest.raises(Exception) as exc_info:
        validate_chat_completion_tool_choice({})
    assert "Invalid tool choice" in str(exc_info.value)
    
    # Invalid type value
    with pytest.raises(Exception) as exc_info:
        validate_chat_completion_tool_choice({"type": "invalid"})
    assert "Invalid tool choice" in str(exc_info.value)
    
    # Has type but missing function when type is "function"
    with pytest.raises(Exception) as exc_info:
        validate_chat_completion_tool_choice({"type": "function"})
    assert "Invalid tool choice" in str(exc_info.value)


def test_validate_tool_choice_invalid_type():
    """Test that invalid types raise exceptions."""
    with pytest.raises(Exception) as exc_info:
        validate_chat_completion_tool_choice(123)
    assert "Got=<class 'int'>" in str(exc_info.value)
    
    with pytest.raises(Exception) as exc_info:
        validate_chat_completion_tool_choice([])
    assert "Got=<class 'list'>" in str(exc_info.value)