import pytest
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


def test_sanitize_usage_obj_handles_null_tokens():
    """
    Tests that _sanitize_usage_obj correctly converts None values for token counts to 0.
    """
    response_json = {
        "choices": [],
        "usage": {"prompt_tokens": None, "completion_tokens": 50, "total_tokens": None},
    }

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert sanitized_json["usage"]["prompt_tokens"] == 0
    assert sanitized_json["usage"]["completion_tokens"] == 50  # Should remain unchanged
    assert sanitized_json["usage"]["total_tokens"] == 0


def test_sanitize_usage_obj_no_usage():
    """
    Tests that the sanitizer handles cases where the 'usage' object is missing.
    """
    response_json = {"choices": []}

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert "usage" not in sanitized_json  # Should not add a usage key


def test_sanitize_usage_obj_valid_usage():
    """
    Tests that the sanitizer does not modify a valid usage object.
    """
    response_json = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Create a copy to compare against
    original_json = response_json.copy()

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert sanitized_json == original_json  # The object should be unchanged
