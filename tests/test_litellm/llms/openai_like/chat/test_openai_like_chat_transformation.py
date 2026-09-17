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


def test_sanitize_usage_obj_normalizes_cache_tokens():
    """
    Tests that _sanitize_usage_obj maps cache_read_input_tokens and cache_creation_input_tokens
    into prompt_tokens_details for OpenAI compatibility and accurate cost attribution.
    """
    response_json = {
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 30,
        },
    }

    sanitized = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    assert "prompt_tokens_details" in sanitized["usage"]
    assert sanitized["usage"]["prompt_tokens_details"]["cached_tokens"] == 70
    assert sanitized["usage"]["prompt_tokens_details"]["cache_write_tokens"] == 30


def test_openai_like_reasoning_effort_supported():
    """
    Tests that get_supported_openai_params includes 'reasoning_effort' when model supports reasoning.
    """
    import litellm

    config = OpenAILikeChatConfig()

    litellm.model_cost["openai_like/custom-r1"] = {"supports_reasoning": True}
    supported = config.get_supported_openai_params("custom-r1")
    assert "reasoning_effort" in supported

    litellm.model_cost["openai_like/custom-standard"] = {"supports_reasoning": False}
    supported_standard = config.get_supported_openai_params("custom-standard")
    assert "reasoning_effort" not in supported_standard


def test_transform_response_preserves_cache_tokens():
    """
    Tests that _transform_response maps cache_read_input_tokens and cache_creation_input_tokens
    to the ModelResponse usage object.
    """
    from unittest.mock import MagicMock

    import httpx

    from litellm.types.utils import ModelResponse

    raw_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        },
    }
    http_resp = httpx.Response(200, json=raw_response, request=httpx.Request("POST", "https://api.example.com"))
    config = OpenAILikeChatConfig()
    logging_obj = MagicMock()

    res = config.transform_response(
        model="deepseek-chat",
        raw_response=http_resp,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert res.usage.cache_read_input_tokens == 80
    assert res.usage.cache_creation_input_tokens == 20
    assert res.usage.prompt_tokens_details.cached_tokens == 80
    assert res.usage.prompt_tokens_details.cache_write_tokens == 20
