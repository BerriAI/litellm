import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm import supports_reasoning
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message


def test_handle_message_content_with_tool_calls():
    config = FireworksAIConfig()
    message = Message(
        content='{"type": "function", "name": "get_current_weather", "parameters": {"location": "Boston, MA", "unit": "fahrenheit"}}',
        role="assistant",
        tool_calls=None,
        function_call=None,
        provider_specific_fields=None,
    )
    expected_tool_call = ChatCompletionMessageToolCall(
        function=Function(**json.loads(message.content)), id=None, type=None
    )
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    updated_message = config._handle_message_content_with_tool_calls(
        message, tool_calls
    )
    assert updated_message.tool_calls is not None
    assert len(updated_message.tool_calls) == 1
    assert updated_message.tool_calls[0].function.name == "get_current_weather"
    assert (
        updated_message.tool_calls[0].function.arguments
        == expected_tool_call.function.arguments
    )


def test_supports_reasoning_effort():
    """Test that reasoning_effort is only supported for specific Fireworks AI models."""
    # Models that support reasoning_effort
    supported_models = [
        "fireworks_ai/accounts/fireworks/models/qwen3-8b",
        "fireworks_ai/accounts/fireworks/models/qwen3-32b",
        "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        "fireworks_ai/accounts/fireworks/models/glm-4p5",
        "fireworks_ai/accounts/fireworks/models/glm-4p5-air",
        "fireworks_ai/accounts/fireworks/models/glm-4p6",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
    ]

    # Models that don't support reasoning_effort
    unsupported_models = [
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct",
        "fireworks_ai/accounts/fireworks/models/mixtral-8x7b-instruct",
    ]

    for model in supported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") == True
        ), f"{model} should support reasoning_effort"

    for model in unsupported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") == False
        ), f"{model} should not support reasoning_effort"


def test_get_supported_openai_params_reasoning_effort():
    """Test that reasoning_effort is always included — Fireworks accepts it on all models."""
    config = FireworksAIConfig()

    # reasoning_effort should be present for reasoning models
    supported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/qwen3-8b"
    )
    assert "reasoning_effort" in supported_params

    # reasoning_effort should also be present for non-reasoning models
    # (Fireworks API accepts it; unsupported models simply ignore it)
    other_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"
    )
    assert "reasoning_effort" in other_params


def test_add_transform_inline_image_block_skips_data_urls():
    """
    data: URLs must not have #transform=inline appended — doing so corrupts the
    base64 payload and raises binascii.Error: Incorrect padding on the Fireworks side.
    Regression test for https://github.com/BerriAI/litellm/issues/23583
    """
    config = FireworksAIConfig()
    data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgAB"

    # str branch
    str_content = {"type": "image_url", "image_url": data_url}
    result = config._add_transform_inline_image_block(
        str_content, model="gpt-4", disable_add_transform_inline_image_block=False
    )
    assert result["image_url"] == data_url, "data URL must not be modified (str branch)"

    # dict branch
    dict_content = {"type": "image_url", "image_url": {"url": data_url}}
    result = config._add_transform_inline_image_block(
        dict_content, model="gpt-4", disable_add_transform_inline_image_block=False
    )
    assert (
        result["image_url"]["url"] == data_url
    ), "data URL must not be modified (dict branch)"

    # regular https URL should still get the suffix
    https_content = {"type": "image_url", "image_url": "https://example.com/image.jpg"}
    result = config._add_transform_inline_image_block(
        https_content, model="gpt-4", disable_add_transform_inline_image_block=False
    )
    assert result["image_url"].endswith(
        "#transform=inline"
    ), "https URL should get #transform=inline"


def test_get_models_serverless_only():
    """Without FIREWORKS_ACCOUNT_ID, get_models queries only the fireworks public account."""
    config = FireworksAIConfig()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "accounts/fireworks/models/deepseek-v3p2"},
            {"name": "accounts/fireworks/models/qwen3-8b"},
        ]
    }

    with (
        patch(
            "litellm.module_level_client.get", return_value=mock_response
        ) as mock_get,
        patch(
            "litellm.llms.fireworks_ai.chat.transformation.get_secret_str",
            return_value=None,
        ),
    ):
        result = config.get_models(api_key="test-key")

        # Should call the management API for the fireworks public account
        mock_get.assert_called_once()
        called_url = mock_get.call_args.kwargs.get("url") or mock_get.call_args[1].get(
            "url", ""
        )
        assert called_url == "https://api.fireworks.ai/v1/accounts/fireworks/models"
        called_params = mock_get.call_args.kwargs.get("params", {})
        assert called_params.get("filter") == "supports_serverless=true"
        assert result == [
            "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
            "fireworks_ai/accounts/fireworks/models/qwen3-8b",
        ]


def test_get_models_with_account_id():
    """With FIREWORKS_ACCOUNT_ID set, get_models queries both public and user accounts."""
    config = FireworksAIConfig()

    serverless_response = MagicMock()
    serverless_response.status_code = 200
    serverless_response.json.return_value = {
        "models": [{"name": "accounts/fireworks/models/deepseek-v3p2"}]
    }

    user_response = MagicMock()
    user_response.status_code = 200
    user_response.json.return_value = {
        "models": [{"name": "accounts/myteam/models/my-finetuned-llama"}]
    }

    with (
        patch(
            "litellm.module_level_client.get",
            side_effect=[serverless_response, user_response],
        ) as mock_get,
        patch(
            "litellm.llms.fireworks_ai.chat.transformation.get_secret_str",
            side_effect=lambda key: "myteam" if key == "FIREWORKS_ACCOUNT_ID" else None,
        ),
    ):
        result = config.get_models(api_key="test-key")

        assert mock_get.call_count == 2
        # First call: fireworks public serverless
        url1 = mock_get.call_args_list[0].kwargs.get("url", "")
        assert url1 == "https://api.fireworks.ai/v1/accounts/fireworks/models"
        # Second call: user account
        url2 = mock_get.call_args_list[1].kwargs.get("url", "")
        assert url2 == "https://api.fireworks.ai/v1/accounts/myteam/models"
        assert result == [
            "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
            "fireworks_ai/accounts/myteam/models/my-finetuned-llama",
        ]


def test_get_models_deduplicates():
    """Models appearing in both public and user accounts are not duplicated."""
    config = FireworksAIConfig()

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "models": [{"name": "accounts/fireworks/models/deepseek-v3p2"}]
    }

    with (
        patch("litellm.module_level_client.get", return_value=response),
        patch(
            "litellm.llms.fireworks_ai.chat.transformation.get_secret_str",
            return_value=None,
        ),
    ):
        result = config.get_models(api_key="test-key")
        assert result == ["fireworks_ai/accounts/fireworks/models/deepseek-v3p2"]
        assert len(result) == 1


def test_get_models_no_api_key_raises():
    """get_models raises ValueError when no API key is available."""
    config = FireworksAIConfig()
    with patch(
        "litellm.llms.fireworks_ai.chat.transformation.get_secret_str",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="API key is not set"):
            config.get_models()


def test_transform_messages_helper_removes_provider_specific_fields():
    """
    Test that _transform_messages_helper removes provider_specific_fields from messages.
    """
    config = FireworksAIConfig()
    # Simulated messages, as dicts, including provider_specific_fields
    messages = [
        {
            "role": "user",
            "content": "Hello!",
            "provider_specific_fields": {"extra": "should be removed"},
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "provider_specific_fields": {"more": "remove this"},
        },
        {
            "role": "user",
            "content": "How are you?",
            # no provider_specific_fields
        },
    ]
    # Call helper
    out = config._transform_messages_helper(
        messages, model="fireworks/test", litellm_params={}
    )
    for msg in out:
        assert "provider_specific_fields" not in msg


def test_custom_llm_provider_property():
    """Test that the custom_llm_provider property returns 'fireworks_ai'."""
    config = FireworksAIConfig()
    assert config.custom_llm_provider == "fireworks_ai"
