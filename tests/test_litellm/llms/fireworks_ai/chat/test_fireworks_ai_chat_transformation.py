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
from litellm.llms.fireworks_ai.chat.transformation import (
    FireworksAIConfig,
    FireworksAIChatCompletionStreamingHandler,
)
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
    """Test that reasoning_effort is only included in supported params for models that support it."""
    config = FireworksAIConfig()

    # Model that supports reasoning_effort
    supported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/qwen3-8b"
    )
    assert "reasoning_effort" in supported_params

    # Model that doesn't support reasoning_effort
    unsupported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"
    )
    assert "reasoning_effort" not in unsupported_params


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


@pytest.mark.parametrize(
    "api_base, expected_url_prefix",
    [
        (
            "https://api.fireworks.ai/inference/v1",
            "https://api.fireworks.ai/inference/v1/accounts/",
        ),
        (
            "https://api.fireworks.ai/inference/v1/",
            "https://api.fireworks.ai/inference/v1/accounts/",
        ),
        (
            "https://custom-host.example.com/v1",
            "https://custom-host.example.com/v1/accounts/",
        ),
        (
            "https://custom-host.example.com/api",
            "https://custom-host.example.com/api/v1/accounts/",
        ),
    ],
    ids=["default", "trailing-slash", "custom-with-v1", "custom-without-v1"],
)
def test_get_models_url_no_double_v1(api_base, expected_url_prefix):
    """Ensure get_models never produces a /v1/v1/ URL segment (fixes #23106)."""
    config = FireworksAIConfig()
    account_id = "fireworks"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "accounts/fireworks/models/llama-v3-70b"}]
    }

    with (
        patch(
            "litellm.module_level_client.get", return_value=mock_response
        ) as mock_get,
        patch(
            "litellm.llms.fireworks_ai.chat.transformation.get_secret_str",
            side_effect=lambda key: {
                "FIREWORKS_API_KEY": "test-key",
                "FIREWORKS_API_BASE": api_base,
                "FIREWORKS_ACCOUNT_ID": account_id,
            }.get(key),
        ),
    ):
        result = config.get_models(api_key="test-key", api_base=api_base)

        called_url = mock_get.call_args.kwargs.get("url") or mock_get.call_args[1].get(
            "url", ""
        )
        assert "/v1/v1/" not in called_url, f"Double /v1/ detected in URL: {called_url}"
        assert called_url.startswith(
            expected_url_prefix
        ), f"URL {called_url} does not start with {expected_url_prefix}"
        assert result == ["fireworks_ai/accounts/fireworks/models/llama-v3-70b"]


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


# ---------------------------------------------------------------------------
# <think> tag extraction — non-streaming transform_response
# ---------------------------------------------------------------------------


def _make_raw_response(body: dict) -> MagicMock:
    """Build a minimal httpx.Response-like mock from a dict body."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = body
    mock_resp.text = json.dumps(body)
    mock_resp.status_code = 200
    mock_resp.headers = {}
    return mock_resp


def _make_completion_body(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def test_transform_response_extracts_think_tags():
    """Non-streaming: <think>...</think> in content → reasoning_content + clean content."""
    config = FireworksAIConfig()
    body = _make_completion_body("<think>step one\nstep two</think>The answer is 42.")
    raw = _make_raw_response(body)

    from litellm.types.utils import ModelResponse
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    model_response = ModelResponse()
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()

    response = config.transform_response(
        model="fireworks_ai/accounts/fireworks/models/kimi-k2.5",
        raw_response=raw,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None,
    )

    msg = response.choices[0].message
    assert msg.reasoning_content == "step one\nstep two"
    assert msg.content == "The answer is 42."


def test_transform_response_no_think_tags_unchanged():
    """Non-streaming: content without <think> tags is not modified."""
    config = FireworksAIConfig()
    body = _make_completion_body("Just a plain response.")
    raw = _make_raw_response(body)

    from litellm.types.utils import ModelResponse
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    model_response = ModelResponse()
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()

    response = config.transform_response(
        model="fireworks_ai/accounts/fireworks/models/kimi-k2.5",
        raw_response=raw,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None,
    )

    msg = response.choices[0].message
    assert msg.content == "Just a plain response."
    assert getattr(msg, "reasoning_content", None) is None


def test_transform_response_existing_reasoning_content_not_overwritten():
    """Non-streaming: explicit reasoning_content field is preserved as-is."""
    config = FireworksAIConfig()
    body = _make_completion_body("The answer.")
    # Inject reasoning_content directly in the raw response message
    body["choices"][0]["message"]["reasoning_content"] = "pre-existing reasoning"
    raw = _make_raw_response(body)

    from litellm.types.utils import ModelResponse
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    model_response = ModelResponse()
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()

    response = config.transform_response(
        model="fireworks_ai/accounts/fireworks/models/kimi-k2.5",
        raw_response=raw,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None,
    )

    msg = response.choices[0].message
    assert msg.reasoning_content == "pre-existing reasoning"
    assert msg.content == "The answer."


# ---------------------------------------------------------------------------
# <think> tag extraction — streaming chunk_parser
# ---------------------------------------------------------------------------


def test_streaming_chunk_parser_no_think_tags():
    """Streaming: plain content chunks pass through unchanged."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [{"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}],
    }
    result = handler.chunk_parser(chunk)
    assert result.choices[0].delta.content == "Hello world"
    assert getattr(result.choices[0].delta, "reasoning_content", None) is None


def test_streaming_chunk_parser_open_think_tag():
    """Streaming: chunk containing <think> starts reasoning accumulation."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [{"index": 0, "delta": {"content": "<think>start of reasoning"}, "finish_reason": None}],
    }
    result = handler.chunk_parser(chunk)
    assert handler.started_reasoning_content is True
    assert handler.finished_reasoning_content is False
    assert result.choices[0].delta.reasoning_content == "start of reasoning"
    assert result.choices[0].delta.content is None


def test_streaming_chunk_parser_mid_think_chunk():
    """Streaming: mid-think chunk (no open/close tag) routed to reasoning_content."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    handler.started_reasoning_content = True
    handler.finished_reasoning_content = False

    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [{"index": 0, "delta": {"content": "middle of reasoning"}, "finish_reason": None}],
    }
    result = handler.chunk_parser(chunk)
    assert result.choices[0].delta.reasoning_content == "middle of reasoning"
    assert result.choices[0].delta.content is None


def test_streaming_chunk_parser_close_think_tag():
    """Streaming: chunk with </think> splits reasoning from content correctly."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    handler.started_reasoning_content = True
    handler.finished_reasoning_content = False

    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "final reasoning bit</think>Actual answer here"},
                "finish_reason": None,
            }
        ],
    }
    result = handler.chunk_parser(chunk)
    assert handler.finished_reasoning_content is True
    assert result.choices[0].delta.reasoning_content == "final reasoning bit"
    assert result.choices[0].delta.content == "Actual answer here"


def test_streaming_chunk_parser_content_after_think_closed():
    """Streaming: chunks after </think> are plain content."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    handler.started_reasoning_content = True
    handler.finished_reasoning_content = True

    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [{"index": 0, "delta": {"content": "More answer text"}, "finish_reason": None}],
    }
    result = handler.chunk_parser(chunk)
    assert result.choices[0].delta.content == "More answer text"
    assert getattr(result.choices[0].delta, "reasoning_content", None) is None


def test_streaming_chunk_parser_think_tags_in_single_chunk():
    """Streaming: single chunk with full <think>...</think> is split correctly."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/kimi-k2.5",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "<think>I think</think>Here is the answer"},
                "finish_reason": None,
            }
        ],
    }
    result = handler.chunk_parser(chunk)
    assert handler.started_reasoning_content is True
    assert handler.finished_reasoning_content is True
    assert result.choices[0].delta.reasoning_content == "I think"
    assert result.choices[0].delta.content == "Here is the answer"


def test_streaming_chunk_parser_no_think_tags_any_model():
    """Streaming: content without <think> tags passes through unchanged regardless of model."""
    handler = FireworksAIChatCompletionStreamingHandler(
        streaming_response=iter([]), sync_stream=True
    )
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "accounts/fireworks/models/llama-v3-70b-instruct",
        "choices": [{"index": 0, "delta": {"content": "Plain response"}, "finish_reason": None}],
    }
    result = handler.chunk_parser(chunk)
    assert result.choices[0].delta.content == "Plain response"
    assert getattr(result.choices[0].delta, "reasoning_content", None) is None
    assert handler.started_reasoning_content is False
