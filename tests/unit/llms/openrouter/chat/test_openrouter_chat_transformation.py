import json
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import ModelResponse


from litellm.llms.openrouter.chat.transformation import (
    OpenRouterChatCompletionStreamingHandler,
    OpenrouterConfig,
    OpenRouterException,
)


class TestOpenRouterChatCompletionStreamingHandler:
    def test_chunk_parser_successful(self):
        handler = OpenRouterChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)

        # Test input chunk
        chunk = {
            "id": "test_id",
            "created": 1234567890,
            "model": "test_model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "choices": [{"delta": {"content": "test content", "reasoning": "test reasoning"}}],
        }

        # Parse chunk
        result = handler.chunk_parser(chunk)

        # Verify response
        assert result.id == "test_id"
        assert result.object == "chat.completion.chunk"
        assert result.created == 1234567890
        assert result.model == "test_model"
        assert result.usage.prompt_tokens == chunk["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == chunk["usage"]["completion_tokens"]
        assert result.usage.total_tokens == chunk["usage"]["total_tokens"]
        assert len(result.choices) == 1
        assert result.choices[0]["delta"]["reasoning_content"] == "test reasoning"

    def test_chunk_parser_error_response(self):
        handler = OpenRouterChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)

        # Test error chunk
        error_chunk = {
            "error": {
                "message": "test error",
                "code": 400,
                "metadata": {"key": "value"},
                "user_id": "test_user",
            }
        }

        # Verify error handling
        with pytest.raises(OpenRouterException) as exc_info:
            handler.chunk_parser(error_chunk)

        assert "Message: test error" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_chunk_parser_key_error(self):
        handler = OpenRouterChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)

        # Test invalid chunk missing required fields
        invalid_chunk = {"incomplete": "data"}

        # Verify KeyError handling
        with pytest.raises(OpenRouterException) as exc_info:
            handler.chunk_parser(invalid_chunk)

        assert "KeyError" in str(exc_info.value)
        assert exc_info.value.status_code == 400


def test_openrouter_extra_body_transformation():
    transformed_request = OpenrouterConfig().transform_request(
        model="openrouter/deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Hello, world!"}],
        optional_params={"extra_body": {"provider": {"order": ["DeepSeek"]}}},
        litellm_params={},
        headers={},
    )

    # https://github.com/BerriAI/litellm/issues/8425, validate its not contained in extra_body still
    assert transformed_request["provider"]["order"] == ["DeepSeek"]
    assert transformed_request["messages"] == [{"role": "user", "content": "Hello, world!"}]


def test_openrouter_supported_params_include_reasoning_for_unmapped_model():
    supported_params = OpenrouterConfig().get_supported_openai_params("openrouter/unmapped-model")

    assert "reasoning_effort" in supported_params
    assert "thinking" in supported_params


def test_openrouter_cache_control_flag_removal():
    transformed_request = OpenrouterConfig().transform_request(
        model="openrouter/deepseek/deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "Hello, world!",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert transformed_request["messages"][0].get("cache_control") is None


def test_openrouter_transform_request_with_cache_control():
    """
    Test transform_request moves cache_control from message level to content blocks (string content).

    Input:
    {
        "role": "user",
        "content": "what are the key terms...",
        "cache_control": {"type": "ephemeral"}
    }

    Expected Output:
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what are the key terms...",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
    """
    config = OpenrouterConfig()

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing legal documents.",
                },
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement",
                },
            ],
        },
        {
            "role": "user",
            "content": "what are the key terms and conditions in this agreement?",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert "messages" in transformed_request
    assert len(transformed_request["messages"]) == 2

    user_message = transformed_request["messages"][1]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_transform_request_with_cache_control_list_content():
    """
    Test transform_request moves cache_control only to the last content block when content is already a list.
    This prevents exceeding Anthropic's limit of 4 cache breakpoints.

    Input:
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a historian..."},
            {"type": "text", "text": "HUGE TEXT BODY"}
        ],
        "cache_control": {"type": "ephemeral"}
    }

    Expected Output:
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a historian..."
            },
            {
                "type": "text",
                "text": "HUGE TEXT BODY",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
    """
    config = OpenrouterConfig()

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a historian studying the fall of the Roman Empire.",
                },
                {"type": "text", "text": "HUGE TEXT BODY"},
            ],
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "user", "content": "What triggered the collapse?"},
    ]

    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert "messages" in transformed_request
    assert len(transformed_request["messages"]) == 2

    system_message = transformed_request["messages"][0]
    assert system_message["role"] == "system"
    assert isinstance(system_message["content"], list)
    assert len(system_message["content"]) == 2
    # Only the last content block should have cache_control
    assert "cache_control" not in system_message["content"][0]
    assert system_message["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_message


def test_openrouter_transform_request_with_cache_control_gemini():
    """
    Test transform_request moves cache_control to content blocks for Gemini models.

    Input:
    {
        "role": "user",
        "content": "Analyze this data",
        "cache_control": {"type": "ephemeral"}
    }

    Expected Output:
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Analyze this data",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
    """
    config = OpenrouterConfig()

    messages = [
        {
            "role": "user",
            "content": "Analyze this data",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    transformed_request = config.transform_request(
        model="openrouter/google/gemini-2.0-flash-exp:free",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert "messages" in transformed_request
    assert len(transformed_request["messages"]) == 1

    user_message = transformed_request["messages"][0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openrouter_transform_request_multiple_cache_controls():
    """
    Test that cache_control is only added to the last content block per message.
    This prevents exceeding Anthropic's limit of 4 cache breakpoints.

    When a message has 5 content blocks with cache_control at message level,
    only the 5th block should have cache_control, not all 5 blocks.
    """
    config = OpenrouterConfig()

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
                {"type": "text", "text": "Block 3"},
                {"type": "text", "text": "Block 4"},
                {"type": "text", "text": "Block 5"},
            ],
            "cache_control": {"type": "ephemeral"},
        }
    ]

    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    system_message = transformed_request["messages"][0]
    assert len(system_message["content"]) == 5

    # Only the last block should have cache_control
    for i in range(4):
        assert "cache_control" not in system_message["content"][i], f"Block {i} should not have cache_control"

    assert system_message["content"][4]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_message


def test_openrouter_cost_tracking_non_streaming():
    """
    Test OpenRouter cost tracking for non-streaming completions.

    Verifies:
    1. Request includes usage.include=true to get cost data
    2. Response extracts cost from usage.cost and stores in _hidden_params
    """
    config: Final = OpenrouterConfig()

    # Test request adds usage parameter
    transformed_request: Final = config.transform_request(
        model="openrouter/test-model",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "usage" in transformed_request
    assert transformed_request["usage"] == {"include": True}

    # Test response extracts cost
    raw_response: Final = httpx.Response(
        200,
        json={
            "id": "gen-123",
            "model": "openrouter/test-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.00015,
            },
        },
    )

    result: Final = _transform_openrouter_response(raw_response, _openrouter_logging())

    assert hasattr(result, "_hidden_params")
    assert "llm_provider-x-litellm-response-cost" in result._hidden_params["additional_headers"]
    assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.00015


def test_openrouter_cost_tracking_ignores_invalid_cost():
    from unittest.mock import Mock
    from litellm.types.utils import ModelResponse

    raw_response = httpx.Response(
        status_code=200,
        json={
            "id": "gen-123",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "openrouter/z-ai/glm-5.1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": "N/A",
            },
        },
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    result = OpenrouterConfig().transform_response(
        model="openrouter/z-ai/glm-5.1",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.choices[0].message.content == "Hello!"
    assert result.usage.total_tokens == 30
    additional_headers = result._hidden_params.get("additional_headers", {})
    assert "llm_provider-x-litellm-response-cost" not in additional_headers


def test_openrouter_transform_response_non_json_2xx_error():
    from unittest.mock import Mock
    from litellm.types.utils import ModelResponse

    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/plain", "x-request-id": "req_test"},
        content=(b"         \n\n" * 100) + b". ",
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with pytest.raises(OpenRouterException) as exc_info:
        OpenrouterConfig().transform_response(
            model="openrouter/z-ai/glm-5.1",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert "OpenRouter returned a non-JSON response" in str(exc_info.value)
    assert "Original Response" in str(exc_info.value)
    assert "." in str(exc_info.value)
    assert exc_info.value.status_code == 502


def test_openrouter_transform_response_embedded_error_before_parent_parse():
    from unittest.mock import Mock
    from litellm.types.utils import ModelResponse

    raw_response = httpx.Response(
        status_code=200,
        json={
            "error": {
                "message": "upstream provider rate limited",
                "code": 429,
                "metadata": {"headers": {"retry-after": "5"}},
            }
        },
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with pytest.raises(OpenRouterException) as exc_info:
        OpenrouterConfig().transform_response(
            model="openrouter/z-ai/glm-5.1",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=Mock(),
            request_data={},
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert "upstream provider rate limited" in str(exc_info.value)
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["retry-after"] == "5"
    assert exc_info.value.headers["content-type"] == raw_response.headers["content-type"]


@pytest.fixture
def openrouter_completion_payload() -> dict[str, object]:
    return {
        "id": "gen-regression",
        "created": 1234567890,
        "object": "chat.completion",
        "model": "openrouter/test-model",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _openrouter_logging(callback: CustomLogger | None = None) -> Logging:
    logging_obj: Final = Logging(
        model="openrouter/test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="openrouter-regression",
        function_id="openrouter-regression",
        dynamic_failure_callbacks=[callback] if callback is not None else None,
    )
    logging_obj.update_environment_variables(
        model="openrouter/test-model",
        user="test-user",
        optional_params={},
        litellm_params={"custom_llm_provider": "openrouter"},
    )
    return logging_obj


def _transform_openrouter_response(raw_response: httpx.Response, logging_obj: Logging) -> ModelResponse:
    return OpenrouterConfig().transform_response(
        model="openrouter/test-model",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={"model": "test-model"},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


@pytest.mark.parametrize("error", ({}, None, {"message": ""}, {"code": None}, {"metadata": {"provider_name": "test"}}))
def test_openrouter_valid_completion_with_empty_error_succeeds(
    openrouter_completion_payload: dict[str, object], error: object
) -> None:
    raw_response: Final = httpx.Response(200, json={**openrouter_completion_payload, "error": error})
    logging_obj: Final = _openrouter_logging()

    result: Final = _transform_openrouter_response(raw_response, logging_obj)

    assert result.choices[0].message.content == "Hello!"
    assert result.usage.total_tokens == 30
    assert logging_obj.model_call_details["original_response"] == raw_response.text


@pytest.mark.parametrize(
    ("body", "status_code"),
    (
        ("<html>" + "upstream failed " * 100 + "</html>", 502),
        ('{"error":{"message":"rate limited","code":429}}', 429),
        ("[]", 502),
        ("null", 502),
    ),
    ids=("long-html", "embedded-429", "json-array", "json-null"),
)
def test_openrouter_failure_callback_receives_complete_raw_body(body: str, status_code: int) -> None:
    class CaptureFailure(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.original_response: object = None

        def log_failure_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            self.original_response = kwargs.get("original_response")

    callback: Final = CaptureFailure()
    logging_obj: Final = _openrouter_logging(callback)
    raw_response: Final = httpx.Response(200, text=body)

    with pytest.raises(OpenRouterException) as exc_info:
        _transform_openrouter_response(raw_response, logging_obj)

    logging_obj.failure_handler(
        exception=exc_info.value,
        traceback_exception="test failure",
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1, 0, 0, 1),
    )
    assert exc_info.value.status_code == status_code
    assert callback.original_response == body


@pytest.mark.parametrize("encoding", ("utf-8", "utf-16"))
@pytest.mark.parametrize("usage", (None, {}))
def test_openrouter_completion_without_usage_preserves_json_decoding(
    openrouter_completion_payload: dict[str, object], encoding: str, usage: object
) -> None:
    raw_response: Final = httpx.Response(
        200, content=json.dumps({**openrouter_completion_payload, "usage": usage}).encode(encoding)
    )

    result: Final = _transform_openrouter_response(raw_response, _openrouter_logging())

    assert result.choices[0].message.content == "Hello!"
    assert "llm_provider-x-litellm-response-cost" not in result._hidden_params.get("additional_headers", {})


@pytest.mark.parametrize("code", (True, False, 200, 302, 700, "429", None))
def test_openrouter_embedded_error_never_uses_invalid_http_status(code: object) -> None:
    raw_response: Final = httpx.Response(200, json={"error": {"message": "upstream failed", "code": code}})

    with pytest.raises(OpenRouterException) as exc_info:
        _transform_openrouter_response(raw_response, _openrouter_logging())

    assert exc_info.value.status_code == 502


@pytest.mark.parametrize(
    "metadata", (None, "invalid", {"headers": None}, {"headers": ["invalid"]}, {"headers": {"retry-after": {}}})
)
def test_openrouter_embedded_error_preserves_http_headers_with_invalid_metadata(metadata: object) -> None:
    raw_response: Final = httpx.Response(
        200,
        json={"error": {"message": "rate limited", "code": 429, "metadata": metadata}},
        headers={"retry-after": "7", "x-request-id": "req-original"},
    )

    with pytest.raises(OpenRouterException) as exc_info:
        _transform_openrouter_response(raw_response, _openrouter_logging())

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["retry-after"] == "7"
    assert exc_info.value.headers["x-request-id"] == "req-original"


def test_openrouter_embedded_error_merges_metadata_and_http_headers() -> None:
    raw_response: Final = httpx.Response(
        200,
        json={"error": {"message": "rate limited", "code": 429, "metadata": {"headers": {"Retry-After": "5"}}}},
        headers={"retry-after": "7", "x-request-id": "req-original"},
    )

    with pytest.raises(OpenRouterException) as exc_info:
        _transform_openrouter_response(raw_response, _openrouter_logging())

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["retry-after"] == "5"
    assert exc_info.value.headers["x-request-id"] == "req-original"


def test_openrouter_cost_tracking_streaming():
    """
    Test OpenRouter cost tracking for streaming completions.

    Verifies:
    1. Request includes usage.include=true (same as non-streaming)
    2. Streaming chunks preserve usage/cost data in the final chunk
    3. Cost field is accessible in the usage object
    """
    config = OpenrouterConfig()

    # Test request adds usage parameter for streaming
    transformed_request = config.transform_request(
        model="openrouter/anthropic/claude-sonnet-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "usage" in transformed_request
    assert transformed_request["usage"] == {"include": True}

    # Test streaming chunks preserve cost data
    handler = OpenRouterChatCompletionStreamingHandler(streaming_response=None, sync_stream=True)

    # First chunk - content only
    chunk1 = {
        "id": "gen-stream-456",
        "created": 1234567890,
        "model": "openrouter/anthropic/claude-sonnet-4.5",
        "choices": [{"delta": {"content": "Hello", "reasoning": None}, "index": 0}],
    }

    # Final chunk - usage and cost
    chunk2 = {
        "id": "gen-stream-456",
        "created": 1234567890,
        "model": "openrouter/anthropic/claude-sonnet-4.5",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "cost": 0.0001,
        },
        "choices": [
            {
                "delta": {"content": "", "reasoning": None},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    result1 = handler.chunk_parser(chunk1)
    result2 = handler.chunk_parser(chunk2)

    # First chunk has content, no usage
    assert result1.choices[0]["delta"]["content"] == "Hello"
    assert result1.usage is None

    # Final chunk has usage with cost preserved
    assert result2.choices[0]["finish_reason"] == "stop"
    assert result2.usage is not None
    assert result2.usage.prompt_tokens == 5
    assert result2.usage.completion_tokens == 10
    assert result2.usage.total_tokens == 15
    # Verify cost field is preserved in the Usage object - this is the key data for cost tracking
    # The chunk_parser converts the dict to a Usage Pydantic model which includes the cost field
    assert result2.usage.cost == 0.0001


def test_openrouter_reasoning_models_allow_reasoning_effort_param():
    """
    OpenRouter reasoning-capable models should accept the reasoning_effort param.
    """
    config = OpenrouterConfig()

    supported_params = config.get_supported_openai_params(model="openrouter/deepseek/deepseek-v3.2")

    assert "reasoning_effort" in supported_params
    assert supported_params.count("reasoning_effort") == 1


def test_openrouter_non_reasoning_models_support_reasoning_params():
    config = OpenrouterConfig()

    supported_params = config.get_supported_openai_params(model="openrouter/anthropic/claude-3-5-haiku")

    assert "reasoning_effort" in supported_params
    assert "thinking" in supported_params


def test_openrouter_reasoning_effort_max_maps_to_xhigh():
    """
    OpenRouter expects 'xhigh' instead of 'max' for reasoning_effort.
    """
    config = OpenrouterConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": "max"},
        optional_params={},
        model="openrouter/deepseek/deepseek-r1",
        drop_params=False,
    )

    assert result["reasoning_effort"] == "xhigh"


def test_openrouter_reasoning_effort_max_does_not_mutate_caller_dict():
    """
    map_openai_params must not mutate the caller-supplied non_default_params dict.
    """
    config = OpenrouterConfig()
    original_params = {"reasoning_effort": "max"}

    config.map_openai_params(
        non_default_params=original_params,
        optional_params={},
        model="openrouter/deepseek/deepseek-r1",
        drop_params=False,
    )

    assert original_params["reasoning_effort"] == "max"


def test_openrouter_reasoning_effort_xhigh_passes_through():
    """
    reasoning_effort='xhigh' should be forwarded unchanged.
    """
    config = OpenrouterConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="openrouter/deepseek/deepseek-r1",
        drop_params=False,
    )

    assert result["reasoning_effort"] == "xhigh"


def test_openrouter_reasoning_effort_high_passes_through():
    """
    Non-max reasoning_effort values should be forwarded unchanged.
    """
    config = OpenrouterConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="openrouter/deepseek/deepseek-r1",
        drop_params=False,
    )

    assert result["reasoning_effort"] == "high"


@respx.mock
def test_openrouter_streaming_completion_assembles_text_and_finish_reason() -> None:
    chunks: Final = (
        {
            "id": "gen-stream-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "gen-stream-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": "The Supreme Court"}, "finish_reason": None}],
        },
        {
            "id": "gen-stream-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": " hears federal cases."}, "finish_reason": None}],
        },
        {
            "id": "gen-stream-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    )
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How does a case reach the Supreme Court?"},
    ]
    stream: Final = litellm.completion(
        model="openrouter/openai/gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=50,
        api_key="test-openrouter-key",
    )
    stream_chunks: Final = tuple(stream)
    assert stream_chunks[0].choices[0].delta.role == "assistant"
    assert tuple(
        chunk.choices[0].finish_reason for chunk in stream_chunks if chunk.choices[0].finish_reason is not None
    ) == ("stop",)
    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in stream_chunks if chunk.choices)
        == "The Supreme Court hears federal cases."
    )
    assembled: Final = litellm.stream_chunk_builder(chunks=list(stream_chunks), messages=messages)
    assert assembled is not None
    assert assembled.choices[0].message.content == "The Supreme Court hears federal cases."
    assert assembled.choices[0].finish_reason == "stop"


@respx.mock
def test_openrouter_streaming_reasoning_is_preserved() -> None:
    chunks: Final = (
        {
            "id": "gen-reasoning-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-sonnet-4.5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning": "Consider the evidence. "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "gen-reasoning-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-sonnet-4.5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "The decision is reasoned.", "reasoning": "Apply the rule."},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "gen-reasoning-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "anthropic/claude-sonnet-4.5",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    )
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    route: Final = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    parsed_reasoning_chunk: Final = OpenRouterChatCompletionStreamingHandler(
        streaming_response=None, sync_stream=True
    ).chunk_parser(chunks[0])
    assert parsed_reasoning_chunk.choices[0]["delta"]["reasoning_content"] == "Consider the evidence. "
    messages: Final = [{"role": "user", "content": "Explain the decision briefly."}]
    stream: Final = litellm.completion(
        model="openrouter/anthropic/claude-sonnet-4.5",
        messages=messages,
        stream=True,
        reasoning={"effort": "high"},
        drop_params=True,
        api_key="test-openrouter-key",
    )
    stream_chunks: Final = tuple(stream)
    assert json.loads(route.calls.last.request.content)["reasoning"] == {"effort": "high"}
    reasoning_deltas: Final = tuple(
        getattr(chunk.choices[0].delta, "reasoning_content", None) for chunk in stream_chunks if chunk.choices
    )
    assert reasoning_deltas == ("Consider the evidence. ", "Apply the rule.", None)
    assembled: Final = litellm.stream_chunk_builder(chunks=list(stream_chunks), messages=messages)
    assert assembled is not None
    assert assembled.choices[0].message.reasoning_content == "Consider the evidence. Apply the rule."
