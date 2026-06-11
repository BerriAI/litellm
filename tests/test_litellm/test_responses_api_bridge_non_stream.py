import os
import sys
from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
    _build_responses_kwargs,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import Choices, Message, ModelResponse, Usage

"""
Test that all providers can transform completion responses to Responses API format
without breaking due to required fields in InputTokensDetails and OutputTokensDetails.

This is a regression test for the change where reasoning_tokens and cached_tokens
were made non-optional (must be int, not Optional[int]).
"""


class _CompletedEvent:
    def __init__(self, response):
        self.response = response


class _FakeResponsesStream:
    def __init__(self, response):
        self._emitted = False
        self._response = response
        self.completed_response = None
        self._hidden_params = {"headers": {"x-test": "1"}}

    def __iter__(self):
        return self

    def __next__(self):
        if not self._emitted:
            self._emitted = True
            self.completed_response = _CompletedEvent(self._response)
            return {"type": "response.completed"}
        raise StopIteration


def test_should_collect_response_from_stream():
    handler = ResponsesToCompletionBridgeHandler()
    response = ResponsesAPIResponse.model_construct(
        id="resp-1",
        created_at=0,
        output=[],
        object="response",
        model="gpt-5.2",
    )
    stream = _FakeResponsesStream(response)

    collected = handler._collect_response_from_stream(stream)

    assert collected.id == "resp-1"
    assert collected._hidden_params.get("headers") == {"x-test": "1"}


def test_streaming_iterator_recovers_output_item_and_text_chunks():
    mock_response = Mock()
    mock_response.headers = {}
    mock_logging_obj = Mock()
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    iterator = BaseResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-mini",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=mock_logging_obj,
        litellm_metadata={},
        custom_llm_provider="chatgpt",
    )

    # No completed response and no recovered chunks are both no-ops.
    iterator._recover_completed_response_output()
    empty_response = ResponsesAPIResponse.model_construct(
        id="resp-empty",
        created_at=0,
        output=[],
        object="response",
        model="gpt-5.4-mini",
    )
    iterator.completed_response = ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=empty_response,
    )
    iterator._recover_completed_response_output()
    assert empty_response.output == []

    iterator._record_output_chunk(
        {
            "type": "response.output_text.done",
            "output_index": 1,
            "content_index": 0,
            "item_id": "msg_text_only",
            "text": "hello from text done",
        }
    )
    iterator._record_output_chunk(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_item",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello from output item",
                    }
                ],
            },
        }
    )
    iterator._recover_completed_response_output()

    assert empty_response.output[0]["content"][0]["text"] == ("hello from output item")
    assert empty_response.output[1]["content"][0]["text"] == "hello from text done"


def test_streaming_iterator_recovery_preserves_existing_output():
    mock_response = Mock()
    mock_response.headers = {}
    mock_logging_obj = Mock()
    mock_logging_obj.model_call_details = {"litellm_params": {}}
    iterator = BaseResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-mini",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=mock_logging_obj,
        litellm_metadata={},
        custom_llm_provider="chatgpt",
    )
    response = ResponsesAPIResponse.model_construct(
        id="resp-existing",
        created_at=0,
        output=[{"type": "message", "content": []}],
        object="response",
        model="gpt-5.4-mini",
    )
    iterator.completed_response = ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )
    iterator._record_output_chunk(
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_ignored",
            "text": "ignored",
        }
    )
    iterator._recover_completed_response_output()

    assert response.output == [{"type": "message", "content": []}]


def test_adapter_direct_system_and_developer_messages_to_developer_input():
    adapter = LiteLLMAnthropicToResponsesAPIAdapter()

    system_result = adapter.translate_messages_to_responses_input(
        [{"role": "system", "content": "Follow policy."}]
    )
    assert system_result == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Follow policy."}],
        }
    ]

    developer_result = adapter.translate_messages_to_responses_input(
        [
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Be brief."},
                    {"type": "image", "source": {}},
                ],
            }
        ]
    )
    assert developer_result == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Be brief."}],
        }
    ]


def test_adapter_build_kwargs_uses_developer_input_for_chatgpt_system():
    kwargs = _build_responses_kwargs(
        max_tokens=128,
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5.5",
        system=[{"type": "text", "text": "Follow policy."}],
        extra_kwargs={"custom_llm_provider": "chatgpt"},
    )

    assert "instructions" not in kwargs
    assert kwargs["input"][0] == {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "Follow policy."}],
    }
    assert kwargs["input"][1]["role"] == "user"


def test_adapter_build_kwargs_keeps_openai_system_as_instructions():
    kwargs = _build_responses_kwargs(
        max_tokens=128,
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5.5",
        system="Follow policy.",
        extra_kwargs={"custom_llm_provider": "openai"},
    )

    assert kwargs["instructions"] == "Follow policy."
    assert kwargs["input"][0]["role"] == "user"


def test_chatgpt_responses_transforms_system_roles_only_for_list_input():
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig

    config = ChatGPTResponsesAPIConfig()
    assert config._transform_system_roles_to_developer("plain input") == "plain input"

    request = config.transform_responses_api_request(
        model="chatgpt/gpt-5.5",
        input=[
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "Follow policy."}],
            },
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Already developer."}],
            },
            "raw item",
        ],
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["input"][0]["role"] == "developer"
    assert request["input"][1]["role"] == "developer"
    assert request["input"][2] == "raw item"


def _forced_stream_provider_config():
    provider_config = Mock()
    provider_config.requires_streaming_response_api_transport = True
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = "https://chatgpt.test/responses"
    provider_config.transform_responses_api_request.return_value = {"input": "hi"}
    provider_config.sign_request.return_value = ({}, None)
    provider_config.get_error_class.side_effect = (
        lambda error_message, status_code, headers: RuntimeError(error_message)
    )
    return provider_config


def test_response_handler_forced_stream_requires_completed_response():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    provider_config = _forced_stream_provider_config()
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter([])
    mock_stream._get_completed_response_object.return_value = None
    mock_stream_factory = Mock(return_value=mock_stream)
    mock_client = Mock(spec=HTTPHandler)
    mock_client.post.return_value = Mock()

    with pytest.raises(RuntimeError, match="Stream ended without a completed response"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "litellm.llms.custom_httpx.llm_http_handler"
                ".SyncResponsesAPIStreamingIterator",
                mock_stream_factory,
            )
            BaseLLMHTTPHandler().response_api_handler(
                model="gpt-5.5",
                input="hi",
                responses_api_provider_config=provider_config,
                response_api_optional_request_params={},
                custom_llm_provider="chatgpt",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=Mock(),
                client=mock_client,
                fake_stream=True,
            )

    assert provider_config.sign_request.call_args.kwargs["stream"] is True
    assert mock_client.post.call_args.kwargs["stream"] is True
    assert mock_stream_factory.call_args.kwargs["log_completed_response"] is False


@pytest.mark.asyncio
async def test_async_response_handler_forced_stream_requires_completed_response():
    from unittest.mock import AsyncMock

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    provider_config = _forced_stream_provider_config()
    mock_stream = MagicMock()
    mock_stream.__aiter__.return_value = []
    mock_stream._get_completed_response_object.return_value = None
    mock_stream_factory = Mock(return_value=mock_stream)
    mock_client = Mock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=Mock())

    with pytest.raises(RuntimeError, match="Stream ended without a completed response"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "litellm.llms.custom_httpx.llm_http_handler"
                ".ResponsesAPIStreamingIterator",
                mock_stream_factory,
            )
            await BaseLLMHTTPHandler().async_response_api_handler(
                model="gpt-5.5",
                input="hi",
                responses_api_provider_config=provider_config,
                response_api_optional_request_params={},
                custom_llm_provider="chatgpt",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=Mock(),
                client=mock_client,
                fake_stream=True,
            )

    assert provider_config.sign_request.call_args.kwargs["stream"] is True
    assert mock_client.post.await_args.kwargs["stream"] is True
    assert mock_stream_factory.call_args.kwargs["log_completed_response"] is False


def test_streaming_iterator_logs_inner_completed_response_object():
    logging_obj = Mock()
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.success_handler = Mock()
    logging_obj.async_success_handler = Mock()
    iterator = BaseResponsesAPIStreamingIterator(
        response=Mock(headers={}),
        model="gpt-5.5",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=logging_obj,
        litellm_metadata={},
        custom_llm_provider="chatgpt",
    )
    iterator._persist_completed_response_before_logging = False
    response = ResponsesAPIResponse.model_construct(
        id="resp-log",
        created_at=0,
        output=[{"type": "message", "content": []}],
        object="response",
        model="gpt-5.5",
    )
    iterator.completed_response = ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )

    import importlib

    streaming_iterator_module = importlib.import_module(
        "litellm.responses.streaming_iterator"
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        run_async = Mock()
        executor = Mock()
        monkeypatch.setattr(streaming_iterator_module, "run_async_function", run_async)
        monkeypatch.setattr(streaming_iterator_module, "executor", executor)
        iterator._log_completed_response(is_async=False)

    success_call = next(
        call
        for call in run_async.call_args_list
        if call.kwargs.get("async_function") is logging_obj.async_success_handler
    )
    assert success_call.kwargs["result"].id == "resp-log"
    assert executor.submit.call_args.kwargs["result"].id == "resp-log"


def create_mock_completion_response(
    model: str = "gpt-4",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
    reasoning_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    text_tokens: Optional[int] = None,
) -> ModelResponse:
    """
    Create a mock ModelResponse (chat completion) with various token details.

    This simulates responses from different providers that may or may not include
    reasoning_tokens, cached_tokens, etc.
    """
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    # Add prompt_tokens_details if we have cached_tokens or text_tokens
    if cached_tokens is not None or text_tokens is not None:
        from litellm.types.utils import PromptTokensDetails

        usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cached_tokens,
            text_tokens=text_tokens,
        )

    # Add completion_tokens_details if we have reasoning_tokens or text_tokens
    if reasoning_tokens is not None or text_tokens is not None:
        from litellm.types.utils import CompletionTokensDetails

        usage.completion_tokens_details = CompletionTokensDetails(
            reasoning_tokens=reasoning_tokens,
            text_tokens=text_tokens,
        )

    return ModelResponse(
        id="chatcmpl-test",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Test response",
                    role="assistant",
                ),
            )
        ],
        usage=usage,
    )


def test_transform_usage_no_token_details():
    """
    Test that transformation works when completion response has NO token details.

    This simulates providers that don't return detailed token breakdowns.
    """
    completion_response = create_mock_completion_response(
        model="gpt-4",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )

    # Transform to Responses API usage format
    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )

    # Should succeed without errors
    assert responses_usage.input_tokens == 10
    assert responses_usage.output_tokens == 20
    assert responses_usage.total_tokens == 30

    # Token details should not be present when not provided
    assert responses_usage.input_tokens_details is None
    assert responses_usage.output_tokens_details is None

    print("✓ Transformation works with no token details")


def test_transform_usage_with_cached_tokens_only():
    """
    Test transformation when only cached_tokens is provided (no reasoning_tokens).

    This simulates providers like Anthropic that support prompt caching but not reasoning.
    """
    completion_response = create_mock_completion_response(
        model="claude-3-opus",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=80,  # Has cached tokens
        reasoning_tokens=None,  # No reasoning tokens
    )

    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )

    # Should succeed and default reasoning_tokens to 0
    assert responses_usage.input_tokens == 100
    assert responses_usage.output_tokens == 50
    assert responses_usage.total_tokens == 150

    # Input details should be present with cached_tokens
    assert responses_usage.input_tokens_details is not None
    assert isinstance(responses_usage.input_tokens_details, InputTokensDetails)
    assert responses_usage.input_tokens_details.cached_tokens == 80

    # Output details should not be present (no reasoning_tokens provided)
    assert responses_usage.output_tokens_details is None

    print("✓ Transformation works with cached_tokens only")


def test_transform_usage_with_reasoning_tokens_only():
    """
    Test transformation when only reasoning_tokens is provided (no cached_tokens).

    This simulates providers like OpenAI o1 that support reasoning but not caching.
    """
    completion_response = create_mock_completion_response(
        model="o1-preview",
        prompt_tokens=50,
        completion_tokens=100,
        total_tokens=150,
        cached_tokens=None,  # No cached tokens
        reasoning_tokens=60,  # Has reasoning tokens
    )

    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )

    # Should succeed and default cached_tokens to 0
    assert responses_usage.input_tokens == 50
    assert responses_usage.output_tokens == 100
    assert responses_usage.total_tokens == 150

    # Input details should not be present (no cached_tokens provided)
    assert responses_usage.input_tokens_details is None

    # Output details should be present with reasoning_tokens
    assert responses_usage.output_tokens_details is not None
    assert isinstance(responses_usage.output_tokens_details, OutputTokensDetails)
    assert responses_usage.output_tokens_details.reasoning_tokens == 60

    print("✓ Transformation works with reasoning_tokens only")


def test_transform_usage_with_both_token_details():
    """
    Test transformation when both cached_tokens and reasoning_tokens are provided.

    This simulates advanced providers that support both features.
    """
    completion_response = create_mock_completion_response(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        cached_tokens=50,
        reasoning_tokens=30,
        text_tokens=50,  # Also include text_tokens
    )

    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )

    # Should succeed with all details
    assert responses_usage.input_tokens == 100
    assert responses_usage.output_tokens == 80
    assert responses_usage.total_tokens == 180

    # Input details should have cached_tokens
    assert responses_usage.input_tokens_details is not None
    assert responses_usage.input_tokens_details.cached_tokens == 50
    assert responses_usage.input_tokens_details.text_tokens == 50

    # Output details should have reasoning_tokens
    assert responses_usage.output_tokens_details is not None
    assert responses_usage.output_tokens_details.reasoning_tokens == 30
    assert responses_usage.output_tokens_details.text_tokens == 50

    print("✓ Transformation works with both cached_tokens and reasoning_tokens")


def test_transform_usage_with_zero_values():
    """
    Test transformation when token details are explicitly set to 0.

    This ensures 0 values are preserved and not treated as None.
    """
    completion_response = create_mock_completion_response(
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=0,  # Explicitly 0
        reasoning_tokens=0,  # Explicitly 0
    )

    responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
        completion_response
    )

    # Should preserve 0 values
    assert responses_usage.input_tokens_details is not None
    assert responses_usage.input_tokens_details.cached_tokens == 0

    assert responses_usage.output_tokens_details is not None
    assert responses_usage.output_tokens_details.reasoning_tokens == 0

    print("✓ Transformation preserves explicit 0 values")


def test_input_tokens_details_requires_cached_tokens():
    """
    Test that InputTokensDetails has cached_tokens as an int with default value 0.

    This ensures backward compatibility while making the field non-optional.
    """
    # Should work with cached_tokens=0
    details1 = InputTokensDetails(cached_tokens=0)
    assert details1.cached_tokens == 0

    # Should work with cached_tokens=100
    details2 = InputTokensDetails(cached_tokens=100)
    assert details2.cached_tokens == 100

    # Should work without cached_tokens (defaults to 0)
    details3 = InputTokensDetails()
    assert details3.cached_tokens == 0

    print("✓ InputTokensDetails correctly defaults cached_tokens to 0")


def test_output_tokens_details_requires_reasoning_tokens():
    """
    Test that OutputTokensDetails has reasoning_tokens as an int with default value 0.

    This ensures backward compatibility while making the field non-optional.
    """
    # Should work with reasoning_tokens=0
    details1 = OutputTokensDetails(reasoning_tokens=0)
    assert details1.reasoning_tokens == 0

    # Should work with reasoning_tokens=100
    details2 = OutputTokensDetails(reasoning_tokens=100)
    assert details2.reasoning_tokens == 100

    # Should work without reasoning_tokens (defaults to 0)
    details3 = OutputTokensDetails()
    assert details3.reasoning_tokens == 0

    print("✓ OutputTokensDetails correctly defaults reasoning_tokens to 0")


def test_all_providers_transformation_scenarios():
    """
    Test various provider scenarios to ensure none break after the field requirement change.

    This tests the most common scenarios across different providers:
    - OpenAI: may have reasoning_tokens
    - Anthropic: may have cached_tokens
    - Azure: similar to OpenAI
    - Other providers: basic usage only
    """
    test_scenarios = [
        {
            "name": "Basic provider (no details)",
            "model": "gpt-3.5-turbo",
            "kwargs": {},
        },
        {
            "name": "OpenAI with reasoning",
            "model": "o1-preview",
            "kwargs": {"reasoning_tokens": 100},
        },
        {
            "name": "Anthropic with caching",
            "model": "claude-3-opus",
            "kwargs": {"cached_tokens": 50},
        },
        {
            "name": "OpenAI with caching",
            "model": "gpt-4o",
            "kwargs": {"cached_tokens": 30},
        },
        {
            "name": "Full details (both)",
            "model": "gpt-4o",
            "kwargs": {"cached_tokens": 40, "reasoning_tokens": 60, "text_tokens": 100},
        },
        {
            "name": "Zero values",
            "model": "gpt-4",
            "kwargs": {"cached_tokens": 0, "reasoning_tokens": 0},
        },
    ]

    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")

        completion_response = create_mock_completion_response(
            model=scenario["model"], **scenario["kwargs"]
        )

        # This should not raise any errors
        responses_usage = LiteLLMCompletionResponsesConfig._transform_chat_completion_usage_to_responses_usage(
            completion_response
        )

        # Basic assertions
        assert responses_usage.input_tokens >= 0
        assert responses_usage.output_tokens >= 0
        assert responses_usage.total_tokens >= 0

        # If input_tokens_details exists, cached_tokens must be an int
        if responses_usage.input_tokens_details is not None:
            assert isinstance(responses_usage.input_tokens_details.cached_tokens, int)

        # If output_tokens_details exists, reasoning_tokens must be an int
        if responses_usage.output_tokens_details is not None:
            assert isinstance(
                responses_usage.output_tokens_details.reasoning_tokens, int
            )

        print(f"  ✓ {scenario['name']} transformation successful")

    print("\n✓ All provider scenarios work correctly")


if __name__ == "__main__":
    # Run all tests
    test_transform_usage_no_token_details()
    test_transform_usage_with_cached_tokens_only()
    test_transform_usage_with_reasoning_tokens_only()
    test_transform_usage_with_both_token_details()
    test_transform_usage_with_zero_values()
    test_input_tokens_details_requires_cached_tokens()
    test_output_tokens_details_requires_reasoning_tokens()
    test_all_providers_transformation_scenarios()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
