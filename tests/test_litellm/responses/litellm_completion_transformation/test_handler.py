"""Regression tests for the responses -> completion fallback bridge guard.

When the Responses API falls back to chat completions (no native responses
config), it must tag the forwarded ``litellm.completion`` / ``litellm.acompletion``
call with ``_skip_responses_api_bridge=True`` so ``completion()`` does not bridge
the request straight back to the Responses API and mutually recurse forever.

Both fallback paths are covered: the sync ``response_api_handler`` (``_is_async``
False) and the async ``async_response_api_handler`` (``_is_async`` True). The
module-level ``litellm.completion`` / ``litellm.acompletion`` are patched to
capture the forwarded kwargs; if the flag-setting line is removed the captured
kwargs lack the flag and these tests fail.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse


class _StopForwarding(Exception):
    """Raised by the mocked (a)completion once the forwarded kwargs are captured."""


def test_sync_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
            )

    assert captured.get("_skip_responses_api_bridge") is True


@pytest.mark.asyncio
async def test_async_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
        )
        with pytest.raises(_StopForwarding):
            await coro

    assert captured.get("_skip_responses_api_bridge") is True


def _capture_sync_bridge_completion_kwargs(**bridge_kwargs) -> dict:
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
                **bridge_kwargs,
            )

    return captured


def test_sync_bridge_drops_unknown_client_fields():
    captured = _capture_sync_bridge_completion_kwargs(
        client_metadata={"x-codex-turn-metadata": '{"turn":1}'},
        custom_metadata={"team": "qa"},
    )

    assert "client_metadata" not in captured
    assert "custom_metadata" not in captured


@pytest.mark.asyncio
async def test_async_bridge_drops_unknown_client_fields():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
            client_metadata={"x-codex-turn-metadata": '{"turn":1}'},
            custom_metadata={"team": "qa"},
        )
        with pytest.raises(_StopForwarding):
            await coro

    assert "client_metadata" not in captured
    assert "custom_metadata" not in captured


def test_sync_bridge_keeps_internal_and_chat_completion_params():
    kept_kwargs = {
        "litellm_call_id": "call-id-1",
        "litellm_metadata": {"user_api_key": "hashed"},
        "proxy_server_request": {"url": "http://localhost:4000/v1/responses"},
        "model_info": {"id": "deployment-1"},
        "drop_params": True,
        "extra_body": {"backend_specific": 1},
        "vertex_project": "my-project",
        "aws_region_name": "us-east-1",
        "prompt_cache_key": "codex-session-1",
        "stream_options": {"include_usage": True},
        "safety_identifier": "user-7",
    }

    captured = _capture_sync_bridge_completion_kwargs(**kept_kwargs)

    for key, value in kept_kwargs.items():
        assert captured.get(key) == value, key


def test_sync_bridge_keeps_params_named_in_allowed_openai_params():
    captured = _capture_sync_bridge_completion_kwargs(
        allowed_openai_params=["top_k"],
        top_k=5,
    )

    assert captured.get("top_k") == 5
    assert captured.get("allowed_openai_params") == ["top_k"]


def _model_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-bridge-test",
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="hi", role="assistant"))],
    )


def test_sync_bridge_returns_responses_api_response_for_model_response():
    """The bridge must also drop unknown fields on the non-streaming success path,
    not just when it errors out before reaching litellm.completion."""
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion", return_value=_model_response()):
        result = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=False,
            client_metadata={"x-codex-turn-metadata": '{"turn":1}'},
        )

    assert isinstance(result, ResponsesAPIResponse)
    assert result.model == "gpt-4o"


def test_sync_bridge_returns_streaming_iterator_for_custom_stream_wrapper():
    handler = LiteLLMCompletionTransformationHandler()
    mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
    mock_stream_wrapper.logging_obj = Mock()

    with patch("litellm.completion", return_value=mock_stream_wrapper):
        result = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=False,
            stream=True,
            litellm_metadata={"user_api_key": "hashed"},
        )

    assert isinstance(result, BaseResponsesAPIStreamingIterator)
    assert result.litellm_metadata == {"user_api_key": "hashed"}


@pytest.mark.asyncio
async def test_async_bridge_returns_responses_api_response_for_model_response():
    handler = LiteLLMCompletionTransformationHandler()

    async def fake_acompletion(**kwargs):
        return _model_response()

    with patch("litellm.acompletion", fake_acompletion):
        result = await handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
            client_metadata={"x-codex-turn-metadata": '{"turn":1}'},
        )

    assert isinstance(result, ResponsesAPIResponse)
    assert result.model == "gpt-4o"


@pytest.mark.asyncio
async def test_async_bridge_returns_streaming_iterator_for_custom_stream_wrapper():
    handler = LiteLLMCompletionTransformationHandler()
    mock_stream_wrapper = Mock(spec=litellm.CustomStreamWrapper)
    mock_stream_wrapper.logging_obj = Mock()

    async def fake_acompletion(**kwargs):
        return mock_stream_wrapper

    with patch("litellm.acompletion", fake_acompletion):
        result = await handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
            stream=True,
            litellm_metadata={"user_api_key": "hashed"},
        )

    assert isinstance(result, BaseResponsesAPIStreamingIterator)
    assert result.litellm_metadata == {"user_api_key": "hashed"}


def test_sync_bridge_raises_on_unexpected_response_type():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion", return_value="not-a-model-response"):
        with pytest.raises(ValueError, match="Unexpected response type"):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
            )


@pytest.mark.asyncio
async def test_async_bridge_raises_on_unexpected_response_type():
    handler = LiteLLMCompletionTransformationHandler()

    async def fake_acompletion(**kwargs):
        return "not-a-model-response"

    with patch("litellm.acompletion", fake_acompletion):
        with pytest.raises(ValueError, match="Unexpected response type"):
            await handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=True,
            )
