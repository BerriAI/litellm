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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.streaming_iterator import MockResponsesAPIStreamingIterator
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


def _completed_model_response() -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(role="assistant", content="hello"), finish_reason="stop")],
        model="claude-sonnet-4-5",
    )


@pytest.mark.asyncio
async def test_async_bridge_rebuilds_fake_stream_after_websearch_interception():
    """Original stream=True request converted to stream=False by websearch
    interception must come back as a synthetic responses stream, not a plain
    ResponsesAPIResponse the proxy cannot async-iterate."""
    handler = LiteLLMCompletionTransformationHandler()
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.dispatch_success_handlers = AsyncMock()
    logging_obj.model_call_details = {}

    async def fake_acompletion(**kwargs):
        return _completed_model_response()

    with patch("litellm.acompletion", fake_acompletion):  # test-quality-ok: bridge forwards to module-level acompletion; no injection seam exists
        result = await handler.async_response_api_handler(
            litellm_completion_request={"model": "anthropic/claude-sonnet-4-5", "messages": []},
            request_input="hello",
            responses_api_request={},
            _websearch_interception_converted_stream=True,
            litellm_logging_obj=logging_obj,
        )

    assert isinstance(result, MockResponsesAPIStreamingIterator)
    events = [event async for event in result]
    assert any(getattr(event, "type", None) == "response.completed" for event in events)


@pytest.mark.asyncio
async def test_async_bridge_returns_plain_response_without_converted_stream_flag():
    handler = LiteLLMCompletionTransformationHandler()

    async def fake_acompletion(**kwargs):
        return _completed_model_response()

    with patch("litellm.acompletion", fake_acompletion):  # test-quality-ok: bridge forwards to module-level acompletion; no injection seam exists
        result = await handler.async_response_api_handler(
            litellm_completion_request={"model": "anthropic/claude-sonnet-4-5", "messages": []},
            request_input="hello",
            responses_api_request={},
            litellm_logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        )

    assert not isinstance(result, MockResponsesAPIStreamingIterator)


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
