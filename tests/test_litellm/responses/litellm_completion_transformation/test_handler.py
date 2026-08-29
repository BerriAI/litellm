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

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm._internal_context import is_proxy_stream_header_prefetch
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)


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


class _EmptyAsyncStream:
    def __init__(self):
        self.aclosed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        self.aclosed = True


@pytest.mark.asyncio
async def test_async_responses_bridge_keeps_sdk_deferred_gemini_stream_lazy():
    handler = LiteLLMCompletionTransformationHandler()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    deferred_stream = litellm.CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3.5-flash",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=AsyncMock(return_value=_EmptyAsyncStream()),
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=deferred_stream)):  # test-quality-ok: handler directly owns this module-level provider-call seam
        result = await handler.async_response_api_handler(
            litellm_completion_request={"model": "gemini-3.5-flash", "stream": True},
            request_input="ping",
            responses_api_request={},
        )

    assert result.litellm_custom_stream_wrapper is deferred_stream
    deferred_stream.make_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_responses_bridge_prefetches_deferred_gemini_stream_for_proxy():
    import datetime

    handler = LiteLLMCompletionTransformationHandler()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}, "llm_api_duration_ms": 40.0}
    logging_obj.start_time = datetime.datetime.now()
    deferred_stream = litellm.CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3.5-flash",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=AsyncMock(return_value=_EmptyAsyncStream()),
    )
    token = is_proxy_stream_header_prefetch.set(True)
    try:
        with patch("litellm.acompletion", new=AsyncMock(return_value=deferred_stream)):  # test-quality-ok: handler directly owns this module-level provider-call seam
            result = await handler.async_response_api_handler(
                litellm_completion_request={"model": "gemini-3.5-flash", "stream": True},
                request_input="ping",
                responses_api_request={},
            )
    finally:
        is_proxy_stream_header_prefetch.reset(token)

    assert result.litellm_custom_stream_wrapper is deferred_stream
    deferred_stream.make_call.assert_awaited_once()
    logging_obj.set_response_timing_metrics.assert_called_once()
    assert result._buffered_chunk is None
    assert result._response_id_primed is False


@pytest.mark.asyncio
async def test_async_responses_bridge_does_not_prefetch_already_connected_gemini_stream_for_proxy():
    handler = LiteLLMCompletionTransformationHandler()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}, "llm_api_duration_ms": 40.0}
    deferred_stream = litellm.CustomStreamWrapper(
        completion_stream=_EmptyAsyncStream(),
        model="gemini-3.5-flash",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=AsyncMock(return_value=_EmptyAsyncStream()),
    )
    token = is_proxy_stream_header_prefetch.set(True)
    try:
        with patch("litellm.acompletion", new=AsyncMock(return_value=deferred_stream)):  # test-quality-ok: handler directly owns this module-level provider-call seam
            result = await handler.async_response_api_handler(
                litellm_completion_request={"model": "gemini-3.5-flash", "stream": True},
                request_input="ping",
                responses_api_request={},
            )
    finally:
        is_proxy_stream_header_prefetch.reset(token)

    assert result.litellm_custom_stream_wrapper is deferred_stream
    assert deferred_stream.make_call.await_count == 0


@pytest.mark.asyncio
async def test_async_responses_bridge_closes_prefetched_gemini_stream_on_early_close():
    handler = LiteLLMCompletionTransformationHandler()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}, "llm_api_duration_ms": 40.0}
    logging_obj.start_time = datetime.datetime.now()
    upstream = _EmptyAsyncStream()
    deferred_stream = litellm.CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3.5-flash",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=AsyncMock(return_value=upstream),
    )
    token = is_proxy_stream_header_prefetch.set(True)
    try:
        with patch("litellm.acompletion", new=AsyncMock(return_value=deferred_stream)):  # test-quality-ok: handler directly owns this module-level provider-call seam
            result = await handler.async_response_api_handler(
                litellm_completion_request={"model": "gemini-3.5-flash", "stream": True},
                request_input="ping",
                responses_api_request={},
            )
    finally:
        is_proxy_stream_header_prefetch.reset(token)

    await result.aclose()

    assert upstream.aclosed is True
    assert deferred_stream.completion_stream is None
    with pytest.raises(StopAsyncIteration):
        await anext(result)
    deferred_stream.make_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_responses_bridge_propagates_initial_fetch_failure():
    from litellm.llms.vertex_ai.common_utils import VertexAIError

    handler = LiteLLMCompletionTransformationHandler()
    expected_error = VertexAIError(status_code=500, message="upstream failed", headers=None)
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}, "llm_api_duration_ms": 40.0}
    deferred_stream = litellm.CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3.5-flash",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=AsyncMock(side_effect=expected_error),
    )
    token = is_proxy_stream_header_prefetch.set(True)
    try:
        with patch("litellm.acompletion", new=AsyncMock(return_value=deferred_stream)):  # test-quality-ok: handler directly owns this module-level provider-call seam
            with pytest.raises(VertexAIError) as excinfo:
                await handler.async_response_api_handler(
                    litellm_completion_request={"model": "gemini-3.5-flash", "stream": True},
                    request_input="ping",
                    responses_api_request={},
                )
    finally:
        is_proxy_stream_header_prefetch.reset(token)

    assert excinfo.value is expected_error


@pytest.mark.asyncio
async def test_async_responses_bridge_does_not_prefetch_non_gemini_stream_for_proxy():
    handler = LiteLLMCompletionTransformationHandler()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}, "llm_api_duration_ms": 40.0}
    deferred_stream = litellm.CustomStreamWrapper(
        completion_stream=None,
        model="other-model",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        make_call=AsyncMock(return_value=_EmptyAsyncStream()),
    )
    token = is_proxy_stream_header_prefetch.set(True)
    try:
        with patch("litellm.acompletion", new=AsyncMock(return_value=deferred_stream)):  # test-quality-ok: handler directly owns this module-level provider-call seam
            result = await handler.async_response_api_handler(
                litellm_completion_request={"model": "other-model", "stream": True},
                request_input="ping",
                responses_api_request={},
            )
    finally:
        is_proxy_stream_header_prefetch.reset(token)

    assert result.litellm_custom_stream_wrapper is deferred_stream
    deferred_stream.make_call.assert_not_awaited()
