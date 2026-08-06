"""Regression tests for #36123 — a bridged /v1/responses stream must release the
upstream provider connection when the client goes away mid-stream."""

from datetime import datetime

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)


class _RecordingUpstreamStream:
    """Stands in for the provider byte stream held by the CustomStreamWrapper."""

    def __init__(self):
        self.aclose_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _make_bridge_iterator(upstream: _RecordingUpstreamStream) -> LiteLLMCompletionStreamingIterator:
    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=upstream,
        model="gpt-4o-mini",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    return LiteLLMCompletionStreamingIterator(
        model="gpt-4o-mini",
        litellm_custom_stream_wrapper=wrapper,
        request_input="hi",
        responses_api_request={},
    )


@pytest.mark.asyncio
async def test_aclose_closes_the_wrapped_chat_completions_stream():
    """The bridge holds the closeable CustomStreamWrapper as an attribute. Without an
    ``aclose`` of its own, the proxy's ``hasattr(response, "aclose")`` cleanup check
    fails and the provider keeps generating after the client disconnects."""
    upstream = _RecordingUpstreamStream()
    iterator = _make_bridge_iterator(upstream)

    await iterator.aclose()

    assert upstream.aclose_calls == 1


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    upstream = _RecordingUpstreamStream()
    iterator = _make_bridge_iterator(upstream)

    await iterator.aclose()
    await iterator.aclose()

    assert upstream.aclose_calls == 1
