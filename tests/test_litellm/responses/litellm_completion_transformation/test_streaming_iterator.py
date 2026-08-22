from unittest.mock import MagicMock

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

CHAT_COMPLETION_ID = "chatcmpl-77d33d09-effa-4cd2-9c0d-c742d4358256"
RESPONSE_ID_EVENT_TYPES = frozenset(
    {"response.created", "response.in_progress", "response.completed"}
)


def _chunk(content: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id=CHAT_COMPLETION_ID,
        created=1748575031,
        model="claude-haiku-4-5",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )


class _FakeStreamWrapper:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.logging_obj = MagicMock()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _build_iterator(chunks) -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="claude-haiku-4-5",
        litellm_custom_stream_wrapper=_FakeStreamWrapper(chunks),
        request_input="What is the weather in San Francisco?",
        responses_api_request={},
        custom_llm_provider="anthropic",
        litellm_metadata={},
    )


def _response_ids(events) -> list[str]:
    return [
        event.response.id
        for event in events
        if getattr(event, "type", None) in RESPONSE_ID_EVENT_TYPES
    ]


@pytest.mark.asyncio
async def test_streaming_events_share_the_chat_completion_response_id():
    """
    Every event of a bridged stream has to carry the same id, and that id has to decode
    to the chat completion id spend tracking stores as `request_id`. Otherwise a
    follow-up `previous_response_id` matches no session and the conversation is dropped.
    """
    iterator = _build_iterator([_chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = [event async for event in iterator]

    response_ids = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])
    assert decoded["response_id"] == CHAT_COMPLETION_ID
    assert decoded["custom_llm_provider"] == "anthropic"


def test_sync_streaming_events_share_the_chat_completion_response_id():
    iterator = _build_iterator([_chunk("Hello"), _chunk("!", finish_reason="stop")])

    events = list(iterator)

    response_ids = _response_ids(events)
    assert len(response_ids) == 3
    assert len(set(response_ids)) == 1
    assert (
        ResponsesAPIRequestUtils._decode_responses_api_response_id(response_ids[0])["response_id"]
        == CHAT_COMPLETION_ID
    )


@pytest.mark.asyncio
async def test_streaming_emits_every_chunk_after_priming_the_response_id():
    iterator = _build_iterator(
        [_chunk("Hel"), _chunk("lo"), _chunk("!", finish_reason="stop")]
    )

    events = [event async for event in iterator]

    deltas = "".join(
        event.delta for event in events if getattr(event, "type", None) == "response.output_text.delta"
    )
    assert deltas == "Hello!"


@pytest.mark.asyncio
async def test_streaming_response_id_falls_back_when_upstream_yields_nothing():
    iterator = _build_iterator([])

    events = [event async for event in iterator]

    response_ids = _response_ids(events)
    assert response_ids
    assert len(set(response_ids)) == 1
    assert response_ids[0].startswith("resp_")
