import time
from collections.abc import Sequence
from typing import Final

import pytest

import litellm
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)

MOCK_TEXT: Final = "hello from the mock response"
MOCK_DELAY_SECONDS: Final = 0.2
EXPECTED_BOOKENDS: Final = (
    (ResponsesAPIStreamEvents.RESPONSE_CREATED, ResponseCreatedEvent),
    (ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS, ResponseInProgressEvent),
    (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, OutputItemAddedEvent),
    (ResponsesAPIStreamEvents.CONTENT_PART_ADDED, ContentPartAddedEvent),
)
EXPECTED_TAIL: Final = (
    (ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE, OutputTextDoneEvent),
    (ResponsesAPIStreamEvents.CONTENT_PART_DONE, ContentPartDoneEvent),
    (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, OutputItemDoneEvent),
    (ResponsesAPIStreamEvents.RESPONSE_COMPLETED, ResponseCompletedEvent),
)


def _assert_is_a_valid_mock_event_stream(events: Sequence[ResponsesAPIStreamingResponse]) -> None:
    delta_count: Final = len(events) - len(EXPECTED_BOOKENDS) - len(EXPECTED_TAIL)
    assert delta_count >= 1
    expected: Final = (
        *EXPECTED_BOOKENDS,
        *((ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, OutputTextDeltaEvent),) * delta_count,
        *EXPECTED_TAIL,
    )
    assert [event.type for event in events] == [event_type for event_type, _ in expected]
    assert all(isinstance(event, event_class) for event, (_, event_class) in zip(events, expected))
    assert [event.sequence_number for event in events] == list(range(len(events)))

    deltas: Final = tuple(event.delta for event in events if isinstance(event, OutputTextDeltaEvent))
    assert "".join(deltas) == MOCK_TEXT
    completed: Final = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.output[0].content[0].text == MOCK_TEXT


@pytest.mark.asyncio
async def test_aresponses_mock_response_with_stream_yields_ordered_typed_events():
    stream: Final = await litellm.aresponses(model="openai/gpt-4.1", input="hi", mock_response=MOCK_TEXT, stream=True)

    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    _assert_is_a_valid_mock_event_stream([event async for event in stream])


def test_responses_mock_response_with_stream_yields_ordered_typed_events():
    stream: Final = litellm.responses(model="openai/gpt-4.1", input="hi", mock_response=MOCK_TEXT, stream=True)

    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    _assert_is_a_valid_mock_event_stream(list(stream))


@pytest.mark.asyncio
async def test_aresponses_mock_response_without_stream_still_returns_the_full_response():
    response: Final = await litellm.aresponses(model="openai/gpt-4.1", input="hi", mock_response=MOCK_TEXT)

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output[0].content[0].text == MOCK_TEXT


def test_responses_mock_response_without_stream_still_returns_the_full_response():
    response: Final = litellm.responses(model="openai/gpt-4.1", input="hi", mock_response=MOCK_TEXT, stream=False)

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output[0].content[0].text == MOCK_TEXT


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_aresponses_mock_delay_delays_the_call(stream: bool):
    started: Final = time.monotonic()
    await litellm.aresponses(
        model="openai/gpt-4.1", input="hi", mock_response=MOCK_TEXT, mock_delay=MOCK_DELAY_SECONDS, stream=stream
    )

    assert time.monotonic() - started >= MOCK_DELAY_SECONDS


@pytest.mark.parametrize("stream", [False, True])
def test_responses_mock_delay_delays_the_call(stream: bool):
    started: Final = time.monotonic()
    litellm.responses(
        model="openai/gpt-4.1", input="hi", mock_response=MOCK_TEXT, mock_delay=MOCK_DELAY_SECONDS, stream=stream
    )

    assert time.monotonic() - started >= MOCK_DELAY_SECONDS
