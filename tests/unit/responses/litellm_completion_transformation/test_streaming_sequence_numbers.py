"""
Every event emitted by the chat-completions → Responses streaming adapter
must carry a consistent, strictly increasing ``sequence_number`` (issue
#44923): previously most events serialized without one and
``response.output_item.done`` always serialized its default of 1.
"""

from unittest.mock import Mock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class FakeCustomStreamWrapper:
    def __init__(self, chunks):
        self.logging_obj = Mock()
        self._chunks = list(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)


def _make_iterator(chunks):
    return LiteLLMCompletionStreamingIterator(
        model="gpt-5-mini",
        litellm_custom_stream_wrapper=FakeCustomStreamWrapper(chunks),
        request_input=[{"role": "user", "content": "hi"}],
        responses_api_request={},
    )


def _chat_chunks():
    return [
        ModelResponseStream(
            id="c1",
            model="gpt-5-mini",
            choices=[
                StreamingChoices(index=0, delta=Delta(role="assistant", content="Hello")),
            ],
        ),
        ModelResponseStream(
            id="c1",
            model="gpt-5-mini",
            choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")],
        ),
    ]


def _drain_events(iterator):
    events = []
    for event in iterator:
        dump = event.model_dump(exclude_none=True) if hasattr(event, "model_dump") else dict(event)
        events.append(dump)
    return events


def test_every_event_carries_sequence_number():
    events = _drain_events(_make_iterator(_chat_chunks()))
    assert events, "adapter emitted no events"
    missing = [event.get("type") for event in events if "sequence_number" not in event]
    assert missing == [], f"events without sequence_number: {missing}"


def test_sequence_numbers_are_strictly_increasing_from_one():
    events = _drain_events(_make_iterator(_chat_chunks()))
    sequence_numbers = [event["sequence_number"] for event in events]
    assert sequence_numbers[0] == 1
    assert sequence_numbers == list(range(1, len(events) + 1))


def test_done_event_sequence_number_is_not_stuck_at_default():
    events = _drain_events(_make_iterator(_chat_chunks()))
    done_events = [
        event["sequence_number"]
        for event in events
        if str(getattr(event.get("type"), "value", event.get("type", ""))).endswith("response.completed")
    ]
    assert done_events, "no response.completed event emitted"
    # the last emitted event is response.completed; it must not serialize the
    # OutputItemDoneEvent default of 1 after other events consumed numbers
    assert done_events[0] != 1 or len(events) == 1
    # the last emitted event is response.completed; it must not serialize the
    # OutputItemDoneEvent default of 1 after other events consumed numbers
    assert done_events[0] != 1 or len(events) == 1
