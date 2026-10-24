import json

import pytest

from litellm.proxy.guardrails.stream_surface import (
    StreamSurface,
    classify_stream,
    is_terminal_error_stream,
)
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def _chat_chunks() -> tuple[ModelResponseStream, ...]:
    return (
        ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content="hi"))],
        ),
    )


def _anthropic_sse_frames() -> tuple[bytes, ...]:
    message_start = {
        "type": "message_start",
        "message": {"id": "msg_1", "model": "claude-haiku-4-5", "content": [], "usage": {"input_tokens": 3}},
    }
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    return (
        f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode(),
        f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode(),
    )


def _google_sse_frames() -> tuple[bytes, ...]:
    """The :streamGenerateContent route marks its stream raw too, but is not Anthropic."""
    payload = {"candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}]}
    return (f"data: {json.dumps(payload)}\n\n".encode(),)


def _responses_events() -> tuple[object, ...]:
    return (
        OutputTextDeltaEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
            item_id="msg_1",
            output_index=0,
            content_index=0,
            delta="hi",
        ),
        ResponseCompletedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
            response=ResponsesAPIResponse(
                id="resp_1",
                created_at=1,
                model="claude-haiku-4-5",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
                top_p=1.0,
            ),
        ),
    )


@pytest.mark.parametrize(
    "chunks, expected",
    [
        (_chat_chunks(), StreamSurface.CHAT_COMPLETIONS),
        (_anthropic_sse_frames(), StreamSurface.ANTHROPIC_MESSAGES),
        (_responses_events(), StreamSurface.RESPONSES),
        (_google_sse_frames(), StreamSurface.OPAQUE_SSE),
        ((), StreamSurface.CHAT_COMPLETIONS),
    ],
)
def test_classify_stream_names_each_wire_format(chunks, expected):
    assert classify_stream(chunks) is expected


def test_opaque_sse_is_not_mistaken_for_anthropic():
    """Reading a Google stream as Anthropic would refuse it in a format its client cannot parse."""
    assert classify_stream(_google_sse_frames()) is not StreamSurface.ANTHROPIC_MESSAGES


def test_responses_events_are_classified_from_a_plain_dict_too():
    """Some producers hand the hook already-serialized events rather than the pydantic models."""
    assert classify_stream(({"type": "response.output_text.delta", "delta": "hi"},)) is StreamSurface.RESPONSES


def test_a_responses_error_only_stream_is_a_terminal_error_stream():
    events = (
        ErrorEvent(
            type=ResponsesAPIStreamEvents.ERROR,
            sequence_number=1,
            error=ErrorEventError(type="guardrail_error", code="400", message="blocked upstream"),
        ),
    )
    assert is_terminal_error_stream(events) is True


def test_an_anthropic_error_only_stream_is_a_terminal_error_stream():
    frames = (b'event: error\ndata: {"type": "error", "error": {"message": "blocked upstream"}}\n\n',)
    assert is_terminal_error_stream(frames) is True


def test_a_stream_carrying_content_is_not_a_terminal_error_stream():
    assert is_terminal_error_stream(_responses_events()) is False
    assert is_terminal_error_stream(_chat_chunks()) is False
    assert is_terminal_error_stream(_anthropic_sse_frames()) is False


def test_an_empty_stream_is_not_a_terminal_error_stream():
    """all() over nothing is True, so an empty stream would otherwise read as somebody's refusal."""
    assert is_terminal_error_stream(()) is False
