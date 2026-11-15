from litellm.litellm_core_utils.upstream_stream_usage import (
    AnthropicStreamUsage,
    recorded_anthropic_stream_event,
)


def _start(usage: object) -> dict[str, object]:
    return {"type": "message_start", "message": {"id": "msg_1", "usage": usage}}


def _delta(usage: object, stop_reason: object = "end_turn") -> dict[str, object]:
    return {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage}


def test_events_are_kept_as_sent_including_absent_and_null_counters() -> None:
    opened = recorded_anthropic_stream_event(None, _start({"input_tokens": 9, "cache_read_input_tokens": None}))
    unrelated = recorded_anthropic_stream_event(opened, {"type": "content_block_delta", "index": 0})
    closed = recorded_anthropic_stream_event(unrelated, _delta({"output_tokens": 2}))
    assert closed == AnthropicStreamUsage(
        opening={"input_tokens": 9, "cache_read_input_tokens": None},
        closing={"output_tokens": 2},
        stop_reason="end_turn",
    )


def test_a_new_message_start_discards_the_previous_messages_usage() -> None:
    first = recorded_anthropic_stream_event(None, _start({"cache_read_input_tokens": 5}))
    finished = recorded_anthropic_stream_event(first, _delta({"output_tokens": 1}))
    restarted = recorded_anthropic_stream_event(finished, _start({"cache_read_input_tokens": 0}))
    assert restarted == AnthropicStreamUsage(opening={"cache_read_input_tokens": 0})
