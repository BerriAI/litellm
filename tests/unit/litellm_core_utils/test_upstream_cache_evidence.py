import copy
import json
from collections.abc import Mapping
from functools import reduce

from litellm.litellm_core_utils.upstream_cache_evidence import (
    AnthropicStreamUsage,
    anthropic_stream_usage_from_sse,
    recorded_anthropic_stream_event,
    recorded_anthropic_stream_text,
)

_ZEROS = {"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def _start(usage: object = _ZEROS) -> Mapping[str, object]:
    return {"type": "message_start", "message": {"id": "msg_1", "usage": usage}}


def _delta(usage: object, stop_reason: object = "end_turn") -> Mapping[str, object]:
    return {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage}


_STOP: Mapping[str, object] = {"type": "message_stop"}


def _recorded(*events: Mapping[str, object]) -> AnthropicStreamUsage | None:
    return reduce(recorded_anthropic_stream_event, events, None)


def test_a_complete_stream_keeps_each_usage_object_as_sent() -> None:
    assert _recorded(
        _start({"input_tokens": 9, "cache_read_input_tokens": None}),
        {"type": "content_block_delta", "index": 0},
        _delta({"output_tokens": 2}),
        _STOP,
    ) == AnthropicStreamUsage(
        opening={"input_tokens": 9, "cache_read_input_tokens": None},
        closing={"output_tokens": 2},
        stop_reason="end_turn",
        starts=1,
        stopped=True,
    )


def test_a_stream_cut_after_its_final_delta_is_not_stopped() -> None:
    recorded = _recorded(_start(), _delta({"output_tokens": 2}))
    assert recorded is not None and (recorded.stop_reason, recorded.stopped) == ("end_turn", False)


def test_events_after_message_stop_and_restarts_are_out_of_order() -> None:
    late_delta = _recorded(_start(), _delta({"output_tokens": 2}), _STOP, _delta({"output_tokens": 3}))
    second_start = _recorded(_start(), _delta({"output_tokens": 2}), _start())
    stop_without_delta = _recorded(_start(), _STOP)
    assert [record.out_of_order if record else None for record in (late_delta, second_start, stop_without_delta)] == [
        True,
        True,
        True,
    ]
    assert second_start is not None and second_start.starts == 2


def test_a_ping_after_message_stop_is_not_a_late_event() -> None:
    recorded = _recorded(_start(), _delta({"output_tokens": 2}), _STOP, {"type": "ping"})
    assert recorded is not None and recorded.out_of_order is False


def test_deltas_or_stop_usage_that_change_a_cache_counter_conflict() -> None:
    changed_delta = _recorded(
        _start(), _delta({"cache_read_input_tokens": 0}), _delta({"cache_read_input_tokens": 512}), _STOP
    )
    stop_disagrees = _recorded(
        _start(),
        _delta({"output_tokens": 2}),
        {"type": "message_stop", "usage": {"cache_creation_input_tokens": 1024}},
    )
    repeated = _recorded(
        _start(), _delta({"cache_read_input_tokens": 0}), _delta({"cache_read_input_tokens": 0}), _STOP
    )
    assert [record.conflicting if record else None for record in (changed_delta, stop_disagrees, repeated)] == [
        True,
        True,
        False,
    ]


def _sse(events: tuple[Mapping[str, object], ...]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def test_sse_frames_fold_into_the_same_record_as_parsed_events() -> None:
    events = (_start(), {"type": "ping"}, _delta({"output_tokens": 2}), _STOP)
    assert anthropic_stream_usage_from_sse(_sse(events)) == _recorded(*events)


def test_sse_frames_without_lifecycle_events_record_nothing() -> None:
    assert anthropic_stream_usage_from_sse(_sse(({"type": "ping"},))) is None
    assert anthropic_stream_usage_from_sse(b"") is None


def test_records_survive_the_deep_copy_logging_makes_of_call_details() -> None:
    record = _recorded(_start(), _delta({"output_tokens": 2}), _STOP)
    assert copy.deepcopy({"record": record}) == {"record": record}


def test_an_error_event_marks_the_stream_even_when_a_stop_follows() -> None:
    error: Mapping[str, object] = {"type": "error", "error": {"type": "overloaded_error"}}
    recorded = _recorded(_start(), _delta({"output_tokens": 2}), error, _STOP)
    from_frames = anthropic_stream_usage_from_sse(_sse((_start(), _delta({"output_tokens": 2}), error, _STOP)))
    assert recorded is not None and recorded.errored is True
    assert from_frames == recorded


def test_an_sse_comment_quoting_a_stop_is_not_an_event() -> None:
    frames = _sse((_start(), _delta({"output_tokens": 2}))) + b': data: {"type": "message_stop"}\n\n'
    recorded = anthropic_stream_usage_from_sse(frames)
    assert recorded is not None and recorded.stopped is False


def test_lifecycle_text_that_is_not_a_json_event_leaves_the_record_unchanged() -> None:
    started = _recorded(_start())
    assert recorded_anthropic_stream_text(started, '{"type": "message_stop"') == started


def test_usage_that_is_not_json_is_kept_as_absent_rather_than_failing_the_stream() -> None:
    recorded = _recorded(_start(object()), _delta(object()), _STOP)
    assert recorded is not None
    assert (recorded.opening, recorded.closing, recorded.starts, recorded.stopped) == (None, None, 1, True)
