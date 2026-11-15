import copy
import json
from collections.abc import Mapping
from functools import reduce

from pydantic import JsonValue

from litellm.litellm_core_utils.upstream_cache_evidence import (
    AnthropicStreamUsage,
    ConverseStreamUsage,
    anthropic_stream_usage_from_sse,
    recorded_anthropic_stream_event,
    recorded_converse_stream_text,
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


def _converse(*events: Mapping[str, object]) -> ConverseStreamUsage | None:
    return reduce(recorded_converse_stream_text, (json.dumps(event) for event in events), None)


def test_converse_records_the_stop_and_metadata_usage_as_sent() -> None:
    usage: dict[str, JsonValue] = {"inputTokens": 9, "outputTokens": 2, "totalTokens": 11}
    assert _converse(
        {"role": "assistant"},
        {"contentBlockIndex": 0, "delta": {"text": "ok"}},
        {"stopReason": "end_turn"},
        {"usage": usage, "metrics": {"latencyMs": 5}},
    ) == ConverseStreamUsage(stop_reason="end_turn", usage=usage, stops=1, usage_events=1)


def test_converse_usage_before_the_stop_is_out_of_order() -> None:
    recorded = _converse({"usage": {"inputTokens": 9}}, {"stopReason": "end_turn"})
    assert recorded is not None and recorded.out_of_order is True


def test_converse_text_that_is_not_json_leaves_the_record_unchanged() -> None:
    assert recorded_converse_stream_text(None, '{"stopReason": ') is None


def test_records_survive_the_deep_copy_logging_makes_of_call_details() -> None:
    records = (
        _recorded(_start(), _delta({"output_tokens": 2}), _STOP),
        _converse({"stopReason": "end_turn"}, {"usage": {"inputTokens": 9}}),
    )
    assert copy.deepcopy({"records": records}) == {"records": records}
