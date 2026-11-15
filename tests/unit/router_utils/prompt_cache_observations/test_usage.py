import json

import pytest
from pydantic import JsonValue

from litellm.router_utils.prompt_cache_observations.types import (
    CacheCounters,
    UpstreamCacheUsage,
    UsageUnknown,
    UsageUnknownReason,
)
from litellm.litellm_core_utils.upstream_cache_evidence import AnthropicStreamUsage
from litellm.router_utils.prompt_cache_observations.usage import stream_cache_usage, upstream_cache_usage


def _anthropic_body(usage: JsonValue, stop_reason: JsonValue = "end_turn") -> str:
    return json.dumps({"type": "message", "stop_reason": stop_reason, "usage": usage})


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            CacheCounters(read_tokens=0, write_tokens=0),
        ),
        (
            {"input_tokens": 9, "cache_read_input_tokens": 2048, "cache_creation_input_tokens": 0},
            CacheCounters(read_tokens=2048, write_tokens=0),
        ),
        (
            {"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2048},
            CacheCounters(read_tokens=0, write_tokens=2048),
        ),
        ({"input_tokens": 9}, UsageUnknown(UsageUnknownReason.MISSING)),
        (
            {"input_tokens": 9, "cache_read_input_tokens": None, "cache_creation_input_tokens": None},
            UsageUnknown(UsageUnknownReason.MISSING),
        ),
        ({"input_tokens": 9, "cache_read_input_tokens": 0}, UsageUnknown(UsageUnknownReason.PARTIAL)),
        (
            {"input_tokens": 9, "cache_read_input_tokens": None, "cache_creation_input_tokens": 0},
            UsageUnknown(UsageUnknownReason.PARTIAL),
        ),
        (
            {"cache_read_input_tokens": "0", "cache_creation_input_tokens": 0},
            UsageUnknown(UsageUnknownReason.MALFORMED),
        ),
        (
            {"cache_read_input_tokens": -1, "cache_creation_input_tokens": 0},
            UsageUnknown(UsageUnknownReason.MALFORMED),
        ),
        (
            {"cache_read_input_tokens": False, "cache_creation_input_tokens": 0},
            UsageUnknown(UsageUnknownReason.MALFORMED),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "iterations": [{"cache_read_input_tokens": 512, "cache_creation_input_tokens": 0}],
            },
            UsageUnknown(UsageUnknownReason.ITERATED),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            CacheCounters(read_tokens=0, write_tokens=0),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1500,
                "cache_creation": {"ephemeral_5m_input_tokens": 500, "ephemeral_1h_input_tokens": 1000},
            },
            CacheCounters(read_tokens=0, write_tokens=1500),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 700, "ephemeral_1h_input_tokens": 0},
            },
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "cache_creation": "none"},
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 700, "ephemeral_1h_input_tokens": -700},
            },
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 1200,
                "cache_creation": {"ephemeral_5m_input_tokens": 1200},
            },
            CacheCounters(read_tokens=0, write_tokens=1200),
        ),
        (
            {
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 300},
            },
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "cache_creation": {}},
            CacheCounters(read_tokens=0, write_tokens=0),
        ),
        (None, UsageUnknown(UsageUnknownReason.MISSING)),
    ],
)
def test_anthropic_usage_is_authoritative_only_when_both_counters_are_present(
    usage: JsonValue, expected: UpstreamCacheUsage
) -> None:
    assert upstream_cache_usage(_anthropic_body(usage)) == expected


@pytest.mark.parametrize("stop_reason", [None, ""])
def test_anthropic_response_without_a_stop_reason_is_not_final_usage(stop_reason: JsonValue) -> None:
    complete_zeros: dict[str, JsonValue] = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert upstream_cache_usage(_anthropic_body(complete_zeros, stop_reason)) == (
        UsageUnknown(UsageUnknownReason.INCOMPLETE)
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {"stop_reason": "end_turn", "usage": {"cache_read_input_tokens": 3, "cache_creation_input_tokens": 0}},
            CacheCounters(read_tokens=3, write_tokens=0),
        ),
        (
            b'{"stop_reason": "end_turn", "usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 5}}',
            CacheCounters(read_tokens=0, write_tokens=5),
        ),
        ('{"stop_reason": "end_turn", "usage": {', UsageUnknown(UsageUnknownReason.BODY_MALFORMED)),
        ("first stream response received", UsageUnknown(UsageUnknownReason.BODY_MALFORMED)),
        ("[1, 2]", UsageUnknown(UsageUnknownReason.BODY_MALFORMED)),
        (None, UsageUnknown(UsageUnknownReason.BODY_UNAVAILABLE)),
        (object(), UsageUnknown(UsageUnknownReason.BODY_UNAVAILABLE)),
    ],
)
def test_raw_response_must_be_a_json_object_body(raw: object, expected: UpstreamCacheUsage) -> None:
    assert upstream_cache_usage(raw) == expected


@pytest.mark.parametrize(
    ("counters", "active"),
    [
        (CacheCounters(read_tokens=0, write_tokens=0), False),
        (CacheCounters(read_tokens=1, write_tokens=0), True),
        (CacheCounters(read_tokens=0, write_tokens=1), True),
    ],
)
def test_any_positive_cache_read_or_write_counts_as_cache_activity(counters: CacheCounters, active: bool) -> None:
    assert counters.active is active


_ZEROS: dict[str, JsonValue] = {"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def _stream(
    opening: JsonValue = None,
    closing: JsonValue = None,
    final: JsonValue = None,
    starts: int = 1,
    stopped: bool = True,
    out_of_order: bool = False,
    conflicting: bool = False,
) -> AnthropicStreamUsage:
    return AnthropicStreamUsage(
        opening=opening,
        closing=closing,
        final=final,
        stop_reason="end_turn",
        starts=starts,
        stopped=stopped,
        out_of_order=out_of_order,
        conflicting=conflicting,
    )


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (_stream(opening=_ZEROS, closing={"output_tokens": 4}), CacheCounters(read_tokens=0, write_tokens=0)),
        (
            _stream(
                opening={"input_tokens": 9},
                closing={"output_tokens": 4, "cache_read_input_tokens": 2048, "cache_creation_input_tokens": 0},
            ),
            CacheCounters(read_tokens=2048, write_tokens=0),
        ),
        (
            _stream(opening={"input_tokens": 9}, closing={"output_tokens": 4}, final=_ZEROS),
            CacheCounters(read_tokens=0, write_tokens=0),
        ),
        (
            _stream(opening={"input_tokens": 9, "cache_read_input_tokens": None}, closing={"output_tokens": 4}),
            UsageUnknown(UsageUnknownReason.MISSING),
        ),
        (
            _stream(opening={"input_tokens": 9, "cache_read_input_tokens": 0}, closing={"output_tokens": 4}),
            UsageUnknown(UsageUnknownReason.PARTIAL),
        ),
        (
            _stream(opening=_ZEROS, closing={"output_tokens": 4}, stopped=False),
            UsageUnknown(UsageUnknownReason.INCOMPLETE),
        ),
        (_stream(opening=_ZEROS, closing={"output_tokens": 4}, starts=2), UsageUnknown(UsageUnknownReason.INCOMPLETE)),
        (
            _stream(opening=_ZEROS, closing={"output_tokens": 4}, out_of_order=True),
            UsageUnknown(UsageUnknownReason.MALFORMED),
        ),
        (
            _stream(opening=_ZEROS, closing={"output_tokens": 4}, conflicting=True),
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            AnthropicStreamUsage(
                opening=_ZEROS,
                closing={"output_tokens": 4},
                stop_reason="end_turn",
                starts=1,
                stopped=True,
                errored=True,
            ),
            UsageUnknown(UsageUnknownReason.INCOMPLETE),
        ),
    ],
)
def test_anthropic_stream_usage_counts_only_a_completed_stream_and_never_reads_absence_as_zero(
    stream: AnthropicStreamUsage, expected: UpstreamCacheUsage
) -> None:
    assert stream_cache_usage(stream) == expected


def test_a_missing_stream_record_is_never_read_as_usage() -> None:
    assert stream_cache_usage(None) == UsageUnknown(UsageUnknownReason.STREAM_USAGE_NOT_CAPTURED)
