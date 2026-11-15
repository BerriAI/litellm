import json

import pytest
from pydantic import JsonValue

from litellm.router_utils.prompt_cache_observations.types import (
    CacheCounters,
    UpstreamCacheUsage,
    UsageUnknown,
    UsageUnknownReason,
    WireShape,
)
from litellm.litellm_core_utils.upstream_stream_usage import AnthropicStreamUsage
from litellm.router_utils.prompt_cache_observations.usage import anthropic_stream_cache_usage, upstream_cache_usage


def _anthropic_body(usage: JsonValue, stop_reason: JsonValue = "end_turn") -> str:
    return json.dumps({"type": "message", "stop_reason": stop_reason, "usage": usage})


def _converse_body(usage: JsonValue, stop_reason: JsonValue = "end_turn") -> str:
    return json.dumps({"stopReason": stop_reason, "usage": usage})


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
    assert upstream_cache_usage(WireShape.ANTHROPIC_MESSAGES, _anthropic_body(usage)) == expected


@pytest.mark.parametrize("stop_reason", [None, ""])
def test_anthropic_response_without_a_stop_reason_is_not_final_usage(stop_reason: JsonValue) -> None:
    complete_zeros: dict[str, JsonValue] = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert upstream_cache_usage(WireShape.ANTHROPIC_MESSAGES, _anthropic_body(complete_zeros, stop_reason)) == (
        UsageUnknown(UsageUnknownReason.INCOMPLETE)
    )


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"inputTokens": 9, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0},
            CacheCounters(read_tokens=0, write_tokens=0),
        ),
        (
            {"inputTokens": 9, "cacheReadInputTokens": 4096, "cacheWriteInputTokens": 0},
            CacheCounters(read_tokens=4096, write_tokens=0),
        ),
        ({"inputTokens": 9, "outputTokens": 3}, UsageUnknown(UsageUnknownReason.MISSING)),
        (
            {"inputTokens": 9, "cacheReadInputTokenCount": 0, "cacheWriteInputTokenCount": 0},
            UsageUnknown(UsageUnknownReason.MISSING),
        ),
        ({"cacheWriteInputTokens": 0}, UsageUnknown(UsageUnknownReason.PARTIAL)),
        (
            {
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 1200,
                "cacheDetails": [{"ttl": "5m", "inputTokens": 200}, {"ttl": "1h", "inputTokens": 1000}],
            },
            CacheCounters(read_tokens=0, write_tokens=1200),
        ),
        (
            {
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "cacheDetails": [{"ttl": "5m", "inputTokens": 1200}],
            },
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            {
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "cacheDetails": [{"ttl": "5m", "inputTokens": 500}, {"ttl": "1h", "inputTokens": -500}],
            },
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            {"cacheReadInputTokens": 0, "cacheWriteInputTokens": 0, "cacheDetails": [{"ttl": "5m"}]},
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
    ],
)
def test_converse_usage_is_authoritative_only_when_both_counters_are_present(
    usage: JsonValue, expected: UpstreamCacheUsage
) -> None:
    assert upstream_cache_usage(WireShape.BEDROCK_CONVERSE, _converse_body(usage)) == expected


def test_converse_response_without_a_stop_reason_is_not_final_usage() -> None:
    complete_zeros: dict[str, JsonValue] = {"cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
    assert upstream_cache_usage(WireShape.BEDROCK_CONVERSE, _converse_body(complete_zeros, None)) == (
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
    assert upstream_cache_usage(WireShape.ANTHROPIC_MESSAGES, raw) == expected


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


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        (
            AnthropicStreamUsage(
                opening={"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                closing={"output_tokens": 4},
                stop_reason="end_turn",
            ),
            CacheCounters(read_tokens=0, write_tokens=0),
        ),
        (
            AnthropicStreamUsage(
                opening={"input_tokens": 9, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                closing={"output_tokens": 4, "cache_read_input_tokens": 2048, "cache_creation_input_tokens": 0},
                stop_reason="end_turn",
            ),
            UsageUnknown(UsageUnknownReason.CONTRADICTORY),
        ),
        (
            AnthropicStreamUsage(
                opening={"input_tokens": 9},
                closing={"output_tokens": 4, "cache_read_input_tokens": 2048, "cache_creation_input_tokens": 0},
                stop_reason="end_turn",
            ),
            CacheCounters(read_tokens=2048, write_tokens=0),
        ),
        (
            AnthropicStreamUsage(
                opening={"input_tokens": 9, "cache_read_input_tokens": None},
                closing={"output_tokens": 4},
                stop_reason="end_turn",
            ),
            UsageUnknown(UsageUnknownReason.MISSING),
        ),
        (
            AnthropicStreamUsage(
                opening={"input_tokens": 9, "cache_read_input_tokens": 0},
                closing={"output_tokens": 4},
                stop_reason="end_turn",
            ),
            UsageUnknown(UsageUnknownReason.PARTIAL),
        ),
        (
            AnthropicStreamUsage(
                opening={"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, closing=None, stop_reason=None
            ),
            UsageUnknown(UsageUnknownReason.INCOMPLETE),
        ),
    ],
)
def test_stream_usage_comes_from_the_raw_events_and_never_reads_absence_as_zero(
    stream: AnthropicStreamUsage, expected: UpstreamCacheUsage
) -> None:
    assert anthropic_stream_cache_usage(stream) == expected
