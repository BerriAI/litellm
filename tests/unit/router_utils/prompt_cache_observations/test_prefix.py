from collections.abc import Mapping
from types import MappingProxyType

import pytest
from pydantic import JsonValue

from litellm.constants import REDACTED_BY_LITELLM
from litellm.router_utils.prompt_cache_observations.prefix import outgoing_cacheable_prefix
from litellm.router_utils.prompt_cache_observations.types import (
    CacheablePrefix,
    NoCacheBreakpoint,
    PrefixEvidence,
    PrefixUnassessed,
    UnassessedReason,
    WireShape,
)

_KEY = b"prompt-cache-observation-test-key"
_EPHEMERAL: Mapping[str, JsonValue] = {"type": "ephemeral"}


def _text(text: str, cache_control: JsonValue = None) -> dict[str, JsonValue]:
    return {"type": "text", "text": text, **({"cache_control": cache_control} if cache_control is not None else {})}


def _anthropic(
    messages: list[JsonValue],
    system: JsonValue = None,
    tools: JsonValue = None,
    envelope: Mapping[str, JsonValue] = MappingProxyType({}),
) -> dict[str, JsonValue]:
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "messages": messages,
        **({"system": system} if system is not None else {}),
        **({"tools": tools} if tools is not None else {}),
        **envelope,
    }


def _anthropic_prefix(body: Mapping[str, JsonValue], key: bytes = _KEY) -> CacheablePrefix:
    evidence = outgoing_cacheable_prefix(WireShape.ANTHROPIC_MESSAGES, body, key)
    assert isinstance(evidence, CacheablePrefix), evidence
    return evidence


def _marked_turn(question: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": [_text("long shared document", dict(_EPHEMERAL))]},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": question},
    ]


def test_content_after_the_last_breakpoint_does_not_change_the_prefix() -> None:
    first = _anthropic_prefix(_anthropic(_marked_turn("first question"), system="be terse"))
    second = _anthropic_prefix(_anthropic(_marked_turn("a different question"), system="be terse"))
    assert first == second


@pytest.mark.parametrize(
    "changed",
    [
        _anthropic(_marked_turn("q"), system="be verbose"),
        _anthropic(_marked_turn("q"), system="be terse", tools=[{"name": "lookup", "input_schema": {}}]),
        _anthropic(_marked_turn("q"), system="be terse", envelope={"tool_choice": {"type": "any"}}),
        _anthropic(
            _marked_turn("q"), system="be terse", envelope={"thinking": {"type": "enabled", "budget_tokens": 1024}}
        ),
        _anthropic(
            [{"role": "user", "content": [_text("other shared document", dict(_EPHEMERAL))]}], system="be terse"
        ),
    ],
)
def test_any_change_before_the_breakpoint_changes_the_prefix(changed: dict[str, JsonValue]) -> None:
    baseline = _anthropic_prefix(_anthropic(_marked_turn("q"), system="be terse"))
    assert _anthropic_prefix(changed).fingerprint != baseline.fingerprint


def test_images_of_equal_size_with_different_bytes_are_different_prefixes() -> None:
    def image_turn(data: str) -> list[JsonValue]:
        image: dict[str, JsonValue] = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        }
        return [{"role": "user", "content": [image, _text("describe", dict(_EPHEMERAL))]}]

    first = _anthropic_prefix(_anthropic(image_turn("A" * 4096 + "B")))
    second = _anthropic_prefix(_anthropic(image_turn("A" * 4096 + "C")))
    assert first.fingerprint != second.fingerprint


def test_fingerprint_is_keyed_so_it_cannot_be_recomputed_without_the_key() -> None:
    body = _anthropic(_marked_turn("q"))
    assert _anthropic_prefix(body, b"key-one").fingerprint != _anthropic_prefix(body, b"key-two").fingerprint


def test_ttl_is_the_last_breakpoints_not_the_longest() -> None:
    body = _anthropic(_marked_turn("q"), system=[_text("rules", {"type": "ephemeral", "ttl": "1h"})])
    assert _anthropic_prefix(body).ttl_seconds == 300


def test_one_hour_ttl_on_the_last_breakpoint_is_kept() -> None:
    messages: list[JsonValue] = [{"role": "user", "content": [_text("doc", {"type": "ephemeral", "ttl": "1h"})]}]
    assert _anthropic_prefix(_anthropic(messages)).ttl_seconds == 3600


def test_breakpoint_on_a_tool_ends_the_prefix_before_system_and_messages() -> None:
    tools: list[JsonValue] = [{"name": "lookup", "input_schema": {}, "cache_control": dict(_EPHEMERAL)}]
    first = _anthropic_prefix(_anthropic([{"role": "user", "content": "one"}], system="a", tools=tools))
    second = _anthropic_prefix(_anthropic([{"role": "user", "content": "two"}], system="b", tools=tools))
    assert first == second


def test_top_level_cache_control_covers_the_whole_prompt() -> None:
    first = _anthropic_prefix(
        _anthropic([{"role": "user", "content": "one"}], envelope={"cache_control": dict(_EPHEMERAL)})
    )
    second = _anthropic_prefix(
        _anthropic([{"role": "user", "content": "two"}], envelope={"cache_control": dict(_EPHEMERAL)})
    )
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_anthropic([{"role": "user", "content": "no breakpoint"}]), NoCacheBreakpoint()),
        (
            _anthropic([{"role": "user", "content": [_text("doc", {"type": "ephemeral", "ttl": "10m"})]}]),
            PrefixUnassessed(UnassessedReason.TTL_UNRECOGNIZED),
        ),
        (
            _anthropic([{"role": "user", "content": [_text("doc", {"type": "persistent"})]}]),
            PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED),
        ),
        (
            _anthropic([{"role": "user", "content": "doc", "cache_control": dict(_EPHEMERAL)}]),
            PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED),
        ),
        (_anthropic([]), PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED)),
        (
            _anthropic([{"role": "user", "content": [_text(REDACTED_BY_LITELLM, dict(_EPHEMERAL))]}]),
            PrefixUnassessed(UnassessedReason.REQUEST_REDACTED),
        ),
    ],
)
def test_anthropic_prefix_that_cannot_be_compared_is_not_assessed(
    body: dict[str, JsonValue], expected: PrefixEvidence
) -> None:
    assert outgoing_cacheable_prefix(WireShape.ANTHROPIC_MESSAGES, body, _KEY) == expected


def _converse(system: list[JsonValue], question: str, tools: list[JsonValue] | None = None) -> dict[str, JsonValue]:
    return {
        "system": system,
        "messages": [{"role": "user", "content": [{"text": question}]}],
        "inferenceConfig": {"maxTokens": 64},
        **({"toolConfig": {"tools": tools}} if tools is not None else {}),
    }


def _converse_evidence(body: Mapping[str, JsonValue]) -> PrefixEvidence:
    return outgoing_cacheable_prefix(WireShape.BEDROCK_CONVERSE, body, _KEY)


def _converse_prefix(body: Mapping[str, JsonValue]) -> CacheablePrefix:
    evidence = _converse_evidence(body)
    assert isinstance(evidence, CacheablePrefix), evidence
    return evidence


def test_converse_cache_point_marks_the_block_before_it() -> None:
    system: list[JsonValue] = [{"text": "rules"}, {"cachePoint": {"type": "default"}}]
    first = _converse_prefix(_converse(system, "one"))
    second = _converse_prefix(_converse(system, "two"))
    changed = _converse_prefix(_converse([{"text": "other rules"}, {"cachePoint": {"type": "default"}}], "one"))
    assert first == second
    assert first.ttl_seconds == 300
    assert first.fingerprint != changed.fingerprint


def test_converse_cache_point_ttl_is_read_from_the_last_cache_point() -> None:
    system: list[JsonValue] = [{"text": "rules"}, {"cachePoint": {"type": "default", "ttl": "1h"}}]
    tools: list[JsonValue] = [{"toolSpec": {"name": "lookup"}}, {"cachePoint": {"type": "default"}}]
    assert _converse_prefix(_converse(system, "q", tools)).ttl_seconds == 3600


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ([{"text": "rules"}], NoCacheBreakpoint()),
        (
            [{"cachePoint": {"type": "default"}}, {"text": "rules"}],
            PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED),
        ),
        ([{"text": "rules"}, {"cachePoint": None}], PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED)),
        (
            [{"text": "rules"}, {"cachePoint": {"type": "default"}}, {"cachePoint": {"type": "default"}}],
            PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED),
        ),
        (
            [{"text": "rules"}, {"cachePoint": {"type": "ephemeral"}}],
            PrefixUnassessed(UnassessedReason.BREAKPOINT_UNRECOGNIZED),
        ),
        (
            [{"text": "rules"}, {"cachePoint": {"type": "default", "ttl": "2h"}}],
            PrefixUnassessed(UnassessedReason.TTL_UNRECOGNIZED),
        ),
    ],
)
def test_converse_prefix_that_cannot_be_compared_is_not_assessed(
    system: list[JsonValue], expected: PrefixEvidence
) -> None:
    assert _converse_evidence(_converse(system, "q")) == expected


def test_output_settings_do_not_split_an_identical_prefix() -> None:
    short = _anthropic(_marked_turn("q"), system="be terse", envelope={"max_tokens": 16, "temperature": 0.2})
    long = _anthropic(_marked_turn("q"), system="be terse", envelope={"max_tokens": 4096, "stream": True})
    assert _anthropic_prefix(short) == _anthropic_prefix(long)


def test_breakpoints_are_counted_so_aggregate_usage_is_not_read_as_proof_for_the_last_one() -> None:
    system: list[JsonValue] = [_text("rules", dict(_EPHEMERAL))]
    tools: list[JsonValue] = [{"name": "lookup", "input_schema": {}, "cache_control": dict(_EPHEMERAL)}]
    assert _anthropic_prefix(_anthropic(_marked_turn("q"), system=system, tools=tools)).breakpoints == 3
    assert _anthropic_prefix(_anthropic(_marked_turn("q"))).breakpoints == 1
    assert (
        _anthropic_prefix(_anthropic(_marked_turn("q"), envelope={"cache_control": dict(_EPHEMERAL)})).breakpoints == 2
    )
