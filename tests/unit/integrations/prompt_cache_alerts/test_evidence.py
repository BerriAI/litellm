import json
from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.constants import REDACTED_BY_LITELLM
from litellm.integrations.prompt_cache_alerts.evidence import (
    CacheablePrefix,
    CacheCounters,
    NotCaching,
    PrefixEvidence,
    Unsupported,
    cache_counters,
    cacheable_prefix,
    fingerprint_key,
)

_KEY: Final = fingerprint_key("evidence-test-salt")
_MARK: Final[dict[str, JsonValue]] = {"type": "ephemeral"}


def _body(system_mark: JsonValue = _MARK, question: str = "what changed?", **extra: JsonValue) -> dict[str, JsonValue]:
    system_block: Final[dict[str, JsonValue]] = {"type": "text", "text": "shared policy"}
    return {
        "model": "claude-sonnet-5-5",
        "max_tokens": 64,
        "system": [{**system_block, "cache_control": system_mark} if system_mark is not None else system_block],
        "messages": [{"role": "user", "content": question}],
        **extra,
    }


def _raw(usage: JsonValue, stop_reason: JsonValue = "end_turn") -> str:
    return json.dumps({"type": "message", "stop_reason": stop_reason, "usage": usage})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_raw({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}), CacheCounters(0, 0)),
        (_raw({"cache_read_input_tokens": 512, "cache_creation_input_tokens": 0}), CacheCounters(512, 0)),
        (
            _raw({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 9, "cache_creation": {"a": 4, "b": 5}}),
            CacheCounters(0, 9),
        ),
        (_raw({"input_tokens": 12}), None),
        (_raw({"cache_read_input_tokens": 0}), None),
        (_raw({"cache_read_input_tokens": None, "cache_creation_input_tokens": 0}), None),
        (_raw({"cache_read_input_tokens": -1, "cache_creation_input_tokens": 0}), None),
        (_raw({"cache_read_input_tokens": True, "cache_creation_input_tokens": 0}), None),
        (_raw({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 9, "cache_creation": {"a": 1}}), None),
        (_raw({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, stop_reason=None), None),
        ('{"usage": {', None),
        ("first stream response received", None),
        (None, None),
    ],
)
def test_only_counters_the_provider_reported_are_authoritative(raw: object, expected: CacheCounters | None) -> None:
    assert cache_counters(raw) == expected


def _prefix(body: Mapping[str, JsonValue], key: bytes = _KEY) -> str:
    evidence: Final = cacheable_prefix(dict(body), key)
    assert isinstance(evidence, CacheablePrefix)
    return evidence.fingerprint


def test_content_after_the_breakpoint_and_output_settings_keep_the_prefix() -> None:
    assert _prefix(_body()) == _prefix(_body(question="something else", max_tokens=8, temperature=0.2))


@pytest.mark.parametrize(
    "changed",
    [
        _body() | {"system": [{"type": "text", "text": "other policy", "cache_control": _MARK}]},
        _body(tools=[{"name": "lookup", "input_schema": {}}]),
        _body(thinking={"type": "enabled", "budget_tokens": 1024}),
    ],
)
def test_anything_before_the_breakpoint_changes_the_prefix(changed: dict[str, JsonValue]) -> None:
    assert _prefix(changed) != _prefix(_body())


def test_the_prefix_is_keyed_so_it_cannot_be_recomputed_without_the_shared_salt() -> None:
    assert _prefix(_body()) != _prefix(_body(), fingerprint_key("another-salt"))


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_body(system_mark=None), NotCaching()),
        (_body(system_mark={"type": "ephemeral", "ttl": "1h"}), Unsupported()),
        (_body(system_mark={"type": "persistent"}), Unsupported()),
        (_body(cache_control={"type": "ephemeral"}), Unsupported()),
        (
            _body(messages=[{"role": "user", "content": [{"type": "text", "text": "q", "cache_control": _MARK}]}]),
            Unsupported(),
        ),
        (_body(messages=[{"role": "user", "content": "q", "cache_control": _MARK}]), Unsupported()),
        (
            _body(system_mark={"type": "ephemeral"}, tools=[{"name": "t", "input_schema": {}, "cache_control": _MARK}]),
            Unsupported(),
        ),
        (_body(messages=[{"role": "user", "content": REDACTED_BY_LITELLM}]), CacheablePrefix(_prefix(_body()))),
        (_body() | {"system": [{"type": "text", "text": REDACTED_BY_LITELLM, "cache_control": _MARK}]}, Unsupported()),
        (_body(messages=[]), Unsupported()),
        ("not json", Unsupported()),
    ],
    ids=[
        "no-breakpoint",
        "one-hour",
        "unknown-type",
        "automatic",
        "two-breakpoints",
        "message-level",
        "tool-and-system",
        "redaction-after-breakpoint",
        "redacted-prefix",
        "no-messages",
        "unreadable",
    ],
)
def test_only_one_explicit_five_minute_breakpoint_is_supported(body: object, expected: PrefixEvidence) -> None:
    assert cacheable_prefix(body, _KEY) == expected
