from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.proxy.spend_tracking.baseline_accounting import (
    BaselineHistory,
    BaselineObservation,
    advance_baseline_history,
)
from litellm.proxy.spend_tracking.cache_history import capture_cache_request, normalize_cache_usage
from litellm.proxy.spend_tracking.savings import BaselineCostSnapshot, price_baseline_comparison
from litellm.types.utils import ModelInfo, Usage

_PRICES: Final[ModelInfo] = {
    **litellm.get_model_info("gpt-6-astra", "openai"),
    "input_cost_per_token": 0.01,
    "output_cost_per_token": 0.03,
    "cache_read_input_token_cost": 0.001,
    "cache_creation_input_token_cost": 0.017,
}
_PROMPT: Final = "A long stable prefix with some reusable content. " * 1000


def _request(**overrides: object) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": _PROMPT}], **overrides}


def _usage(*, read: int = 0, write: int = 8000) -> Usage:
    return normalize_cache_usage(
        Usage(
            prompt_tokens=8000,
            completion_tokens=20,
            total_tokens=8020,
            prompt_tokens_details={"cached_tokens": read, "cache_creation_tokens": write},
        )
    )


def _observation(
    request: dict[str, object],
    started: float = 10000.0,
    provider: str = "openai",
    baseline_params: Mapping[str, object] | None = None,
) -> BaselineObservation:
    captured: Final = capture_cache_request(request, "gpt-6-astra", provider, _PRICES, baseline_params or {})
    assert captured is not None
    usage: Final = _usage()
    return BaselineObservation(
        request_id=str(started),
        started_at=started,
        available_at=started + 1,
        outcome="complete",
        baseline_equivalent=False,
        usage=usage,
        plan=captured.plan(usage),
        cache_policy="estimated",
        cache_write_pricing="standard",
        assumptions=captured.assumptions,
    )


@pytest.mark.parametrize(
    "provider", ("openai", "azure", "gemini", "vertex_ai", "deepseek", "mistral", "custom_provider")
)
def test_switched_model_pays_actual_cold_write_while_baseline_reads_its_own_history(provider: str) -> None:
    first: Final = _observation(_request(), provider=provider)
    history, initial = advance_baseline_history(BaselineHistory(), (first,))
    _, second = advance_baseline_history(
        history, (first.model_copy(update={"request_id": "switch", "started_at": 10002.0, "available_at": 10003.0}),)
    )
    assert initial[0].usage is not None and second[0].usage is not None
    assert initial[0].usage.prompt_tokens_details.cache_creation_tokens == 8000
    assert getattr(initial[0].usage.prompt_tokens_details, "cache_creation_token_details", None) is None
    assert second[0].usage.prompt_tokens_details.cached_tokens == 8000
    assert second[0].usage.prompt_tokens_details.cache_creation_tokens == 0
    assert first.usage == _usage()
    actual: Final = 8000 * _PRICES["cache_creation_input_token_cost"] + 20 * _PRICES["output_cost_per_token"]
    snapshot: Final = BaselineCostSnapshot(
        model="gpt-6-astra",
        provider=provider,
        prices=_PRICES,
        actual_spend=actual,
        actual_token_cost=actual,
        classifier_cost=0.1,
    )
    priced: Final = price_baseline_comparison(snapshot, second[0].usage, second[0].provenance)
    assert priced is not None
    assert priced.actual == pytest.approx(actual + 0.1)
    assert priced.baseline == pytest.approx(
        8000 * _PRICES["cache_read_input_token_cost"] + 20 * _PRICES["output_cost_per_token"]
    )
    assert priced.savings < 0


@pytest.mark.parametrize("seconds,expected_reads", ((1799, 8000), (1800, 0)))
def test_unspecified_provider_lifetime_is_labeled_and_expires(seconds: int, expected_reads: int) -> None:
    first: Final = _observation(_request(), provider="custom_provider")
    assert "cache_lifetime_assumed_30m" in first.assumptions
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    _, estimates = advance_baseline_history(
        history,
        (
            first.model_copy(
                update={
                    "request_id": "later",
                    "started_at": first.started_at + seconds,
                    "available_at": first.available_at + seconds,
                }
            ),
        ),
    )
    assert estimates[0].usage is not None
    assert estimates[0].usage.prompt_tokens_details.cached_tokens == expected_reads
    assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 8000 - expected_reads


@pytest.mark.parametrize("explicit_ttl,seconds", (("5m", 300), ("1h", 3600)))
@pytest.mark.parametrize("retention,lifetime", ((None, 1800), ("24h", 86400)))
def test_mixed_lifetime_prefixes_reuse_and_expire_independently(
    explicit_ttl: str, seconds: int, retention: str | None, lifetime: int
) -> None:
    request: Final = _request(
        messages=[
            {"role": "user", "content": _PROMPT, "cache_control": {"type": "ephemeral", "ttl": explicit_ttl}},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "A second reusable prefix. " * 200},
        ],
        prompt_cache_retention=retention,
    )
    first: Final = _observation(request)
    assert first.plan is not None and len(first.plan.breakpoints) == 2
    history, cold = advance_baseline_history(BaselineHistory(), (first,))
    assert cold[0].usage is not None, cold[0].reason
    assert cold[0].usage.prompt_tokens_details.cache_creation_tokens == 8000
    for elapsed in (2, seconds, lifetime):
        _, estimates = advance_baseline_history(history, (_observation(request, started=10000.0 + elapsed),))
        usage: Final = estimates[0].usage
        reads: Final = max(
            (marker.prefix_tokens for marker in first.plan.breakpoints if elapsed < marker.ttl_seconds), default=0
        )
        assert usage is not None, estimates[0].reason
        assert usage.prompt_tokens_details.cached_tokens == reads
        assert usage.prompt_tokens_details.cache_creation_tokens == 8000 - reads
        assert getattr(usage.prompt_tokens_details, "cache_creation_token_details", None) is None


@pytest.mark.parametrize("ttl", ("5m", "1h"))
def test_mixed_lifetime_growth_reuses_the_matching_shorter_prefix(ttl: str) -> None:
    marked: Final = {"role": "user", "content": _PROMPT, "cache_control": {"type": "ephemeral", "ttl": ttl}}
    request: Final = _request(messages=[marked, {"role": "user", "content": "first suffix " * 200}])
    first: Final = _observation(request)
    assert first.plan is not None
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    grown: Final = _observation(
        _request(messages=[marked, {"role": "user", "content": "changed suffix " * 200}]), started=10002.0
    )
    _, estimates = advance_baseline_history(history, (grown,))
    usage: Final = estimates[0].usage
    assert usage is not None, estimates[0].reason
    reads: Final = first.plan.breakpoints[0].prefix_tokens
    assert usage.prompt_tokens_details.cached_tokens == reads
    assert usage.prompt_tokens_details.cache_creation_tokens == 8000 - reads


def test_scaled_prefix_counts_in_a_growing_conversation_do_not_invent_a_lifetime_change() -> None:
    messages: Final = [
        {"role": "user", "content": _PROMPT, "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        {"role": "user", "content": "A second prefix. " * 200},
    ]
    first: Final = _observation(_request(messages=messages))
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    grown: Final = _observation(
        _request(messages=[*messages, {"role": "assistant", "content": "OK"}, {"role": "user", "content": "Next"}]),
        started=10002.0,
    )
    _, estimates = advance_baseline_history(history, (grown,))
    usage: Final = estimates[0].usage
    assert usage is not None, estimates[0].reason
    assert usage.prompt_tokens_details.cached_tokens == 8000
    assert usage.prompt_tokens_details.cache_creation_tokens == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"messages": [{"role": "user", "content": "a changed prefix"}]},
        {"tools": [{"type": "function", "function": {"name": "different_tool"}}]},
        {"extra_body": {"prompt_cache_key": "different_key"}},
    ),
)
def test_changed_prefix_or_cache_settings_cannot_hit(changes: dict[str, object]) -> None:
    history, _ = advance_baseline_history(BaselineHistory(), (_observation(_request()),))
    _, estimates = advance_baseline_history(history, (_observation(_request(**changes), 10002.0),))
    assert estimates[0].usage is not None
    assert estimates[0].usage.prompt_tokens_details.cached_tokens == 0


def test_growing_conversation_reuses_only_the_previously_written_prefix() -> None:
    first: Final = _observation(_request())
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    request: Final = _request(
        messages=[
            {"role": "user", "content": _PROMPT},
            {"role": "assistant", "content": "A result"},
            {"role": "user", "content": "Continue"},
        ]
    )
    usage: Final = normalize_cache_usage(Usage(prompt_tokens=8200, completion_tokens=20, total_tokens=8220))
    captured: Final = capture_cache_request(request, "gpt-6-astra", "openai", _PRICES, {})
    assert captured is not None
    _, estimates = advance_baseline_history(
        history, (_observation(request, 10002.0).model_copy(update={"usage": usage, "plan": captured.plan(usage)}),)
    )
    assert estimates[0].usage is not None
    assert estimates[0].usage.prompt_tokens_details.cached_tokens == 8000
    assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 200


def test_explicit_only_mode_does_not_write_without_markers_and_honors_ttl() -> None:
    empty: Final = capture_cache_request(
        _request(prompt_cache_options={"mode": "explicit"}), "gpt-6-astra", "openai", _PRICES, {}
    )
    assert empty is not None and empty.plan(_usage()).breakpoints == ()
    request: Final = _request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT, "prompt_cache_breakpoint": {"mode": "explicit"}},
                    {"type": "text", "text": "uncached suffix"},
                ],
            }
        ],
        prompt_cache_options={"mode": "explicit", "ttl": "1h"},
    )
    captured: Final = capture_cache_request(request, "test-model", "custom_provider", _PRICES, {})
    assert captured is not None
    plan: Final = captured.plan(_usage())
    assert len(plan.breakpoints) == 1
    assert plan.breakpoints[0].ttl_seconds == 3600
    assert 0 < plan.breakpoints[0].prefix_tokens < plan.total_tokens
    assert "cache_lifetime_assumed_30m" not in captured.assumptions
    assert _PROMPT not in repr(captured)


def test_failed_or_overlapping_request_does_not_manufacture_a_cache_hit() -> None:
    first: Final = _observation(_request())
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    _, concurrent = advance_baseline_history(
        history, (first.model_copy(update={"request_id": "concurrent", "started_at": 10000.5}),)
    )
    assert concurrent[0].usage is not None and concurrent[0].usage.prompt_tokens_details.cached_tokens == 0
    failed: Final = first.model_copy(
        update={
            "request_id": "failed",
            "started_at": 10002.0,
            "available_at": 10003.0,
            "outcome": "uncertain",
            "reason": "retried_request",
        }
    )
    invalidated, estimates = advance_baseline_history(history, (failed,))
    assert estimates[0].usage is None
    _, next_request = advance_baseline_history(invalidated, (_observation(_request(), 10004.0),))
    assert next_request[0].usage is None and next_request[0].reason == "history_unavailable"


def test_baseline_deployment_settings_override_selected_model_settings() -> None:
    first: Final = capture_cache_request(
        _request(reasoning_effort="low"), "gpt-6-astra", "openai", _PRICES, {"reasoning_effort": "high"}
    )
    second: Final = capture_cache_request(
        _request(reasoning_effort="high"), "gpt-6-astra", "openai", _PRICES, {"reasoning_effort": "high"}
    )
    assert first is not None and second is not None
    assert first.plan(_usage()) == second.plan(_usage())


def test_response_input_uses_shared_normalization_and_missing_server_history_stays_unknown() -> None:
    request: Final = {"input": [{"role": "user", "content": _PROMPT}]}
    captured: Final = capture_cache_request(request, "gpt-6-astra", "openai", _PRICES, {})
    assert captured is not None and captured.plan(_usage()).breakpoints
    assert (
        capture_cache_request(
            {**request, "previous_response_id": "unseen-history"}, "gpt-6-astra", "openai", _PRICES, {}
        )
        is None
    )


def test_missing_cache_write_rate_uses_ordinary_input_and_zero_rate_stays_free() -> None:
    usage: Final = _usage()
    for write_rate in (None, 0.0):
        prices: Final[ModelInfo] = {**_PRICES, "cache_creation_input_token_cost": write_rate}
        snapshot: Final = BaselineCostSnapshot(
            model="gpt-6-astra", provider="openai", prices=prices, actual_spend=10.0, actual_token_cost=10.0
        )
        priced: Final = price_baseline_comparison(snapshot, usage, "modeled")
        assert priced is not None
        assert priced.baseline == pytest.approx(
            8000 * (prices["input_cost_per_token"] if write_rate is None else write_rate)
            + 20 * prices["output_cost_per_token"]
        )


def test_publication_retains_assumptions_and_generic_writes_after_storage_round_trip() -> None:
    from litellm.proxy.db.baseline_accounting import BaselineAccountingRecord, baseline_publication
    from litellm.proxy.spend_tracking.savings import baseline_cost_snapshot

    observation: Final = _observation(_request(), provider="custom_provider")
    snapshot: Final = baseline_cost_snapshot("gpt-6-astra", _PRICES, 10.0, None, None, "openai", 10.0)
    record: Final = BaselineAccountingRecord(
        scope="autorouter-baseline:v3:" + "a" * 64,
        api_key="test",
        session_id="test",
        router_name="test",
        baseline_model="openai/gpt-6-astra",
        observation=observation,
        pricing=snapshot,
        turn=None,
        daily=None,
    )
    restored: Final = BaselineAccountingRecord.model_validate_json(record.model_dump_json())
    _, estimates = advance_baseline_history(BaselineHistory(), (restored.observation,))
    publication: Final = baseline_publication(restored, estimates[0], observation.started_at)
    assert publication.status == "estimated"
    assert publication.baseline_spend == pytest.approx(
        8000 * _PRICES["cache_creation_input_token_cost"] + 20 * _PRICES["output_cost_per_token"]
    )
    assert publication.cache_creation_input_tokens == 8000
    assert publication.cache_creation_5m_input_tokens is None and publication.cache_creation_1h_input_tokens is None
    assert "cache_lifetime_assumed_30m" in publication.assumptions
    reported: Final = baseline_cost_snapshot(
        "gpt-6-astra", _PRICES, 10.0, {"input_cost": 7.0, "output_cost": 2.0}, None, "openai", 10.0
    )
    assert reported.actual_token_cost == 9.0


def test_openai_lookup_keeps_initial_developer_group_but_limits_earlier_user_endings() -> None:
    initial: Final = [{"role": "developer", "content": "first instruction"}, {"role": "developer", "content": _PROMPT}]
    later: Final = [{"role": "user", "content": str(index)} for index in range(25)]
    first: Final = capture_cache_request(_request(messages=initial), "gpt-6-astra", "openai", _PRICES, {})
    grown: Final = capture_cache_request(_request(messages=initial + later), "gpt-6-astra", "openai", _PRICES, {})
    user: Final = capture_cache_request(_request(messages=later[:1]), "gpt-6-astra", "openai", _PRICES, {})
    users: Final = capture_cache_request(_request(messages=later), "gpt-6-astra", "openai", _PRICES, {})
    assert first is not None and grown is not None and user is not None and users is not None
    assert (
        first.plan(_usage()).breakpoints[-1].fingerprint in grown.plan(_usage()).breakpoints[-1].lookback_fingerprints
    )
    assert (
        user.plan(_usage()).breakpoints[-1].fingerprint
        not in users.plan(_usage()).breakpoints[-1].lookback_fingerprints
    )


def test_multipart_tool_envelope_is_counted_once_in_bounded_chunks() -> None:
    from queue import SimpleQueue

    from litellm.proxy.spend_tracking.cache_history import prepare_cache_request

    samples: Final[SimpleQueue[str]] = SimpleQueue()
    arguments: Final = "X" * 32768
    request: Final = _request(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call", "type": "function", "function": {"name": "tool", "arguments": arguments}}
                ],
                "content": [{"type": "text", "text": "part"} for _ in range(1000)],
            }
        ]
    )

    def count(model: str, text: str) -> int:
        samples.put(text)
        return len(text)

    prepared: Final = prepare_cache_request(request, "gpt-6-astra", "openai", _PRICES, {})
    assert prepared is not None
    captured: Final = prepared.count("gpt-6-astra", count)
    counted: Final = tuple(samples.get_nowait() for _ in range(samples.qsize()))
    assert sum(part.count("X") for part in counted) == len(arguments)
    assert max(map(len, counted)) <= 8192
    assert captured.weight < 100000
    assert arguments not in repr(captured) and arguments not in repr(prepared)


@pytest.mark.parametrize("location", ("messages", "input", "system", "tools", "extra_body", "baseline"))
def test_oversized_fields_are_rejected_before_tokenization(location: str) -> None:
    from litellm.proxy.spend_tracking.cache_history import prepare_cache_request

    large: Final = "x" * 750000
    request: Final = {
        "messages": [{"role": "user", "content": "small"}],
        **({location: [{"role": "user", "content": large}]} if location in ("messages", "input") else {}),
        **({location: large} if location in ("system", "tools") else {}),
        **({"extra_body": {"system": large}} if location == "extra_body" else {}),
    }
    baseline: Final = {"system": large} if location == "baseline" else {}
    assert prepare_cache_request(request, "gpt-6-astra", "openai", _PRICES, baseline) is None


@pytest.mark.parametrize("shape", ("blocks", "nodes", "depth", "integer"))
def test_small_text_cannot_bypass_estimator_structure_budgets(shape: str) -> None:
    from functools import reduce

    from litellm.proxy.spend_tracking.cache_history import prepare_cache_request

    request: Final = _request(
        messages=[{"role": "user", "content": [{} for _ in range(2049)]}]
        if shape == "blocks"
        else [{"role": "user", "content": "small"}],
        tools={str(index): 0 for index in range(12000)}
        if shape == "nodes"
        else reduce(lambda value, _: {"value": value}, range(40), {})
        if shape == "depth"
        else {"oversized_integer": 1 << 10000}
        if shape == "integer"
        else [],
    )
    assert prepare_cache_request(request, "gpt-6-astra", "openai", _PRICES, {}) is None


def test_deferred_capture_owns_a_snapshot_before_provider_mutation() -> None:
    from litellm.proxy.spend_tracking.cache_history import prepare_cache_request

    content: Final = [{"type": "text", "text": _PROMPT, "cache_control": {"type": "ephemeral"}}]
    request: Final = _request(messages=[{"role": "user", "content": content}])
    prepared: Final = prepare_cache_request(request, "gpt-6-astra", "openai", _PRICES, {})
    assert prepared is not None
    original: Final = prepared.count("gpt-6-astra").plan(_usage())
    content.clear()
    assert prepared.count("gpt-6-astra").plan(_usage()) == original


@pytest.mark.parametrize("surface", ("messages", "input"))
@pytest.mark.parametrize("marker", ("cache_control", "prompt_cache_breakpoint"))
def test_message_cache_marker_covers_all_multipart_content(surface: str, marker: str) -> None:
    from pydantic import JsonValue

    control: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    content: Final[list[dict[str, JsonValue]]] = [
        {"type": "text", "text": "stable prefix"},
        {"type": "text", "text": "marked content " * 1000},
    ]
    message: Final = {"role": "user", "content": content, marker: control}
    request: Final = {surface: [message], "prompt_cache_options": {"mode": "explicit", "ttl": "1h"}}
    captured: Final = capture_cache_request(request, "test-model", "custom_provider", _PRICES, {})
    normalized: Final = capture_cache_request(
        {**request, surface: [{"role": "user", "content": [content[0], {**content[1], marker: control}]}]},
        "test-model",
        "custom_provider",
        _PRICES,
        {},
    )
    changed: Final = capture_cache_request(
        {**request, surface: [{**message, "content": [content[0], {**content[1], "text": "different content"}]}]},
        "test-model",
        "custom_provider",
        _PRICES,
        {},
    )
    assert captured is not None and normalized is not None and changed is not None
    plan: Final = captured.plan(_usage())
    assert plan == normalized.plan(_usage())
    assert len(plan.breakpoints) == 1
    assert plan.breakpoints[0].prefix_tokens == plan.total_tokens
    assert plan.breakpoints[0].ttl_seconds == 3600
    first: Final = BaselineObservation(
        request_id="first",
        started_at=10000,
        available_at=10001,
        outcome="complete",
        baseline_equivalent=False,
        usage=_usage(),
        plan=plan,
        cache_policy="estimated",
        cache_write_pricing="standard",
    )
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    for prefix, reads in ((normalized, 8000), (changed, 0)):
        _, estimates = advance_baseline_history(
            history,
            (
                first.model_copy(
                    update={
                        "request_id": "next",
                        "started_at": 10002,
                        "available_at": 10003,
                        "plan": prefix.plan(_usage()),
                    }
                ),
            ),
        )
        assert estimates[0].usage is not None
        assert estimates[0].usage.prompt_tokens_details.cached_tokens == reads
        assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 8000 - reads


@pytest.mark.parametrize("surface", ("messages", "input"))
def test_message_cache_marker_covers_string_content_parts(surface: str) -> None:
    captured: Final = capture_cache_request(
        {surface: [{"role": "user", "content": ["first", "last"], "cache_control": {"type": "ephemeral"}}]},
        "claude-test",
        "custom_provider",
        _PRICES,
        {},
    )
    assert captured is not None
    plan: Final = captured.plan(_usage())
    assert len(plan.breakpoints) == 1
    assert plan.breakpoints[0].prefix_tokens == plan.total_tokens


def test_message_marker_matches_provider_precedence_without_losing_earlier_breakpoints() -> None:
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
    from litellm.types.llms.openai import AllMessageValues

    messages: Final[list[AllMessageValues]] = [
        {
            "role": "user",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "content": [
                {"type": "text", "text": "earlier prefix", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "remaining content", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ],
        },
    ]
    transformed: Final = OpenrouterConfig()._move_cache_control_to_content(messages)
    before: Final = capture_cache_request({"messages": messages}, "claude-test", "openrouter", _PRICES, {})
    after: Final = capture_cache_request({"messages": transformed}, "claude-test", "openrouter", _PRICES, {})
    assert before is not None and after is not None
    plan: Final = before.plan(_usage())
    assert plan == after.plan(_usage())
    assert tuple(marker.ttl_seconds for marker in plan.breakpoints) == (3600, 300)
    assert plan.breakpoints[-1].prefix_tokens == plan.total_tokens
    assert "cache_control" in messages[0]


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    "baseline_params", ({}, {"reasoning_effort": "medium"}, {"extra_body": {"thinking": {"type": "disabled"}}})
)
@pytest.mark.parametrize(
    "tier_settings",
    (
        {"reasoning_effort": "high"},
        {"reasoning": {"effort": "high"}},
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
    ),
)
def test_routed_tier_settings_do_not_reset_baseline_history(
    nested: bool,
    baseline_params: dict[str, object],
    tier_settings: dict[str, object],
) -> None:
    history, _ = advance_baseline_history(
        BaselineHistory(), (_observation(_request(), baseline_params=baseline_params),)
    )
    request: Final = _request(**({"extra_body": tier_settings} if nested else tier_settings))
    _, estimates = advance_baseline_history(history, (_observation(request, 10002.0, baseline_params=baseline_params),))
    assert estimates[0].usage is not None
    assert estimates[0].usage.prompt_tokens_details.cached_tokens == 8000
    assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 0


@pytest.mark.parametrize(
    "baseline_params",
    (
        {"reasoning_effort": "high"},
        {"extra_body": {"reasoning": {"effort": "high"}}},
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
    ),
)
def test_baseline_model_settings_still_invalidate_prefixes(baseline_params: dict[str, object]) -> None:
    history, _ = advance_baseline_history(BaselineHistory(), (_observation(_request()),))
    _, estimates = advance_baseline_history(
        history, (_observation(_request(), 10002.0, baseline_params=baseline_params),)
    )
    assert estimates[0].usage is not None
    assert estimates[0].usage.prompt_tokens_details.cached_tokens == 0
    assert estimates[0].usage.prompt_tokens_details.cache_creation_tokens == 8000
