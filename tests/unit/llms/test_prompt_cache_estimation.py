from dataclasses import replace
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.prompt_cache_estimation import estimate_cache_plan, normalize_cache_usage, prepare_cache_request
from litellm.proxy.spend_tracking.baseline_accounting import (
    BaselineHistory,
    BaselineObservation,
    advance_baseline_history,
)
from litellm.proxy.spend_tracking.savings import BaselineCostSnapshot, _baseline_usage, price_baseline_comparison
from litellm.router_utils.baseline_request import NATIVE_ONLY_PARAMETERS
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES
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


@pytest.mark.parametrize("supplied", (None, [], ["unresolved"], {"unexpected": "input"}))
def test_unsupported_input_skips_plan_without_discarding_observed_usage(supplied: JsonValue) -> None:
    usage: Final = _usage()
    assert estimate_cache_plan({"input": supplied}, "gpt-6-astra", "openai", _PRICES, usage) is None
    assert usage == _usage()


def test_plain_responses_input_remains_estimatable() -> None:
    captured: Final = estimate_cache_plan({"input": _PROMPT}, "gpt-6-astra", "openai", _PRICES, _usage())
    assert captured is not None
    assert captured.plan.total_tokens == _usage().prompt_tokens


def _observation(
    request: dict[str, object],
    started: float = 10000.0,
    provider: str = "openai",
) -> BaselineObservation:
    prepared: Final = prepare_cache_request(request)
    assert prepared is not None
    usage: Final = _usage()
    captured: Final = estimate_cache_plan(
        prepared, "gpt-6-astra", provider, _PRICES, usage, lambda model, text: len(text)
    )
    assert captured is not None
    return BaselineObservation(
        request_id=str(started),
        started_at=started,
        available_at=started + 1,
        outcome="complete",
        baseline_equivalent=False,
        usage=usage,
        plan=captured.plan,
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


@pytest.mark.parametrize("surface,kind", (("messages", "text"), ("input", "input_text")))
@pytest.mark.parametrize("ttl,seconds", (("5m", 300), ("1h", 3600)))
def test_message_marker_covers_the_last_content_block(surface: str, kind: str, ttl: str, seconds: int) -> None:
    request: Final = {
        surface: [
            {
                "role": "user",
                "cache_control": {"type": "ephemeral", "ttl": ttl},
                "content": [{"type": kind, "text": _PROMPT}, {"type": kind, "text": "suffix"}],
            }
        ],
        "prompt_cache_options": {"mode": "explicit"},
    }
    observation: Final = _observation(request)
    assert observation.plan is not None
    assert len(observation.plan.breakpoints) == 1
    marker: Final = observation.plan.breakpoints[0]
    assert (marker.ttl_seconds, marker.prefix_tokens) == (seconds, observation.usage.prompt_tokens)


@pytest.mark.parametrize("provider,control", (("anthropic", "5m"), ("bedrock", "1h")))
def test_duration_pricing_and_top_level_cache_control_use_configured_rates(provider: str, control: str) -> None:
    prices: Final[ModelInfo] = {**_PRICES, "cache_creation_input_token_cost_above_1hr": 0.024}
    raw: Final = prepare_cache_request(_request(cache_control={"type": "ephemeral", "ttl": control}))
    assert raw is not None
    estimated: Final = estimate_cache_plan(
        raw, "custom-assistant", provider, prices, _usage(), lambda model, text: len(text)
    )
    assert estimated is not None
    observation: Final = _observation(_request()).model_copy(
        update={"plan": estimated.plan, "cache_write_pricing": "duration"}
    )
    _, estimates = advance_baseline_history(BaselineHistory(), (observation,))
    usage: Final = estimates[0].usage
    assert usage is not None
    assert estimated.plan.breakpoints[0].ttl_seconds == (300 if control == "5m" else 3600)
    snapshot: Final = BaselineCostSnapshot(
        model="gpt-6-astra", provider="openai", prices=prices, actual_spend=1.0, actual_token_cost=1.0
    )
    priced: Final = price_baseline_comparison(snapshot, usage, estimates[0].provenance)
    assert priced is not None
    rate: Final = (
        prices["cache_creation_input_token_cost"]
        if control == "5m"
        else prices["cache_creation_input_token_cost_above_1hr"]
    )
    assert priced.baseline == pytest.approx(8000 * rate + 20 * prices["output_cost_per_token"])


def test_json_schema_fields_are_part_of_identity_and_oversized_requests_are_rejected() -> None:
    first: Final = _observation(
        _request(response_format={"json_schema": {"properties": {"cache_control": {"const": "first"}}}})
    )
    changed: Final = _observation(
        _request(response_format={"json_schema": {"properties": {"cache_control": {"const": "changed"}}}})
    )
    assert first.plan is not None and changed.plan is not None
    assert first.plan.breakpoints != changed.plan.breakpoints
    assert prepare_cache_request(_request(messages=[{"role": "user", "content": "x" * 5_000_000}])) is None


@pytest.mark.parametrize(
    "key,block", (("system", {"type": "text", "text": _PROMPT}), ("tools", {"name": "lookup", "description": _PROMPT}))
)
def test_explicit_system_and_tool_cache_writes_are_reused(key: str, block: dict[str, str]) -> None:
    first: Final = _observation(
        _request(**{key: [{**block, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}), provider="anthropic"
    )
    history, initial = advance_baseline_history(BaselineHistory(), (first,))
    _, replay = advance_baseline_history(
        history, (first.model_copy(update={"request_id": "replay", "started_at": 10002.0, "available_at": 10003.0}),)
    )
    assert initial[0].usage is not None and replay[0].usage is not None
    writes: Final = initial[0].usage.prompt_tokens_details.cache_creation_tokens
    assert 0 < writes < first.usage.prompt_tokens
    assert replay[0].usage.prompt_tokens_details.cached_tokens == writes
    assert replay[0].usage.prompt_tokens_details.cache_creation_tokens == 0


@pytest.mark.parametrize("carrier", ("native", "chat_outer", "chat_nested", "chat_both"))
@pytest.mark.parametrize("ttl,seconds", (("5m", 300), ("1h", 3600)))
def test_tool_cache_controls_warm_the_marked_prefix_with_requested_lifetime(
    carrier: str, ttl: str, seconds: int
) -> None:
    control: Final = {"type": "ephemeral", "ttl": ttl}
    definition: Final = {"name": "lookup", "description": _PROMPT, "parameters": {"type": "object"}}
    tool: Final = (
        {"name": "lookup", "description": _PROMPT, "input_schema": {"type": "object"}, "cache_control": control}
        if carrier == "native"
        else {
            "type": "function",
            "function": {
                **definition,
                **(
                    {"cache_control": {"type": "ephemeral", "ttl": "1h" if ttl == "5m" else "5m"}}
                    if carrier == "chat_both"
                    else {"cache_control": control}
                    if carrier == "chat_nested"
                    else {}
                ),
            },
            **({"cache_control": control} if carrier in ("chat_outer", "chat_both") else {}),
        }
    )
    outgoing: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
        AnthropicConfig().map_openai_params({"tools": [tool]}, {}, "claude-opus-5-5", False)["tools"]
    )
    assert outgoing[0]["cache_control"] == control
    first: Final = _observation(
        _request(tools=[tool, {"name": "following", "description": _PROMPT}], system=_PROMPT),
        provider="anthropic",
    )
    assert first.plan is not None and len(first.plan.breakpoints) == 1
    marker: Final = first.plan.breakpoints[0]
    assert marker.ttl_seconds == seconds
    assert 0 < marker.prefix_tokens < first.usage.prompt_tokens
    history, cold = advance_baseline_history(BaselineHistory(), (first,))
    _, warm = advance_baseline_history(
        history, (first.model_copy(update={"request_id": "warm", "started_at": 10002.0, "available_at": 10003.0}),)
    )
    assert cold[0].usage is not None and warm[0].usage is not None
    assert cold[0].usage.prompt_tokens_details.cache_creation_tokens == marker.prefix_tokens
    assert warm[0].usage.prompt_tokens_details.cached_tokens == marker.prefix_tokens
    assert warm[0].usage.prompt_tokens_details.cache_creation_tokens == 0


def test_nested_tool_marker_keeps_only_content_before_its_boundary_in_cache_identity() -> None:
    definition: Final = {
        "name": "lookup",
        "description": _PROMPT,
        "parameters": {"type": "object", "properties": {"cache_control": {"const": "original"}}},
    }
    marked: Final = {
        "type": "function",
        "function": {**definition, "cache_control": {"type": "ephemeral", "ttl": "5m"}},
    }
    first: Final = _observation(
        _request(tools=[{"name": "earlier"}, marked, {"name": "following"}], system="system"), provider="anthropic"
    )
    assert first.plan is not None and len(first.plan.breakpoints) == 1
    original: Final = first.plan.breakpoints[0].fingerprint
    for earlier, current, suffix, equivalent in (
        ({"name": "earlier"}, marked, "changed", True),
        (
            {"name": "earlier"},
            {"type": "function", "function": definition, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            "changed",
            True,
        ),
        ({"name": "changed"}, marked, "following", False),
        (
            {"name": "earlier"},
            {
                "type": "function",
                "function": {
                    **definition,
                    "parameters": {"type": "object", "properties": {"cache_control": {"const": "changed"}}},
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
            },
            "following",
            False,
        ),
    ):
        changed: Final = _observation(
            _request(
                tools=[earlier, current, {"name": suffix}],
                system=suffix,
                messages=[{"role": "user", "content": suffix}],
            ),
            provider="anthropic",
        )
        assert changed.plan is not None and len(changed.plan.breakpoints) == 1
        assert (changed.plan.breakpoints[0].fingerprint == original) is equivalent


@pytest.mark.parametrize("kind", sorted(ANTHROPIC_TOOL_SEARCH_TOOL_TYPES))
@pytest.mark.parametrize("nested", (False, True))
def test_search_tool_markers_match_the_effective_provider_mapping(kind: str, nested: bool) -> None:
    control: Final = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    tool: Final = {"type": kind, "name": "search", **({"function": control} if nested else control)}
    outgoing: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_python(
        AnthropicConfig().map_openai_params({"tools": [tool]}, {}, "claude-opus-5-5", False)["tools"]
    )
    system: Final = [{"type": "text", "text": _PROMPT, **control}]
    observed: Final = _observation(_request(tools=[tool], system=system), provider="anthropic")
    expected: Final = _observation(_request(tools=outgoing, system=system), provider="anthropic")
    assert observed.plan is not None
    assert len(observed.plan.breakpoints) == 1 + sum(bool(tool.get("cache_control")) for tool in outgoing)
    assert observed.plan == expected.plan


@pytest.mark.parametrize("surface", ("chat", "responses", "native"))
@pytest.mark.parametrize("mode", ("configured", "request", "global"))
@pytest.mark.parametrize("ttl,seconds", (("5m", 300), ("1h", 3600)))
def test_injected_baseline_markers_are_reused_with_the_requested_lifetime(
    monkeypatch: pytest.MonkeyPatch, surface: str, mode: str, ttl: str, seconds: int
) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", mode == "global")
    monkeypatch.setattr(litellm, "anthropic_prompt_caching_ttl", ttl)
    controls: Final = (
        {
            "cache_control_injection_points": [
                {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": ttl}},
                {"location": "message", "role": "user", "control": {"type": "ephemeral", "ttl": ttl}},
            ]
        }
        if mode == "configured"
        else {"enable_prompt_caching": mode == "request"}
    )
    request: Final = {
        "chat": {"messages": [{"role": "system", "content": _PROMPT}, {"role": "user", "content": "question"}]},
        "responses": {"instructions": _PROMPT, "input": [{"role": "user", "content": "question"}]},
        "native": {"system": _PROMPT, "messages": [{"role": "user", "content": "question"}]},
    }[surface]
    prepared: Final = prepare_cache_request(
        {**request, **controls}, "claude-opus-5-5", "anthropic", native=surface == "native"
    )
    assert prepared is not None
    estimated: Final = estimate_cache_plan(
        prepared, "claude-opus-5-5", "anthropic", _PRICES, _usage(), lambda model, text: len(text)
    )
    assert estimated is not None
    assert len(estimated.plan.breakpoints) == 2
    assert all(marker.ttl_seconds == seconds for marker in estimated.plan.breakpoints)
    first: Final = _observation(_request()).model_copy(update={"plan": estimated.plan})
    history, cold = advance_baseline_history(BaselineHistory(), (first,))
    _, warm = advance_baseline_history(
        history, (first.model_copy(update={"request_id": "warm", "started_at": 10002.0, "available_at": 10003.0}),)
    )
    assert cold[0].usage is not None and warm[0].usage is not None
    assert cold[0].usage.prompt_tokens_details.cache_creation_tokens == first.usage.prompt_tokens
    assert warm[0].usage.prompt_tokens_details.cached_tokens == first.usage.prompt_tokens
    assert warm[0].usage.prompt_tokens_details.cache_creation_tokens == 0


def test_responses_ordinal_injection_targets_input_before_instructions() -> None:
    request: Final = {
        "instructions": _PROMPT,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "question"}]}],
        "cache_control_injection_points": [{"location": "message", "index": 0}],
    }
    prepared: Final = prepare_cache_request(request, "claude-opus-5-5", "anthropic")
    assert prepared is not None
    estimated: Final = estimate_cache_plan(
        prepared, "claude-opus-5-5", "anthropic", _PRICES, _usage(), lambda model, text: len(text)
    )
    assert estimated is not None and len(estimated.plan.breakpoints) == 1
    assert estimated.plan.breakpoints[0].prefix_tokens == _usage().prompt_tokens
    assert "cache_control" not in str(request["input"])


@pytest.mark.parametrize("configured", (False, True))
def test_injected_native_cache_respects_client_marks_and_the_shared_cap(configured: bool) -> None:
    control: Final = {"type": "ephemeral", "ttl": "1h"}
    request: Final = _request(
        system=_PROMPT,
        enable_prompt_caching=True,
        tools=[
            {"name": f"tool_{index}", "input_schema": {"type": "object"}, "cache_control": control}
            for index in range(3)
        ],
        **(
            {
                "cache_control_injection_points": [
                    {"location": "message", "role": "system"},
                    {"location": "message", "index": -1},
                ]
            }
            if configured
            else {}
        ),
    )
    prepared: Final = prepare_cache_request(request, "claude-opus-5-5", "anthropic", native=True)
    assert prepared is not None
    estimated: Final = estimate_cache_plan(
        prepared, "claude-opus-5-5", "anthropic", _PRICES, _usage(), lambda model, text: len(text)
    )
    assert estimated is not None and len(estimated.plan.breakpoints) == (4 if configured else 3)
    assert all(marker.ttl_seconds == 3600 for marker in estimated.plan.breakpoints[:3])
    assert request["system"] == _PROMPT


@pytest.mark.parametrize("user_agent", ("claude-cli/2.1.263 (external, cli)", "ordinary-client"))
def test_cache_preparation_matches_native_subagent_policy_without_retaining_headers(user_agent: str) -> None:
    from litellm.llms.anthropic.pass_through.messages.utils import prepare_native_messages

    request: Final = {
        "messages": [{"role": "user", "content": "unique document"}],
        "system": "x-anthropic-billing-header: cc_is_subagent=true;",
        "enable_prompt_caching": True,
        "proxy_server_request": {"headers": {"User-Agent": user_agent, "authorization": "private-token"}},
    }
    messages, system = prepare_native_messages(
        [{"role": "user", "content": "unique document"}],
        "x-anthropic-billing-header: cc_is_subagent=true;",
        {"enable_prompt_caching": True, "proxy_server_request": request["proxy_server_request"]},
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
    )
    prepared: Final = prepare_cache_request(request, "claude-opus-5-5", "anthropic", native=True)
    assert prepared is not None
    assert (prepared["messages"], prepared["system"]) == (messages, system)
    assert "proxy_server_request" not in prepared and "private-token" not in str(prepared)


@pytest.mark.parametrize("field", NATIVE_ONLY_PARAMETERS)
def test_opaque_native_options_are_not_silently_ignored_by_estimates(field: str) -> None:
    prepared: Final = prepare_cache_request(_request(**{field: "opaque"}))
    assert prepared is not None
    assert estimate_cache_plan(prepared, "claude-opus-5-5", "anthropic", _PRICES, _usage()) is None


@pytest.mark.parametrize("modality", ("audio_tokens", "image_tokens", "video_tokens"))
def test_multimodal_estimates_keep_history_and_price_cold_warm_expired_tokens(modality: str) -> None:
    usage: Final = normalize_cache_usage(
        Usage(
            prompt_tokens=8000,
            completion_tokens=20,
            total_tokens=8020,
            prompt_tokens_details={modality: 2000, "cached_tokens": 0, "cache_creation_tokens": 0},
        )
    )
    first: Final = _observation(_request(), provider="gemini").model_copy(update={"usage": usage})
    history, cold = advance_baseline_history(BaselineHistory(), (first,))
    warmed, warm = advance_baseline_history(
        history, (first.model_copy(update={"request_id": "warm", "started_at": 10002.0, "available_at": 10003.0}),)
    )
    _, expired = advance_baseline_history(
        warmed, (first.model_copy(update={"request_id": "expired", "started_at": 12000.0, "available_at": 12001.0}),)
    )
    prices: Final[ModelInfo] = {
        **_PRICES,
        "cache_read_input_audio_token_cost": 0.002,
        "input_cost_per_audio_token": 0.02,
        "input_cost_per_image_token": 0.03,
        "input_cost_per_video_token": 0.04,
    }
    snapshot: Final = BaselineCostSnapshot(
        model="test-model", provider="gemini", prices=prices, actual_spend=100.0, actual_token_cost=100.0
    )
    for estimate, cost in (
        (cold[0], 8000 * 0.017 + 20 * 0.03),
        (warm[0], 6000 * 0.001 + 2000 * (0.002 if modality == "audio_tokens" else 0.001) + 20 * 0.03),
        (expired[0], 8000 * 0.017 + 20 * 0.03),
    ):
        assert estimate.usage is not None, estimate.reason
        assert (priced := price_baseline_comparison(snapshot, estimate.usage, estimate.provenance)) is not None
        assert priced.baseline == pytest.approx(cost)


@pytest.mark.parametrize(
    "reason", ("estimation_capacity_exhausted", "baseline_estimation_timeout", "unsupported_cache_request")
)
def test_skipped_estimation_preserves_paid_cache_without_refreshing_it(reason: str) -> None:
    first: Final = _observation(_request())
    history, _ = advance_baseline_history(BaselineHistory(), (first,))
    skipped: Final = first.model_copy(
        update={"request_id": "skipped", "started_at": 10002.0, "available_at": 10003.0, "plan": None, "reason": reason}
    )
    retained, missing = advance_baseline_history(history, (skipped,))
    assert missing[0].usage is None
    assert retained.entries == history.entries
    _, followup = advance_baseline_history(
        retained, (first.model_copy(update={"request_id": "followup", "started_at": 10004.0, "available_at": 10005.0}),)
    )
    assert followup[0].usage is not None and followup[0].usage.prompt_tokens_details.cached_tokens == 8000
    _, expired = advance_baseline_history(
        retained, (first.model_copy(update={"request_id": "expired", "started_at": 11800.0, "available_at": 11801.0}),)
    )
    assert expired[0].usage is not None and expired[0].usage.prompt_tokens_details.cached_tokens == 0


def test_partial_multimodal_cache_replaces_observed_splits_without_double_charging() -> None:
    usage: Final = normalize_cache_usage(
        Usage(
            prompt_tokens=8000,
            completion_tokens=20,
            total_tokens=8020,
            prompt_tokens_details={
                "audio_tokens": 2000,
                "image_tokens": 1000,
                "video_tokens": 500,
                "cached_tokens": 2000,
                "cached_tokens_details": {
                    "audio_tokens": 800,
                    "image_tokens": 300,
                    "text_tokens": 900,
                },
            },
        )
    )
    original: Final = _observation(_request(), provider="gemini")
    assert original.plan is not None
    plan: Final = replace(original.plan, breakpoints=(replace(original.plan.breakpoints[0], prefix_tokens=4000),))
    first: Final = original.model_copy(update={"usage": usage, "plan": plan})
    history, cold = advance_baseline_history(BaselineHistory(), (first,))
    _, warm = advance_baseline_history(
        history,
        (
            first.model_copy(
                update={
                    "request_id": "warm",
                    "started_at": 10002.0,
                    "available_at": 10003.0,
                }
            ),
        ),
    )
    prices: Final[ModelInfo] = {
        **_PRICES,
        "cache_read_input_audio_token_cost": 0.002,
        "input_cost_per_audio_token": 0.02,
        "input_cost_per_image_token": 0.03,
        "input_cost_per_video_token": 0.04,
    }
    snapshot: Final = BaselineCostSnapshot(
        model="test-model",
        provider="gemini",
        prices=prices,
        actual_spend=100.0,
        actual_token_cost=100.0,
    )
    ordinary: Final = 2250 * 0.01 + 1000 * 0.02 + 500 * 0.03 + 250 * 0.04 + 20 * 0.03
    observed: Final = sum(
        generic_cost_per_token("test-model", _baseline_usage(usage, prices), "gemini", model_info=prices)
    )
    assert observed == pytest.approx(
        3600 * 0.01 + 1200 * 0.02 + 700 * 0.03 + 500 * 0.04 + 1200 * 0.001 + 800 * 0.002 + 20 * 0.03
    )
    for value, expected in (
        (cold[0].usage, ordinary + 4000 * 0.017),
        (warm[0].usage, ordinary + 3000 * 0.001 + 1000 * 0.002),
    ):
        assert value is not None
        assert (priced := price_baseline_comparison(snapshot, value, "modeled")) is not None
        assert priced.baseline == pytest.approx(expected)
    assert usage.prompt_tokens_details.cached_tokens_details.audio_tokens == 800
