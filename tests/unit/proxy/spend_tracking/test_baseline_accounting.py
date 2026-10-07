from dataclasses import replace
from itertools import groupby
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.cost_calculation import cost_per_token
from litellm.llms.anthropic.prompt_cache_prediction import CountedBreakpoint, CountedPromptCachePlan
from litellm.proxy.spend_tracking.baseline_accounting import (
    BaselineEstimate,
    BaselineHistory,
    BaselineObservation,
    CacheEntry,
    advance_baseline_history,
)
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage


def _usage() -> Usage:
    return Usage(
        prompt_tokens=6200,
        completion_tokens=30,
        total_tokens=6230,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=6000,
        speed="fast",
        inference_geo="us",
        completion_tokens_details={"reasoning_tokens": 20},
        server_tool_use={"web_search_requests": 1},
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=200,
            cached_tokens=0,
            cache_creation_tokens=6000,
            cache_write_tokens=6000,
            cache_creation_token_details=CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=6000
            ),
        ),
    )


def _marker(
    name: str = "prefix", ttl: int = 3600, tokens: int = 6000, previous: tuple[str, ...] = ()
) -> CountedBreakpoint:
    return CountedBreakpoint(
        fingerprint=f"{name}:{ttl}",
        ttl_seconds=ttl,
        prefix_tokens=tokens,
        lookback_fingerprints=(*(f"{item}:{ttl}" for item in previous), f"{name}:{ttl}"),
        content_fingerprint=name,
        lookback_content_fingerprints=(*previous, name),
    )


def _observation(request_id: str, started: float = 10000.0, **overrides: object) -> BaselineObservation:
    return BaselineObservation.model_validate(
        {
            "request_id": request_id,
            "started_at": started,
            "available_at": started + 0.1,
            "outcome": "complete",
            "baseline_equivalent": False,
            "usage": _usage(),
            "plan": CountedPromptCachePlan(6200, (_marker(),)),
            "minimum_cache_tokens": 4096,
            **overrides,
        }
    )


def _replay(*observations: BaselineObservation) -> tuple[BaselineEstimate, ...]:
    history = BaselineHistory()
    results: list[BaselineEstimate] = []
    for _, group in groupby(sorted(observations, key=lambda item: item.started_at), key=lambda item: item.started_at):
        history, estimates = advance_baseline_history(history, tuple(group))
        results.extend(estimates)
    return tuple(results)


def test_initial_identical_path_preserves_full_usage_without_counting_or_exclusive_owner() -> None:
    initial: Final = _observation("main", baseline_equivalent=True, plan=None, reason="unsupported_request_headers")
    background: Final = initial.model_copy(update={"request_id": "background"})
    later: Final = initial.model_copy(update={"request_id": "later", "started_at": 10001.0, "available_at": 10002.0})
    estimates: Final = _replay(initial, background, later)
    assert all(item.provenance == "observed_identical" and item.usage == initial.usage for item in estimates)
    assert all(item.usage is not initial.usage for item in estimates)
    assert all(item.usage.prompt_tokens == 6200 for item in estimates if item.usage is not None)


def test_late_divergent_observation_replays_in_event_order_and_removes_initial_zero() -> None:
    same: Final = _observation("same", 10001.0, baseline_equivalent=True)
    early: Final = _observation("early")
    assert _replay(same)[0].provenance == "observed_identical"
    replayed: Final = _replay(same, early)
    assert replayed == _replay(early, same)
    assert replayed[0].usage is None
    assert replayed[1].provenance == "modeled"
    assert replayed[1].usage is not None and replayed[1].usage.prompt_tokens_details.cached_tokens == 6000


@pytest.mark.parametrize("ttl", [300, 3600])
def test_prefix_match_expiry_and_usage_pricing_fields(ttl: int) -> None:
    plan: Final = CountedPromptCachePlan(6200, (_marker(ttl=ttl),))
    first: Final = _observation("first", baseline_equivalent=True, plan=plan)
    # Each replay starts from the original observation, so warm does not refresh the expiry case.
    warm: Final = _replay(first, _observation("warm", 10000.0 + ttl - 0.01, plan=plan))[-1]
    cold: Final = _replay(first, _observation("cold", 10000.0 + ttl, plan=plan))[-1]
    assert warm.reason == "cache_prefix_available" and cold.reason == "cache_prefix_expired"
    assert warm.usage is not None and cold.usage is not None
    assert warm.usage.prompt_tokens_details.cached_tokens == 6000
    assert cold.usage.prompt_tokens_details.cached_tokens == 0
    assert cold.usage.prompt_tokens_details.cache_creation_tokens == 6000
    unaffected: Final = {"prompt_tokens", "total_tokens", "prompt_tokens_details", "cache_read_input_tokens", "cache_creation_input_tokens"}
    assert warm.usage.model_dump(exclude=unaffected) == first.usage.model_dump(exclude=unaffected)
    assert cold.usage.model_dump(exclude=unaffected) == first.usage.model_dump(exclude=unaffected)


@pytest.mark.parametrize("warm_tail", (False, True))
def test_growth_lookback_and_mixed_ttl_keep_distinct_read_write_buckets(warm_tail: bool) -> None:
    first: Final = _observation("first", baseline_equivalent=True)
    grown: Final = CountedPromptCachePlan(7100, (_marker("grown", 3600, 6500, ("prefix",)), _marker("tail", 300, 7000)))
    # Initial unseen suffixes remain unknown within their potential pre-existing cache horizon.
    second: Final = _replay(first, _observation("second", 10001.0, plan=grown))[-1]
    assert second.reason == "history_unavailable"
    history: Final = BaselineHistory(
        first_at=1.0, last_at=10000.0, equivalent=False, uncertain_before=1.0,
        entries=(CacheEntry("tail:300", "tail", 7000, 300, 10000.0, 10300.0),) if warm_tail else (),
    )
    _, estimates = advance_baseline_history(history, (_observation("mixed", 10001.0, plan=grown),))
    usage: Final = estimates[0].usage
    assert usage is not None
    assert usage.prompt_tokens_details.text_tokens == 100
    # Anthropic billing locations: B is the highest 1h breakpoint AFTER the highest hit A.
    # https://platform.claude.com/docs/en/build-with-claude/prompt-caching#mixing-different-ttls (2026-09-15)
    assert usage.prompt_tokens_details.cached_tokens == (7000 if warm_tail else 0)
    assert usage.prompt_tokens_details.cache_creation_token_details.ephemeral_1h_input_tokens == (0 if warm_tail else 6500)
    assert usage.prompt_tokens_details.cache_creation_token_details.ephemeral_5m_input_tokens == (0 if warm_tail else 500)


@pytest.mark.parametrize("change", ["prefix", "ttl", "unavailable", "failed", "response_cache"])
def test_uncertainty_and_replays_do_not_manufacture_hits(change: str) -> None:
    first: Final = _observation("first", baseline_equivalent=True)
    changes: Final = {
        "prefix": {"plan": CountedPromptCachePlan(6200, (_marker("changed"),))},
        "ttl": {"plan": CountedPromptCachePlan(6200, (_marker(ttl=300),))},
        "unavailable": {"plan": None, "reason": "token_count_unavailable"},
        "failed": {"outcome": "uncertain", "reason": "incomplete_response"},
        "response_cache": {"outcome": "response_cache"},
    }
    second: Final = _observation("second", 10001.0, **changes[change])
    third: Final = _observation("third", 10002.0)
    middle, result = _replay(first, second, third)[1:]
    assert middle.usage is None
    if change in ("unavailable", "failed", "ttl"):
        assert result.usage is None
    else:
        assert result.usage is not None and result.usage.prompt_tokens_details.cached_tokens == 6000


def test_first_token_availability_and_simultaneous_divergence_are_conservative() -> None:
    slow: Final = _observation("slow", available_at=10002.0, baseline_equivalent=True)
    overlap: Final = _observation("overlap", 10001.0)
    assert _replay(slow, overlap)[-1].usage is None
    assert all(item.provenance != "observed_identical" for item in _replay(slow, _observation("tie")))


def test_invalid_usage_and_invalid_count_plan_cannot_seed_cache() -> None:
    bad: Final = _observation("bad", baseline_equivalent=True, usage=_usage().model_copy(update={"total_tokens": 1}))
    assert all(item.usage is None for item in _replay(bad, _observation("next", 10001.0)))
    broken: Final = CountedPromptCachePlan(6200, (replace(_marker(), prefix_tokens=7000),))
    assert _replay(_observation("bad", plan=broken))[0].usage is None


def test_overlapping_uncertain_request_cannot_be_warmed_by_a_later_callback() -> None:
    uncertain: Final = _observation("incomplete", outcome="uncertain", available_at=10010.0)
    overlap: Final = _observation("overlap", 10001.0)
    during: Final = _observation("during", 10002.0)
    after: Final = _observation("after", 10011.0)
    warmed: Final = _observation("warmed", 10012.0)
    estimates: Final = _replay(uncertain, overlap, during, after, warmed)
    assert estimates[1].reason == estimates[2].reason == "concurrent_uncertainty"
    assert estimates[3].usage is None
    assert estimates[4].usage is not None and estimates[4].usage.prompt_tokens_details.cached_tokens == 6000


def test_modeled_read_cannot_recharge_the_original_private_write_count() -> None:
    warm: Final = _replay(_observation("initial", baseline_equivalent=True), _observation("warm", 10001.0))[-1]
    assert warm.usage is not None
    prices: Final = {
        **litellm.get_model_info("claude-opus-5", custom_llm_provider="anthropic"),
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 1e-7,
        "cache_creation_input_token_cost": 1.25e-6,
        "provider_specific_entry": {"fast": 2.0, "us": 1.1},
    }
    input_cost, output_cost = cost_per_token("claude-opus-5", warm.usage, model_info=prices)
    assert input_cost + output_cost == pytest.approx((200 * 1e-6 + 6000 * 1e-7 + 30 * 2e-6) * 2.0 * 1.1)
