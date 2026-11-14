"""
Unit tests for TPM rate limit for concurrent requests

Verifies token-reservation pattern:
- Concurrent requests cannot all observe "under limit" before any of them
  has incremented the counter (atomic reservation via
  ``atomic_check_and_increment_by_n``).
- After a successful request, the counter is reconciled to actual usage.
- After a failed request, the full reservation is released.

The reservation path delegates atomicity to ``atomic_check_and_increment_by_n``,
which uses Redis Lua when available and an asyncio-locked in-memory check
otherwise. These tests exercise the in-memory fallback so they run without
Redis.
"""

import asyncio
import base64
import copy
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, Dict, Final

import pytest
import soundfile as sf
from fastapi import HTTPException
from PIL import Image

from litellm import get_model_info, token_counter
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.litellm_core_utils.token_counter import UnsupportedContentTypeError, estimate_image_tokens
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    DEFAULT_AUDIO_TOKEN_ESTIMATE,
    PROJECT_ITPM_DESCRIPTOR_KEY,
    _request_stash,
    get_or_create_request_stash,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.redis.stores.rate_limits import CounterIncrement
from litellm.types.utils import ModelResponse, Usage
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits
from tests._support.rust_parallel_request_limiter import (
    PROXY_MaxParallelRequestsHandler_v3 as RateLimitHandler,
)
from tests._support.rust_parallel_request_limiter import (
    estimate_input_and_output_tokens,
    estimate_tokens_for_request,
    get_request_stash,
)


@pytest.fixture(autouse=True)
def _native_parallel_request_limiter(monkeypatch):
    monkeypatch.setenv("LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER", "true")


@pytest.fixture
def rate_limiter():
    cache = DualCache()
    handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
    return handler, cache


@dataclass
class PipelineRecordingRateLimits(FakeRateLimits):
    connected: bool = False
    pipelined: list[CounterIncrement] = field(default_factory=list)

    async def increment_many(self, increments: Sequence[CounterIncrement]) -> None:
        self.pipelined.extend(increments)
        await super().increment_many(increments)


@pytest.fixture
def recording_rate_limiter():
    cache = DualCache()
    rate_limits = PipelineRecordingRateLimits()
    handler = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache),
        rate_limit_store=rate_limits.store(),
    )
    return handler, cache, rate_limits.pipelined


@pytest.fixture(autouse=True)
def _isolated_request_stash():
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


@pytest.mark.asyncio
async def test_token_reservation_prevents_concurrent_bypass(rate_limiter):
    """
    With a 100 TPM limit and 5 concurrent requests each estimated at ~50+ tokens,
    upfront reservation must reject the late arrivals — not let all 5 through.
    Exercises the in-memory fallback in ``atomic_check_and_increment_by_n``.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-test-key")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        tpm_limit=100,
    )

    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message for concurrent bypass testing.",
            }
        ],
        "max_tokens": 50,
    }

    async def make_request(request_id: int) -> Dict[str, Any]:
        data = request_data.copy()
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="",
            )
            return {
                "request_id": request_id,
                "success": True,
                "reserved_tokens": get_request_stash().reserved_tokens,
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "status_code": getattr(e, "status_code", None),
            }

    tasks = [make_request(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    rate_limited = [r for r in failed if r.get("status_code") == 429]

    assert len(rate_limited) > 0, (
        f"Expected some rate-limited requests but all {len(successful)} succeeded — "
        f"the concurrent bypass bug is still present."
    )


@pytest.mark.asyncio
async def test_no_leak_on_over_limit_rejection(rate_limiter):
    """
    When a reservation would exceed the TPM limit, the counter must NOT be
    bumped. Otherwise rejected requests would silently consume quota with no
    path to refund (the failure callback only fires after the reservation
    was successfully stashed).
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-no-leak"),
        tpm_limit=10,  # tiny limit, easy to blow past
    )
    tokens_counter = handler.rate_limit_counter(
        key="api_key", value=user_api_key_dict.api_key, rate_limit_type="tokens"
    )

    # Reservation will estimate >> 10 tokens, so this should be rejected.
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "x" * 200}],
        "max_tokens": 200,
    }

    estimated = estimate_tokens_for_request(handler, data=data)
    assert estimated > user_api_key_dict.tpm_limit, (
        "Test assumes the reservation amount blows past the limit; "
        f"estimated={estimated}, limit={user_api_key_dict.tpm_limit}"
    )

    with pytest.raises(Exception, match="Limit type: tokens\\. Current limit") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    # The reservation bump (estimated_tokens) must NOT have committed. The
    # counter may carry a tiny pre-existing bump from should_rate_limit's
    # per-request +1 sliding-window logic, but it must be far below the
    # reservation amount — proving the all-or-nothing primitive rolled back
    # cleanly on rejection.
    cached_value = handler.rate_limits.local((tokens_counter,))[0]
    cached_int = int(cached_value or 0)
    assert cached_int < estimated, (
        f"Reservation leaked: counter={cached_int} after rejection of an estimated_tokens={estimated} reservation."
    )


@pytest.mark.asyncio
async def test_should_rate_limit_does_not_inflate_tokens_counter(rate_limiter):
    """
    The pre-call sliding-window check (`should_rate_limit`) must not bump the
    `:tokens` counter. That counter is owned exclusively by the atomic
    `reserve_tpm_tokens` path; double-handling shrinks the effective TPM
    budget by 1 per concurrent in-flight request.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-no-tokens-inflation")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=100,
        tpm_limit=10_000,
    )

    tokens_counter = handler.rate_limit_counter(key="api_key", value=api_key, rate_limit_type="tokens")

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }

    estimated = estimate_tokens_for_request(handler, data=data)

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    cached = int(handler.rate_limits.local((tokens_counter,))[0] or 0)

    # The :tokens counter should reflect ONLY the reservation amount — not
    # an additional +1 from the should_rate_limit pre-pass.
    assert cached == estimated, (
        f"Expected :tokens counter to equal the reservation ({estimated}) "
        f"with no +1 inflation from should_rate_limit, got {cached}"
    )


@pytest.mark.asyncio
async def test_concurrent_burst_within_tpm_budget_all_succeed(rate_limiter):
    """
    With a TPM limit comfortably above (N concurrent × per-request reservation),
    all N requests must succeed. Pre-fix the should_rate_limit +1-per-key
    inflation could 429 late arrivals on tight budgets.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-burst-budget"),
        tpm_limit=1000,
        rpm_limit=100,
    )

    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "x" * 40}],  # ~10 input tokens
        "max_tokens": 100,
    }

    estimated_per_request = estimate_tokens_for_request(handler, data=request_data)
    n_concurrent = 3
    # Sanity: total reservation must fit within tpm_limit and we want enough
    # headroom that any +1 inflation would NOT push us over.
    assert estimated_per_request * n_concurrent < user_api_key_dict.tpm_limit

    async def make_request(request_id: int):
        data = request_data.copy()
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="",
            )
            return True
        except Exception:
            return False

    results = await asyncio.gather(*[make_request(i) for i in range(n_concurrent)])

    assert all(results), (
        f"All {n_concurrent} requests should fit within tpm_limit="
        f"{user_api_key_dict.tpm_limit} (estimated_per_request="
        f"{estimated_per_request}), but only {sum(results)} succeeded — "
        f"the should_rate_limit :tokens-counter inflation bug is back."
    )


@pytest.mark.asyncio
async def test_estimate_tokens_uses_max_tokens_when_explicit(rate_limiter):
    """When max_tokens is set explicitly, reservation should equal input + max_tokens."""
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={
            "messages": [{"role": "user", "content": "abcd" * 4}],  # 16 chars ~ 4 tokens
            "max_tokens": 25,
        },
    )
    # input ~= 16/4 = 4 tokens; max_tokens = 25; total ~= 29
    assert estimate == 4 + 25


@pytest.mark.asyncio
async def test_estimate_tokens_honors_explicit_zero_max_tokens(rate_limiter):
    """
    Regression for a Greptile finding: explicit_max_tokens was resolved via
    `data.get("max_tokens") or data.get("max_completion_tokens") or
    data.get("max_output_tokens")`, so an explicit 0 in the first field was
    falsy and fell through to the next field (or the no-max_tokens floor),
    silently discarding a caller's explicit zero-output request.
    """
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={
            "messages": [{"role": "user", "content": "abcd" * 4}],  # 16 chars ~ 4 tokens
            "max_tokens": 0,
        },
    )
    assert estimate == 4, f"expected input-only reservation (4) for an explicit max_tokens=0, got {estimate}"


@pytest.mark.asyncio
async def test_estimate_tokens_honors_explicit_zero_max_output_tokens_for_responses(
    rate_limiter,
):
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={
            "input": "describe this image in detail",  # 29 chars ~ 7 tokens
            "max_output_tokens": 0,
        },
        min_configured_tpm_limit=40,
        call_type="aresponses",
    )
    assert estimate == 23


@pytest.mark.asyncio
async def test_estimate_tokens_zero_for_empty_embeddings(rate_limiter):
    """Embeddings have no output budget — reservation should equal input only."""
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={"input": "hello world"},  # 11 chars
    )
    # input ~= 11/4 = 2 tokens (max(1, 11//4)); max_tokens = 0
    assert estimate == 2


@pytest.mark.asyncio
async def test_contentless_request_reserves_minimum(rate_limiter):
    """
    A contentless request (no messages/prompt/input — e.g. /responses,
    tool-call continuations) must still hit the atomic counter so concurrent
    contentless requests don't all observe "under limit". Pre-fix the
    `has_estimable_content` short-circuit skipped the reservation entirely
    and post-call reconciliation provided no backpressure.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-contentless")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, tpm_limit=2)

    tokens_counter = handler.rate_limit_counter(key="api_key", value=api_key, rate_limit_type="tokens")

    # Two contentless requests should consume two slots of the 2-token
    # budget. The third must 429.
    for _ in range(2):
        data = {"model": "gpt-3.5-turbo"}
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
        assert get_request_stash().reserved_tokens == 1, "Contentless request should reserve the floor of 1 token"

    counter_after_two = int(handler.rate_limits.local((tokens_counter,))[0] or 0)
    assert counter_after_two == 2, (
        f"After two contentless requests at the floor, the api_key tokens counter should be 2, got {counter_after_two}"
    )

    with pytest.raises(Exception, match="Limit type: tokens\\. Current limit") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Third contentless request must be rate-limited; pre-fix it would have bypassed the TPM check entirely."
    )


@pytest.mark.asyncio
async def test_reservation_released_on_proxy_rejection(rate_limiter):
    """
    If the request is rejected after the pre-call reservation succeeds but
    before the LLM call (e.g. a downstream guardrail/auth hook raises),
    `async_post_call_failure_hook` must release the reservation. Otherwise
    the tokens leak — `async_log_failure_event` is a litellm completion
    callback and never fires for proxy-side rejections.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-leak-fix")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, tpm_limit=1000)

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )
    reserved = get_request_stash().reserved_tokens
    assert reserved > 0

    tokens_counter = handler.rate_limit_counter(key="api_key", value=api_key, rate_limit_type="tokens")
    counter_after_reserve = int(handler.rate_limits.local((tokens_counter,))[0] or 0)
    assert counter_after_reserve == reserved

    # Simulate a downstream guardrail rejecting the request.
    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    counter_after_release = int(handler.rate_limits.local((tokens_counter,))[0] or 0)
    assert counter_after_release == 0, (
        f"Reservation leaked: counter={counter_after_release} after proxy-level rejection refund (expected 0)."
    )
    assert get_request_stash().reservation_released is True, (
        "Released flag must be set to prevent async_log_failure_event from double-refunding."
    )


@pytest.mark.asyncio
async def test_token_rate_limit_headers_present_in_stored_response(rate_limiter):
    """
    With `skip_tpm_check=True` on the RPM sliding-window pass, token statuses
    only come from `reserve_tpm_tokens`. They must be merged into the stashed
    rate-limit response so the post-call hook can emit
    `x-ratelimit-{key}-remaining-tokens` / `-limit-tokens` headers to the
    client.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-headers")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=100,
        tpm_limit=10_000,
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    response = get_request_stash().rate_limit_response
    assert isinstance(response, dict), "Expected the stashed rate-limit response to be set after pre-call"

    statuses = response.get("statuses") or []
    token_statuses = [s for s in statuses if s.get("rate_limit_type") == "tokens"]
    request_statuses = [s for s in statuses if s.get("rate_limit_type") == "requests"]

    assert token_statuses, (
        f"Token rate-limit status missing from stored response. Without it, "
        f"x-ratelimit-*-tokens headers never reach the client. Got "
        f"statuses: {[(s.get('descriptor_key'), s.get('rate_limit_type')) for s in statuses]}"
    )
    assert request_statuses, (
        "RPM rate-limit status was clobbered by the TPM merge — both must coexist in the stored response."
    )

    # The token status carries the limit and a positive remaining budget.
    api_key_tokens = next(
        (s for s in token_statuses if s.get("descriptor_key") == "api_key"),
        None,
    )
    assert api_key_tokens is not None, f"api_key token status absent: {token_statuses}"
    assert api_key_tokens["current_limit"] == 10_000
    assert api_key_tokens["limit_remaining"] >= 0


@pytest.mark.asyncio
async def test_estimate_tokens_floor_caps_at_smallest_configured_tpm(rate_limiter):
    """
    Regression: with a small configured TPM cap and no max_tokens, the
    output-budget floor must be capped at a fraction of that limit so the
    reservation alone can't trip the limit.
    """
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={"messages": [{"role": "user", "content": "hello"}]},
        min_configured_tpm_limit=1000,
    )
    # input ~= 5//4 = 1 token; output floor capped at 1000//4 = 250;
    # total ~= 251 (well under 1000).
    assert estimate <= 1000 // 2, f"With TPM=1000, reservation must stay well under the limit; got {estimate}"
    assert estimate >= 1, "Estimate must be at least the call-site floor of 1"


@pytest.mark.asyncio
async def test_estimate_tokens_floor_unchanged_for_large_tpm(rate_limiter):
    """
    Large TPM budgets must keep the 1024-token floor so a stream of small
    concurrent requests can't collectively bypass the limit.
    """
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={"messages": [{"role": "user", "content": "hello"}]},
        min_configured_tpm_limit=100_000,
    )
    # input ~= 1; output floor = min(1024, 100_000//4=25_000) = 1024;
    # total ~= 1025.
    assert estimate == 1 + 1024


@pytest.mark.asyncio
async def test_estimate_tokens_floor_unchanged_when_kwarg_omitted(rate_limiter):
    """
    Callers that don't pass min_configured_tpm_limit (legacy path, tests that
    stub the estimator) must observe the pre-fix floor.
    """
    handler, _cache = rate_limiter

    estimate = estimate_tokens_for_request(
        handler,
        data={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert estimate == 1 + 1024


@pytest.mark.asyncio
async def test_small_tpm_cap_admits_no_max_tokens_request(rate_limiter):
    """
    Regression (end-to-end at the hook level): a project-level model_tpm_limit
    of 1000 with a tiny no-max_tokens request must not 429 on the first call.
    Pre-fix the 1024-token floor tripped OVER_LIMIT against the 1000-token cap
    on every request.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-small-tpm")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-small-tpm",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000},
            "model_rpm_limit": {"gpt-3.5-turbo": 60},
        },
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
    }

    # Must not raise — pre-fix this was a 429.
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    reserved = get_request_stash().reserved_tokens
    assert reserved > 0, "Reservation should have been stashed"
    assert reserved <= 1000 // 2, f"Capped floor must keep the reservation well under the 1000 TPM cap; got {reserved}"


@pytest.mark.asyncio
async def test_small_tpm_cap_injects_matching_max_tokens(rate_limiter):
    """
    When a small TPM cap forces the no-max_tokens floor below the baseline,
    the hook must also write data['max_tokens'] = capped_floor so the actual
    model output is bounded by the reservation. Without this cap, concurrent
    no-max_tokens generations can spend past the TPM limit before post-call
    reconciliation runs.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-small-tpm-cap"),
        project_id="proj-small-tpm-cap",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000},
        },
    )

    data: dict = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    assert data.get("max_tokens") == 1000 // 4, (
        f"Capped floor must be written to max_tokens to bound the actual model output; got {data.get('max_tokens')}"
    )


@pytest.mark.asyncio
async def test_large_tpm_cap_does_not_inject_max_tokens(rate_limiter):
    """
    A TPM cap that doesn't constrain the floor must not silently inject
    max_tokens — that would change behaviour for tenants who already have
    plenty of budget.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-large-tpm-cap"),
        project_id="proj-large-tpm-cap",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 100_000},
        },
    )

    data: dict = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    assert "max_tokens" not in data, f"Large TPM caps should leave max_tokens alone; got {data.get('max_tokens')}"


@pytest.mark.asyncio
async def test_small_tpm_cap_preserves_explicit_max_tokens(rate_limiter):
    """
    Explicit max_tokens from the caller must never be overwritten by the
    bypass mitigation — the user already declared their budget.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-explicit-max-tokens"),
        project_id="proj-explicit-max-tokens",
        project_metadata={
            "model_tpm_limit": {"gpt-3.5-turbo": 1000},
        },
    )

    data: dict = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 500,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    assert data["max_tokens"] == 500


@pytest.mark.asyncio
async def test_project_otpm_reservation_prevents_concurrent_bypass(rate_limiter):
    """
    Bedrock Mantle-style OTPM: with a 100 OTPM limit and 5 concurrent
    requests each reserving 50+ output tokens, upfront reservation must
    reject the late arrivals -- not let all 5 through. Exercises the
    in-memory fallback in ``atomic_check_and_increment_by_n`` for the
    project-scoped ITPM/OTPM descriptors specifically.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-bypass"),
        project_id="proj-mantle-bypass",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 100},
        },
    )

    request_data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    async def make_request(request_id: int) -> Dict[str, Any]:
        data = request_data.copy()
        try:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="",
            )
            return {"request_id": request_id, "success": True}
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "status_code": getattr(e, "status_code", None),
            }

    results = await asyncio.gather(*[make_request(i) for i in range(5)])

    successful = [r for r in results if r["success"]]
    rate_limited = [r for r in results if not r["success"] and r.get("status_code") == 429]

    assert len(rate_limited) > 0, f"Expected some OTPM-rate-limited requests but all {len(successful)} succeeded."


@pytest.mark.asyncio
async def test_project_otpm_rejects_multiple_completion_candidates(rate_limiter):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-multiple-candidates"),
        project_id="proj-multiple-candidates",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 500},
        },
    )
    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
        "n": 10,
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="acompletion",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_project_otpm_reserves_largest_conflicting_output_cap(rate_limiter):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-conflicting-caps"),
        project_id="proj-conflicting-caps",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 50},
        },
    )
    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
        "max_completion_tokens": 100,
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="acompletion",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
@pytest.mark.parametrize("config_field", ["config", "generationConfig"])
async def test_project_otpm_rejects_google_genai_native_output_cap(
    rate_limiter,
    call_type,
    config_field,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-3-flash-preview"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-otpm"),
        project_id="project-google-genai-native-otpm",
        project_metadata={"model_otpm_limit": {model: 50}},
    )

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": model,
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                config_field: {"maxOutputTokens": 100},
            },
            call_type=call_type,
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
@pytest.mark.parametrize("candidate_count_field", ["candidateCount", "candidate_count"])
async def test_project_otpm_rejects_google_genai_native_candidate_count(
    rate_limiter,
    call_type,
    candidate_count_field,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-3-flash-preview"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-candidate-count"),
        project_id="project-google-genai-native-candidate-count",
        project_metadata={"model_otpm_limit": {model: 150}},
    )

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": model,
                "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                "config": {
                    "maxOutputTokens": 50,
                    candidate_count_field: 4,
                },
            },
            call_type=call_type,
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
@pytest.mark.parametrize("config_field", [None, "config", "generationConfig"])
async def test_project_otpm_injects_google_genai_native_output_cap(
    rate_limiter,
    call_type,
    config_field,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-3-flash-preview"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-implicit-otpm"),
        project_id="project-google-genai-native-implicit-otpm",
        project_metadata={"model_otpm_limit": {model: 40}},
    )
    data = {
        "model": model,
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    }
    if config_field is not None:
        data[config_field] = {"temperature": 0}

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type=call_type,
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.otpm_reserved_tokens == 10
    expected_config_field = config_field or "config"
    assert data[expected_config_field]["maxOutputTokens"] == 10
    assert "max_tokens" not in data


@pytest.mark.asyncio
async def test_project_otpm_over_limit_rolls_back_itpm_reservation(rate_limiter):
    """
    When ITPM reserves fine but OTPM is then over limit, the ITPM
    reservation this same pre-call already made must be rolled back --
    otherwise it leaks until the window's TTL, silently shrinking the ITPM
    budget for every other request in that minute.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-rollback"),
        project_id="proj-mantle-rollback",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 1000000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 10},
        },
    )

    itpm_counter = handler.rate_limit_counter(
        key="model_per_project_itpm",
        value="proj-mantle-rollback:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 500,  # blows past the 10-token OTPM limit
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    cached_value = handler.rate_limits.local((itpm_counter,))[0]
    assert int(cached_value or 0) == 0, f"ITPM reservation leaked after OTPM rejection: counter={cached_value}"


@pytest.mark.asyncio
async def test_project_reconciliation_does_not_decrement_later_window():
    current_time = datetime(2026, 8, 5, 12, 0, 0)
    cache = DualCache()
    handler = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache),
        time_provider=lambda: current_time,
    )
    handler.window_size = 60
    scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    descriptor = {
        "key": scope[0],
        "value": scope[1],
        "rate_limit": {"tokens_per_unit": 1000, "window_size": 60},
    }

    reservation = await handler.atomic_check_and_increment_by_n(
        descriptors=[descriptor],
        increments=[{"tokens": 100}],
    )
    counter = handler.rate_limit_counter(*scope, rate_limit_type="tokens")
    window_identity = next(identity for identity in reservation["reservation_windows"] if identity[0] == counter)
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({scope})
    stash.itpm_reserved_window_identities = frozenset({window_identity})

    current_time += timedelta(seconds=61)
    later_reservation = await handler.atomic_check_and_increment_by_n(
        descriptors=[descriptor],
        increments=[{"tokens": 20}],
    )
    assert window_identity not in later_reservation["reservation_windows"]

    await handler.async_log_success_event(
        kwargs={},
        response_obj=ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)),
        start_time=current_time,
        end_time=current_time,
    )

    assert float(handler.rate_limits.local((counter,))[0] or 0) == 20


@pytest.mark.asyncio
async def test_project_reconciliation_decrements_its_active_window(rate_limiter):
    handler, cache = rate_limiter
    scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    descriptor = {
        "key": scope[0],
        "value": scope[1],
        "rate_limit": {"tokens_per_unit": 1000, "window_size": 60},
    }
    reservation = await handler.atomic_check_and_increment_by_n(
        descriptors=[descriptor],
        increments=[{"tokens": 100}],
    )
    counter = handler.rate_limit_counter(*scope, rate_limit_type="tokens")
    window_identity = next(identity for identity in reservation["reservation_windows"] if identity[0] == counter)
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({scope})
    stash.itpm_reserved_window_identities = frozenset({window_identity})

    await handler.async_log_success_event(
        kwargs={},
        response_obj=ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert float(handler.rate_limits.local((counter,))[0] or 0) == 10


@pytest.mark.asyncio
async def test_proxy_rejection_refunds_itpm_otpm_by_their_own_amount_not_combined(
    rate_limiter,
):
    """
    Regression for a Greptile-flagged bug: when a project configures both a
    combined model_tpm_limit and split model_itpm_limit/model_otpm_limit for
    the same model, async_post_call_failure_hook's proxy-side refund path
    used to decrement every token descriptor -- including the ITPM/OTPM
    ones -- by the flat combined reservation amount, instead of each
    bucket's own reserved amount. That drives the split counters negative
    (or under-refunds them) instead of returning them to exactly zero.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-mixed-tpm-io")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-mixed",
        project_metadata={
            "model_tpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 100000},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello there, this is a test message"}],
        "max_tokens": 60,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    tpm_counter = handler.rate_limit_counter(
        key="model_per_project",
        value="proj-mixed:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    itpm_counter = handler.rate_limit_counter(
        key="model_per_project_itpm",
        value="proj-mixed:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    otpm_counter = handler.rate_limit_counter(
        key="model_per_project_otpm",
        value="proj-mixed:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    tpm_reserved = int(handler.rate_limits.local((tpm_counter,))[0] or 0)
    itpm_reserved = int(handler.rate_limits.local((itpm_counter,))[0] or 0)
    otpm_reserved = int(handler.rate_limits.local((otpm_counter,))[0] or 0)
    assert tpm_reserved > 0 and itpm_reserved > 0 and otpm_reserved > 0

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    tpm_after = int(handler.rate_limits.local((tpm_counter,))[0] or 0)
    itpm_after = int(handler.rate_limits.local((itpm_counter,))[0] or 0)
    otpm_after = int(handler.rate_limits.local((otpm_counter,))[0] or 0)

    assert tpm_after == 0, f"combined TPM counter leaked: {tpm_after}"
    assert itpm_after == 0, f"ITPM counter corrupted by combined-amount refund: {itpm_after}"
    assert otpm_after == 0, f"OTPM counter corrupted by combined-amount refund: {otpm_after}"


@pytest.mark.asyncio
async def test_proxy_rejection_refunds_itpm_otpm_only_reservation_with_no_combined_tpm(
    rate_limiter,
):
    """
    Regression for the second half of the same bug: with only
    model_itpm_limit/model_otpm_limit configured (no model_tpm_limit), the
    combined reserved_tokens is 0, and the proxy-side refund path used to
    return immediately on that -- leaking the ITPM/OTPM reservations until
    the rate-limit window's TTL expired.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-io-only")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-io-only",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 100000},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello there, this is a test message"}],
        "max_tokens": 60,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    itpm_counter = handler.rate_limit_counter(
        key="model_per_project_itpm",
        value="proj-io-only:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    otpm_counter = handler.rate_limit_counter(
        key="model_per_project_otpm",
        value="proj-io-only:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )
    assert int(handler.rate_limits.local((itpm_counter,))[0] or 0) > 0
    assert int(handler.rate_limits.local((otpm_counter,))[0] or 0) > 0

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    itpm_after = int(handler.rate_limits.local((itpm_counter,))[0] or 0)
    otpm_after = int(handler.rate_limits.local((otpm_counter,))[0] or 0)
    assert itpm_after == 0, f"ITPM-only reservation leaked on proxy rejection: {itpm_after}"
    assert otpm_after == 0, f"OTPM-only reservation leaked on proxy rejection: {otpm_after}"


@pytest.mark.asyncio
async def test_otpm_rejection_does_not_double_refund_combined_tpm(rate_limiter):
    """
    Regression for a High-severity review finding: when the project ITPM
    reservation succeeds but OTPM is then over limit,
    _reserve_project_io_tokens_or_raise rolls back the combined-TPM
    reservation that already succeeded earlier in the same pre-call, then
    raises. If it doesn't also mark that reservation released,
    async_post_call_failure_hook -- which fires next in the real request
    lifecycle, since raising from async_pre_call_hook triggers it -- sees
    the same still-stashed reservation and refunds it a second time,
    driving the combined TPM counter negative and letting a caller push
    past the project's real TPM budget by repeatedly triggering OTPM
    rejections.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-double-refund")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-double-refund",
        project_metadata={
            "model_tpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 100000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 5},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello there, this is a test message"}],
        "max_tokens": 60,  # blows past the 5-token OTPM limit
    }

    tpm_counter = handler.rate_limit_counter(
        key="model_per_project",
        value="proj-double-refund:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    tpm_after_pre_call = int(handler.rate_limits.local((tpm_counter,))[0] or 0)
    assert tpm_after_pre_call == 0, f"combined TPM reservation not rolled back: {tpm_after_pre_call}"

    # In the real request lifecycle, async_post_call_failure_hook fires next
    # for a pre-call rejection. It must not refund the same reservation again.
    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=exc_info.value,
        user_api_key_dict=user_api_key_dict,
    )

    tpm_after_failure_hook = int(handler.rate_limits.local((tpm_counter,))[0] or 0)
    assert tpm_after_failure_hook == 0, (
        f"combined TPM counter went negative from a double refund: {tpm_after_failure_hook}"
    )


@pytest.mark.parametrize(
    "embedding_input",
    [
        list(range(51)),
        [list(range(25)), list(range(26))],
    ],
)
@pytest.mark.asyncio
async def test_project_itpm_rejects_pretokenized_embedding_input(
    rate_limiter,
    embedding_input,
):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-pretokenized-embedding-itpm"),
        project_id="proj-pretokenized-embedding",
        project_metadata={
            "model_itpm_limit": {"text-embedding-3-small": 50},
        },
    )
    data = {
        "model": "text-embedding-3-small",
        "input": embedding_input,
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_itpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="aembedding",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_responses_api_not_misclassified_as_embedding_for_output_estimate(
    rate_limiter,
):
    """
    Regression for a High-severity review finding: the Responses API also
    puts its prompt in data["input"], the same field embeddings use, so the
    output-token estimate treated every Responses call as an embedding and
    reserved zero output tokens. call_type now disambiguates the two: the
    same input-only payload gets zero output tokens for an embedding call
    but a real floor for a Responses API call.
    """
    handler, _cache = rate_limiter

    data = {"input": "describe this image in detail"}

    _, embedding_output_estimate = estimate_input_and_output_tokens(handler, data=data, call_type="aembedding")
    assert embedding_output_estimate == 0

    _, responses_output_estimate = estimate_input_and_output_tokens(handler, data=data, call_type="aresponses")
    assert responses_output_estimate > 0, (
        "Responses API call was misclassified as an embedding and reserved zero output tokens"
    )


@pytest.mark.parametrize(
    ("data", "call_type", "expected_output_tokens"),
    [
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "n": 10,
            },
            "acompletion",
            1000,
        ),
        (
            {
                "prompt": "hello",
                "max_tokens": 100,
                "n": 2,
                "best_of": 5,
            },
            "text_completion",
            500,
        ),
        (
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "n": 0,
                "best_of": "invalid",
            },
            "acompletion",
            100,
        ),
    ],
)
def test_output_estimate_accounts_for_completion_candidates(
    rate_limiter,
    data,
    call_type,
    expected_output_tokens,
):
    handler, _cache = rate_limiter

    _, estimated_output_tokens = estimate_input_and_output_tokens(
        handler,
        data=data,
        call_type=call_type,
    )

    assert estimated_output_tokens == expected_output_tokens


@pytest.mark.asyncio
async def test_itpm_reservation_accounts_for_audio_content_not_just_text(rate_limiter):
    """
    Regression for the audio half of a Medium-severity review finding:
    litellm.token_counter has no per-type handling for `input_audio`
    content blocks (unlike images, which it does count via
    use_default_image_token_count), so it silently contributes 0 tokens for
    them. Without DEFAULT_AUDIO_TOKEN_ESTIMATE, a burst of audio-heavy
    requests with minimal text would each reserve only the one-token floor
    and blow past the project ITPM limit before post-call reconciliation.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-audio-itpm")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-audio",
        project_metadata={
            # Tighter than DEFAULT_AUDIO_TOKEN_ESTIMATE (300), but far bigger
            # than the handful of tokens the bare text "hi" would cost.
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 50},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "base64-audio-bytes", "format": "wav"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_itpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Expected the audio content to push the ITPM reservation over the "
        "50-token limit; if this doesn't raise, audio content isn't being "
        "counted again."
    )


@pytest.mark.asyncio
async def test_itpm_rejects_large_audio_payload_that_would_pass_flat_estimate(
    rate_limiter,
):
    """
    Regression: a caller placing a long audio clip in one block previously
    reserved only 300 tokens (the flat estimate). With the size-proportional
    estimate, the same clip now reserves proportionally more and must trip
    the ITPM limit when the limit is tuned to exactly expose the difference.

    1 100 000 b64 chars → decoded ≈ 825 000 bytes → 825 000 // 1600 ≈ 515
    tokens > the 400-token limit. The flat estimate (300) would have passed.
    """
    handler, cache = rate_limiter

    large_b64 = "A" * 1_100_000

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-large-audio"),
        project_id="proj-large-audio",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 400},
        },
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": large_b64, "format": "wav"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_itpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Large audio payload must exceed the 400-token ITPM limit under the "
        "size-proportional estimate; the old flat-rate estimate (300 tokens) "
        "would have passed this limit silently"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "request_data"),
    [
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/high-resolution.png",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "summarize this"},
                            {
                                "type": "file",
                                "file": {
                                    "filename": "document.pdf",
                                    "file_data": "data:application/pdf;base64,dGVzdA==",
                                },
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "aresponses",
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe this"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/high-resolution.png",
                                "detail": "high",
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "aresponses",
            {"input": "continue", "previous_response_id": "resp-123"},
        ),
    ],
)
async def test_multimodal_requests_reserve_measured_project_itpm_not_full_limit(
    rate_limiter,
    call_type,
    request_data,
):
    """
    Regression: image, file, and previous_response_id requests used to
    reserve the project's whole ITPM limit up front. Because the atomic
    check is ``current + increment > limit``, that made every such request
    429 as soon as the window carried any usage at all and, while in
    flight, blocked every other request for the same project + model. They
    now reserve the token_counter estimate like everything else, so two
    multimodal requests fit in the same window.
    """
    handler, cache = rate_limiter
    model = "bedrock_mantle/claude-opus"
    project_itpm_limit = 10_000
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-multimodal-measured"),
        project_id="project-multimodal-measured",
        project_metadata={"model_itpm_limit": {model: project_itpm_limit}},
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data={"model": model, **request_data},
        call_type=call_type,
    )
    first_stash = get_request_stash()
    assert first_stash is not None
    first_reservation = first_stash.itpm_reserved_tokens
    assert 0 < first_reservation < project_itpm_limit // 2

    _request_stash.set(None)
    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data={"model": model, **request_data},
        call_type=call_type,
    )
    second_stash = get_request_stash()
    assert second_stash is not None
    assert second_stash is not first_stash
    assert second_stash.itpm_reserved_tokens == first_reservation


@pytest.mark.asyncio
async def test_responses_api_otpm_output_cap_applied_not_skipped_as_embedding(
    rate_limiter,
):
    """
    Regression for a Greptile P1 finding: _reserve_project_io_tokens_or_raise
    classified any request with data["input"] set as an embedding (no output
    tokens), which also misclassifies the Responses API -- it puts its prompt
    in "input" too, but does generate output. That skipped the output cap
    applied whenever the configured OTPM limit is small enough to need it,
    letting an unbounded Responses generation blow past OTPM before
    post-call reconciliation catches up.

    The cap must land on data["max_output_tokens"], not data["max_tokens"]:
    the Responses-to-chat-completion transformation only reads
    max_output_tokens, so a max_tokens cap is silently dropped before
    provider dispatch (a second Greptile finding on the same code path).
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-responses-otpm-cap")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-responses-otpm",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 40},
        },
    )

    data: Dict[str, Any] = {
        "model": "bedrock_mantle/claude-opus",
        "input": "describe this image in detail",
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="aresponses",
    )

    assert data.get("max_output_tokens") is not None, (
        "Responses call was misclassified as an embedding and skipped the OTPM output cap"
    )
    assert data["max_output_tokens"] == 16
    assert data.get("max_tokens") is None, (
        "OTPM output cap was written to max_tokens, which the Responses transformation ignores"
    )


@pytest.mark.asyncio
async def test_explicit_zero_output_responses_call_reserves_effective_provider_minimum(
    rate_limiter,
):
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-responses-zero-output"),
        project_id="proj-responses-zero-output",
        project_metadata={
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 5},
        },
    )

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": "bedrock_mantle/claude-opus",
                "input": "describe this image in detail",
                "max_output_tokens": 0,
            },
            call_type="aresponses",
        )

    assert getattr(exc_info.value, "status_code", None) == 429


@pytest.mark.asyncio
async def test_responses_api_combined_tpm_output_cap_applied_not_skipped_as_embedding(
    rate_limiter,
):
    """
    Regression for the same misclassification bug in the combined-TPM
    output-cap block of async_pre_call_hook (a second, independent
    `is_embedding = data.get("input") is not None` check). A project with
    only a combined model_tpm_limit (no split itpm/otpm) configured small
    enough to need the output cap must still apply it to a Responses call,
    and must write it to max_output_tokens for the same reason as the OTPM
    case above.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-responses-tpm-cap")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-responses-tpm",
        project_metadata={
            "model_tpm_limit": {"bedrock_mantle/claude-opus": 40},
        },
    )

    data: Dict[str, Any] = {
        "model": "bedrock_mantle/claude-opus",
        "input": "describe this image in detail",
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="aresponses",
    )

    assert data.get("max_output_tokens") is not None, (
        "Responses call was misclassified as an embedding and skipped the combined-TPM output cap"
    )
    assert data["max_output_tokens"] == 16
    assert data.get("max_tokens") is None, (
        "combined-TPM output cap was written to max_tokens, which the Responses transformation ignores"
    )


@pytest.mark.asyncio
async def test_reserve_io_tokens_noop_when_no_itpm_otpm_descriptors(rate_limiter):
    """reserve_io_tokens returns OK immediately when no ITPM/OTPM descriptors present."""
    handler, _cache = rate_limiter

    non_io_descriptor = {
        "key": "api_key",
        "value": "sk-test",
        "rate_limit": {"tokens_per_unit": 1000, "window_size": 60},
    }
    response, itpm_reserved, otpm_reserved = await handler.reserve_io_tokens(
        descriptors=[non_io_descriptor],
        estimated_input_tokens=50,
        estimated_output_tokens=50,
    )

    assert response["overall_code"] == "OK"
    assert itpm_reserved == 0
    assert otpm_reserved == 0


@pytest.mark.asyncio
async def test_reserve_io_tokens_itpm_only_no_otpm(rate_limiter):
    """When only ITPM descriptors are present (no OTPM), returns itpm_reserved with otpm=0."""
    handler, cache = rate_limiter

    itpm_descriptor = {
        "key": PROJECT_ITPM_DESCRIPTOR_KEY,
        "value": "proj-a:model",
        "rate_limit": {"tokens_per_unit": 10000, "window_size": 60},
    }
    response, itpm_reserved, otpm_reserved = await handler.reserve_io_tokens(
        descriptors=[itpm_descriptor],
        estimated_input_tokens=100,
        estimated_output_tokens=50,
    )

    assert response["overall_code"] == "OK"
    assert itpm_reserved == 100
    assert otpm_reserved == 0


@pytest.mark.asyncio
async def test_post_call_failure_skips_rpm_only_descriptor_in_tpm_refund(rate_limiter):
    """
    async_post_call_failure_hook skips descriptors without tokens_per_unit
    (e.g. an RPM-only api_key scope) when building the combined-TPM refund ops,
    so a key with rpm_limit but no tpm_limit doesn't receive a spurious refund
    that would drive its counter negative. Covers the continue guard at line 4250.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-rpm-only-desc")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        rpm_limit=100,
        project_id="proj-rpm-only-desc",
        project_metadata={"model_tpm_limit": {"gpt-3.5-turbo": 100000}},
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    rpm_tokens_counter = handler.rate_limit_counter(key="api_key", value=api_key, rate_limit_type="tokens")

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    api_key_tokens_after = int(handler.rate_limits.local((rpm_tokens_counter,))[0] or 0)
    assert api_key_tokens_after >= 0, (
        f"RPM-only api_key scope must not receive a negative TPM refund; got {api_key_tokens_after}"
    )


@pytest.mark.asyncio
async def test_max_output_tokens_prevents_cap_injection(rate_limiter):
    """
    Regression for veria-ai comment: when a Responses API request supplies
    max_output_tokens (the canonical Responses output bound) but not max_tokens
    or max_completion_tokens, the has_explicit_max_tokens check was False, so
    the code injected data["max_tokens"] = capped_floor and silently truncated
    the response.

    With the fix, max_output_tokens is included in the explicit-cap check and
    data["max_tokens"] must NOT be injected when max_output_tokens is already
    set.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-max-output-tokens"),
        project_id="proj-responses-max-output",
        project_metadata={
            "model_otpm_limit": {"mock-model": 100},
        },
    )

    data: dict = {
        "model": "mock-model",
        "input": "Summarise the document",
        "max_output_tokens": 80,
        "litellm_call_id": "test-max-output-tokens",
        "metadata": {},
    }

    try:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="responses",
        )
    except Exception:
        pass

    assert "max_tokens" not in data, (
        "data['max_tokens'] must not be injected when max_output_tokens is already "
        "set; the cap injection was overriding the caller's explicit output bound"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "request_data", "cap_field", "reserved_tokens"),
    [
        ("aresponses", {"input": "hello", "max_tokens": 1}, "max_output_tokens", 16),
        (
            "acompletion",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_output_tokens": 1,
            },
            "max_tokens",
            10,
        ),
    ],
)
async def test_output_reservation_ignores_cap_fields_from_other_endpoints(
    rate_limiter,
    call_type,
    request_data,
    cap_field,
    reserved_tokens,
):
    handler, cache = rate_limiter
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token(f"sk-{call_type}"),
        project_id=f"project-{call_type}",
        project_metadata={"model_otpm_limit": {"model": 40}},
    )
    data = {"model": "model", **request_data}

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type=call_type,
    )

    assert data[cap_field] == reserved_tokens
    stash = get_request_stash()
    assert stash is not None
    assert stash.otpm_reserved_tokens == reserved_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    ["agenerate_content", "agenerate_content_stream"],
)
async def test_google_genai_native_contents_reserve_project_itpm(
    rate_limiter,
    call_type,
):
    handler, cache = rate_limiter
    model = "gemini/gemini-2.5-flash"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-google-genai-native-itpm"),
        project_id="project-google-genai-native-itpm",
        project_metadata={"model_itpm_limit": {model: 10_000}},
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data={
            "model": model,
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Gemini quota input " * 200}],
                }
            ],
        },
        call_type=call_type,
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens > 100


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["rerank", "arerank"])
async def test_rerank_query_and_documents_enforce_project_itpm(
    rate_limiter,
    monkeypatch,
    call_type,
):
    handler, cache = rate_limiter
    captured = {}

    def token_counter(**kwargs):
        captured.update(kwargs)
        return 101

    monkeypatch.setattr("litellm.token_counter", token_counter)
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token(f"sk-{call_type}-itpm"),
        project_id=f"project-{call_type}-itpm",
        project_metadata={"model_itpm_limit": {"rerank-model": 100}},
    )

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_itpm") as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={
                "model": "rerank-model",
                "query": "Which document is most relevant?",
                "documents": ["first document", {"text": "second document"}],
            },
            call_type=call_type,
        )

    assert getattr(exc_info.value, "status_code", None) == 429
    assert captured["text"] == ("Which document is most relevant?\nfirst document\n{'text': 'second document'}")


@pytest.mark.asyncio
async def test_project_io_counters_not_double_charged_when_reservation_disabled(
    monkeypatch,
):
    """With LITELLM_TPM_TOKEN_RESERVATION_ENABLED=false the first
    should_rate_limit pass used to +1 every ITPM/OTPM counter on top of the
    full reservation _reserve_project_io_tokens_or_raise always makes,
    permanently inflating each bucket by one token per request."""
    monkeypatch.setenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false")
    cache = DualCache()
    handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
    assert handler.tpm_reservation_enabled is False

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-io-no-reservation"),
        project_id="proj-io-no-reservation",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 1000000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 1000000},
        },
    )
    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    stash = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens > 0
    assert stash.otpm_reserved_tokens > 0

    for descriptor_key, reserved in (
        ("model_per_project_itpm", stash.itpm_reserved_tokens),
        ("model_per_project_otpm", stash.otpm_reserved_tokens),
    ):
        tokens_counter = handler.rate_limit_counter(
            key=descriptor_key,
            value="proj-io-no-reservation:bedrock_mantle/claude-opus",
            rate_limit_type="tokens",
        )
        cached = handler.rate_limits.local((tokens_counter,))[0]
        assert int(cached or 0) == reserved, (
            f"{descriptor_key} counter {cached} != reserved {reserved}: "
            "first-pass should_rate_limit double-charged the bucket"
        )


@pytest.mark.parametrize(
    ("call_type", "data"),
    [
        (
            "text_completion",
            {
                "messages": [{"role": "user", "content": "ignored"}],
                "prompt": "abcd",
                "input": "ignored",
                "max_tokens": 1,
            },
        ),
        (None, {"prompt": "abcd", "max_tokens": 1}),
        (None, {"prompt": ["abcd", "efgh"], "max_tokens": 1}),
    ],
)
def test_split_token_estimate_selects_endpoint_input(rate_limiter, call_type, data):
    handler, _cache = rate_limiter

    estimated_input, estimated_output = estimate_input_and_output_tokens(
        handler,
        data=data,
        call_type=call_type,
    )

    assert estimated_input > 0
    assert estimated_output == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


_COMPACTION_SUMMARY: Final = "The user planned a Tokyo trip."
_NEXT_TURN: Final = {"type": "message", "role": "user", "content": "And Osaka?"}
_COMPACTED_INPUT: Final = [
    {"type": "message", "role": "user", "content": "Plan a Tokyo trip. " * 200},
    {"type": "message", "role": "assistant", "content": "Day one, Asakusa. " * 200},
    {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps({"type": "compaction", "content": _COMPACTION_SUMMARY}),
    },
    _NEXT_TURN,
]
_SUMMARY_INPUT: Final = [{"type": "message", "role": "user", "content": _COMPACTION_SUMMARY}, _NEXT_TURN]


def test_reservation_counts_a_compaction_item_by_its_summary(rate_limiter):
    handler, _cache = rate_limiter
    padded_item: Final = {
        **_COMPACTED_INPUT[2],
        "encrypted_content": json.dumps(
            {"type": "compaction", "content": _COMPACTION_SUMMARY, "opaque_state": "x" * 4000}
        ),
    }

    assert estimate_input_and_output_tokens(
        handler,
        data={"model": "gpt-4o", "input": [padded_item, _NEXT_TURN], "max_output_tokens": 1},
        call_type="aresponses",
    ) == estimate_input_and_output_tokens(
        handler, data={"model": "gpt-4o", "input": _COMPACTED_INPUT[2:], "max_output_tokens": 1}, call_type="aresponses"
    )


_QUOTA_TEXT: Final = "a" * 400
_QUOTA_JSON_TEXT: Final = '{"type":"input_image","image_url":"data:image/png;base64,AAAA"}'


@pytest.mark.parametrize(
    ("call_type", "request_data", "expected"),
    [
        ("acompletion", {"messages": [{"role": "user", "content": _QUOTA_TEXT}], "max_tokens": 16}, (100, 16)),
        ("acompletion", {"messages": [{"role": "user", "content": _QUOTA_JSON_TEXT}], "max_tokens": 16}, (15, 16)),
        (
            "anthropic_messages",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [{"type": "text", "text": _QUOTA_TEXT}],
                            }
                        ],
                    }
                ],
                "max_tokens": 16,
            },
            (0, 16),
        ),
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "t", "type": "function", "function": {"name": "f", "arguments": _QUOTA_TEXT}}
                        ],
                    }
                ],
                "max_tokens": 16,
            },
            (0, 16),
        ),
        ("aresponses", {"input": _QUOTA_TEXT, "max_output_tokens": 16}, (100, 16)),
        ("aresponses", {"input": _QUOTA_TEXT, "instructions": "b" * 400, "max_output_tokens": 16}, (100, 16)),
        (
            "aresponses",
            {"input": [{"role": "user", "content": _QUOTA_TEXT}], "instructions": "b" * 400, "max_output_tokens": 16},
            (107, 16),
        ),
        (
            "aresponses",
            {
                "input": [
                    {
                        "type": "file_search_call",
                        "id": "f",
                        "status": "completed",
                        "queries": ["q"],
                        "results": [
                            {
                                "file_id": "file_1",
                                "filename": "result.txt",
                                "attributes": {},
                                "score": 1,
                                "text": _QUOTA_TEXT,
                            }
                        ],
                    }
                ],
                "max_output_tokens": 16,
            },
            (146, 16),
        ),
        (
            "aresponses",
            {
                "input": [{"type": "custom_tool_call_output", "call_id": "t", "output": _QUOTA_JSON_TEXT}],
                "max_output_tokens": 16,
            },
            (32, 16),
        ),
        ("aresponses", {"input": "", "instructions": "b" * 400}, (0, 0)),
        ("aembedding", {"input": [_QUOTA_TEXT]}, (100, 0)),
        ("atext_completion", {"prompt": [_QUOTA_TEXT], "max_tokens": 16}, (100, 16)),
    ],
)
def test_media_correction_preserves_exact_non_media_reservations(
    rate_limiter: tuple[RateLimitHandler, DualCache],
    call_type: str,
    request_data: dict[str, object],
    expected: tuple[int, int],
) -> None:
    handler, _ = rate_limiter
    assert estimate_input_and_output_tokens(handler, request_data, call_type=call_type) == expected


def _quota_media_payload(kind: str, compressed: bool) -> str:
    output: Final = io.BytesIO()
    if kind == "image":
        Image.new("RGB", (1024, 1024)).save(output, format="PNG", compress_level=9 if compressed else 0)
    else:
        sample_rate: Final = 8000 if compressed else 48000
        with sf.SoundFile(output, "w", samplerate=sample_rate, channels=1, format="WAV") as audio:
            audio.buffer_write(b"\0" * (sample_rate * 2), dtype="int16")
    return base64.b64encode(output.getvalue()).decode()


def _quota_media_request(kind: str, payload: str, call_type: str) -> dict[str, object]:
    block: Final = (
        {"type": "input_audio", "input_audio": {"data": payload, "format": "wav"}}
        if kind == "audio"
        else {"type": "input_image", "image_url": "data:image/png;base64," + payload}
        if call_type == "aresponses"
        else {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": payload}}
        if call_type == "anthropic_messages"
        else {"type": "image_url", "image_url": {"url": "data:image/png;base64," + payload}}
    )
    content: Final = (
        [{"type": "tool_result", "tool_use_id": "t", "content": [block]}]
        if call_type == "anthropic_messages"
        else [block]
    )
    field: Final = "input" if call_type == "aresponses" else "messages"
    return {
        "model": "gpt-6-astra",
        field: [{"role": "user", "content": content}],
        "max_tokens": 16,
        "max_output_tokens": 16,
    }


@pytest.mark.parametrize(
    ("kind", "call_type"),
    [
        ("image", "acompletion"),
        ("image", "aresponses"),
        ("image", "anthropic_messages"),
        ("audio", "acompletion"),
        ("audio", "aresponses"),
    ],
)
def test_media_reservation_is_independent_of_encoding_and_preserves_request(
    rate_limiter: tuple[RateLimitHandler, DualCache], kind: str, call_type: str
) -> None:
    handler, _ = rate_limiter
    compressed: Final = _quota_media_request(kind, _quota_media_payload(kind, True), call_type)
    uncompressed: Final = _quota_media_request(kind, _quota_media_payload(kind, False), call_type)
    original: Final = copy.deepcopy(uncompressed)
    estimate: Final = estimate_input_and_output_tokens(handler, uncompressed, call_type=call_type)
    assert estimate == estimate_input_and_output_tokens(handler, compressed, call_type=call_type)
    assert estimate[0] > 0 and estimate[1] == 16
    assert uncompressed == original


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["acompletion", "aresponses"])
@pytest.mark.parametrize("limit", [1000, 20000])
async def test_media_pre_call_rejects_tight_quota_and_reserves_sufficient_quota(
    rate_limiter: tuple[RateLimitHandler, DualCache], call_type: str, limit: int
) -> None:
    handler, cache = rate_limiter
    request: Final = {
        key: value
        for key, value in _quota_media_request("image", _quota_media_payload("image", True), call_type).items()
        if key not in ("max_tokens", "max_output_tokens")
    }
    auth: Final = UserAPIKeyAuth(api_key=hash_token("sk-media-quota"), tpm_limit=limit)
    if limit == 1000:
        with pytest.raises(HTTPException) as error:
            await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=request, call_type=call_type)
        assert error.value.status_code == 429
        return
    await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=request, call_type=call_type)
    stash: Final = get_request_stash()
    assert stash is not None
    expected: Final = estimate_input_and_output_tokens(
        handler, request, call_type=call_type, min_configured_tpm_limit=limit
    )
    assert stash.reserved_tokens == sum(expected)
    assert expected[0] > 1000 and expected[1] >= expected[0]


_EMBEDDING_MODEL: Final = "gemini-embedding-2"
_EMBEDDING_CAPTION: Final = "a red bicycle leaning on a wall"


def _embedding_media_uri(kind: str, compressed: bool) -> str:
    mime: Final = "image/png" if kind == "image" else "audio/wav"
    return f"data:{mime};base64,{_quota_media_payload(kind, compressed)}"


@pytest.mark.parametrize("kind", ["image", "audio"])
@pytest.mark.parametrize(
    "shape",
    ["string", "list", "file_block", "with_caption", "combined"],
)
def test_embedding_media_input_is_estimated_independently_of_encoding(
    rate_limiter: tuple[RateLimitHandler, DualCache], kind: str, shape: str
) -> None:
    handler, _ = rate_limiter

    def embedding_request(uri: str) -> dict[str, object]:
        shapes: Final[dict[str, object]] = {
            "string": uri,
            "list": [uri],
            "file_block": [{"type": "file", "file": {"file_data": uri}}],
            "with_caption": [_EMBEDDING_CAPTION, uri],
            "combined": [[_EMBEDDING_CAPTION, uri]],
        }
        return {"model": _EMBEDDING_MODEL, "input": shapes[shape]}

    uncompressed: Final = embedding_request(_embedding_media_uri(kind, False))
    original: Final = copy.deepcopy(uncompressed)
    estimate: Final = estimate_input_and_output_tokens(handler, uncompressed, call_type="aembedding")
    assert estimate == estimate_input_and_output_tokens(
        handler, embedding_request(_embedding_media_uri(kind, True)), call_type="aembedding"
    )
    assert 0 < estimate[0] < 2_000 and estimate[1] == 0
    assert uncompressed == original


@pytest.mark.parametrize("kind", ["image", "audio"])
def test_embedding_media_input_costs_the_same_as_chat_media(
    rate_limiter: tuple[RateLimitHandler, DualCache], kind: str
) -> None:
    handler, _ = rate_limiter
    uri: Final = _embedding_media_uri(kind, False)
    chat_block: Final = (
        {"type": "image_url", "image_url": {"url": uri}}
        if kind == "image"
        else {"type": "input_audio", "input_audio": {"data": _quota_media_payload(kind, False), "format": "wav"}}
    )
    chat_input_tokens, _ = estimate_input_and_output_tokens(
        handler,
        {"model": _EMBEDDING_MODEL, "messages": [{"role": "user", "content": [chat_block]}]},
        call_type="acompletion",
    )
    caption_tokens, _ = estimate_input_and_output_tokens(
        handler, {"model": _EMBEDDING_MODEL, "input": [_EMBEDDING_CAPTION]}, call_type="aembedding"
    )

    assert estimate_input_and_output_tokens(
        handler, {"model": _EMBEDDING_MODEL, "input": uri}, call_type="aembedding"
    ) == (chat_input_tokens, 0)
    assert estimate_input_and_output_tokens(
        handler, {"model": _EMBEDDING_MODEL, "input": [_EMBEDDING_CAPTION, uri]}, call_type="aembedding"
    ) == (caption_tokens + chat_input_tokens, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_field", ["model_tpm_limit", "model_itpm_limit"])
async def test_project_token_limit_admits_image_embedding_and_still_rejects_long_text(
    rate_limiter: tuple[RateLimitHandler, DualCache], limit_field: str
) -> None:
    handler, cache = rate_limiter
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token(f"sk-embedding-image-{limit_field}"),
        project_id=f"proj-embedding-image-{limit_field}",
        project_metadata={limit_field: {_EMBEDDING_MODEL: 75_000}},
    )
    image_uri: Final = _embedding_media_uri("image", False)
    long_text: Final = "word " * 80_000

    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"model": _EMBEDDING_MODEL, "input": image_uri},
        call_type="aembedding",
    )
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=auth,
            cache=cache,
            data={"model": _EMBEDDING_MODEL, "input": long_text},
            call_type="aembedding",
        )
    assert error.value.status_code == 429
    assert len(image_uri) > len(long_text)


_EMBEDDING_FILE_MIMES: Final = MappingProxyType({"pdf": "application/pdf", "video": "video/mp4"})


def _embedding_file_uri(kind: str) -> str:
    return f"data:{_EMBEDDING_FILE_MIMES[kind]};base64,{base64.b64encode(bytes(range(256)) * 1200).decode()}"


def _embedding_uri(kind: str) -> str:
    return _embedding_file_uri(kind) if kind in _EMBEDDING_FILE_MIMES else _embedding_media_uri(kind, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_field", ["model_tpm_limit", "model_itpm_limit"])
@pytest.mark.parametrize("kind", ["pdf", "video"])
@pytest.mark.parametrize("shape", ["data_uri", "file_block"])
async def test_project_token_limit_admits_document_embedding_up_to_the_model_input_limit(
    rate_limiter: tuple[RateLimitHandler, DualCache], limit_field: str, kind: str, shape: str
) -> None:
    handler, cache = rate_limiter
    uri: Final = _embedding_file_uri(kind)
    model_input_limit: Final = get_model_info(_EMBEDDING_MODEL)["max_input_tokens"]
    assert isinstance(model_input_limit, int) and len(uri) // 4 > 75_000 > model_input_limit

    def project_auth(limit: int) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key=hash_token(f"sk-embedding-{kind}-{shape}-{limit_field}-{limit}"),
            project_id=f"proj-embedding-{kind}-{shape}-{limit_field}-{limit}",
            project_metadata={limit_field: {_EMBEDDING_MODEL: limit}},
        )

    item: Final = uri if shape == "data_uri" else {"type": "file", "file": {"file_data": uri}}
    request: Final = {"model": _EMBEDDING_MODEL, "input": [_EMBEDDING_CAPTION, item]}
    await handler.async_pre_call_hook(
        user_api_key_dict=project_auth(75_000), cache=cache, data=dict(request), call_type="aembedding"
    )
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=project_auth(model_input_limit // 2),
            cache=cache,
            data=dict(request),
            call_type="aembedding",
        )
    assert error.value.status_code == 429


_TEXT_ONLY_EMBEDDING_MODEL: Final = "text-embedding-3-small"


@pytest.mark.parametrize("kind", ["image", "audio", "pdf", "video"])
def test_text_only_embedding_model_counts_data_uri_as_text(
    rate_limiter: tuple[RateLimitHandler, DualCache], kind: str
) -> None:
    handler, _ = rate_limiter
    uri: Final = _embedding_uri(kind)

    assert estimate_input_and_output_tokens(
        handler, {"model": _TEXT_ONLY_EMBEDDING_MODEL, "input": [_EMBEDDING_CAPTION, uri]}, call_type="aembedding"
    ) == estimate_input_and_output_tokens(
        handler,
        {"model": _TEXT_ONLY_EMBEDDING_MODEL, "input": [_EMBEDDING_CAPTION, "x" * len(uri)]},
        call_type="aembedding",
    )


def _project_limited_embedding_call(
    limit_field: str, backends: tuple[str, ...], kind: str = "image"
) -> tuple[RateLimitHandler, DualCache, UserAPIKeyAuth, dict[str, object]]:
    cache: Final = DualCache()
    handler: Final = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache), media_model_resolver=lambda _model, _team_id: backends
    )
    scope: Final = f"{limit_field}-{kind}-{'-'.join(backends)}"
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token(f"sk-embedding-backends-{scope}"),
        project_id=f"proj-embedding-backends-{scope}",
        project_metadata={limit_field: {"team-embedder": 75_000}},
    )
    return handler, cache, auth, {"model": "team-embedder", "input": _embedding_uri(kind)}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_field", ["model_tpm_limit", "model_itpm_limit"])
@pytest.mark.parametrize(
    ("backends", "kind"),
    [
        ((_TEXT_ONLY_EMBEDDING_MODEL,), "image"),
        ((_EMBEDDING_MODEL, _TEXT_ONLY_EMBEDDING_MODEL), "image"),
        (("voyage/voyage-multimodal-3.5",), "image"),
        (("oci/cohere.embed-v4.0",), "image"),
        (("bedrock/us.twelvelabs.marengo-embed-2-7-v1:0",), "image"),
        (("bedrock/twelvelabs.marengo-embed-3-0-v1:0",), "image"),
        (("bedrock/amazon.nova-2-multimodal-embeddings-v1:0",), "pdf"),
    ],
    ids=["text", "mixed", "voyage", "oci", "marengo_2_7", "marengo_3", "nova_pdf"],
)
async def test_project_token_limit_rejects_embedding_media_a_backend_reads_as_text(
    limit_field: str, backends: tuple[str, ...], kind: str
) -> None:
    handler, cache, auth, request = _project_limited_embedding_call(limit_field, backends, kind)

    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=request, call_type="aembedding")
    assert error.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_field", ["model_tpm_limit", "model_itpm_limit"])
@pytest.mark.parametrize(
    "backend",
    [
        _EMBEDDING_MODEL,
        "bedrock/amazon.titan-embed-image-v1",
        "bedrock/async_invoke/amazon.nova-2-multimodal-embeddings-v1:0",
    ],
    ids=["gemini", "titan", "nova_async_invoke"],
)
async def test_project_token_limit_admits_image_embedding_through_an_alias_of_an_image_backend(
    limit_field: str, backend: str
) -> None:
    handler, cache, auth, request = _project_limited_embedding_call(limit_field, (backend,))

    await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=request, call_type="aembedding")
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=auth,
            cache=cache,
            data={**request, "input": "word " * 100_000},
            call_type="aembedding",
        )
    assert error.value.status_code == 429


def _malformed_embedding_uri(shape: str) -> str:
    payload: Final = _quota_media_payload("image", False)
    whole_quanta: Final = payload.rstrip("=")[: len(payload.rstrip("=")) // 4 * 4]
    words: Final = "word " * 100_000
    return {
        "no_comma": f"data:audio/wav;base64{words}",
        "text_payload": f"data:image/png;base64,{words}",
        "no_base64_marker": f"data:image/png,{payload}",
        "header_past_256_chars": f"data:image/png;{'x' * 300};base64,{payload}",
        "noncanonical_padding": f"data:image/png;base64,{whole_quanta}AB==",
        "padding_mid_payload": f"data:image/png;base64,{whole_quanta}AA==AAAA",
        "length_not_multiple_of_4": f"data:image/png;base64,{whole_quanta}A",
    }[shape]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        "no_comma",
        "text_payload",
        "no_base64_marker",
        "header_past_256_chars",
        "noncanonical_padding",
        "padding_mid_payload",
        "length_not_multiple_of_4",
    ],
)
async def test_project_token_limit_counts_a_malformed_data_uri_as_text(
    rate_limiter: tuple[RateLimitHandler, DualCache], shape: str
) -> None:
    handler, cache = rate_limiter
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token(f"sk-embedding-malformed-{shape}"),
        project_id=f"proj-embedding-malformed-{shape}",
        project_metadata={"model_tpm_limit": {_EMBEDDING_MODEL: 75_000}},
    )

    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=auth,
            cache=cache,
            data={"model": _EMBEDDING_MODEL, "input": _malformed_embedding_uri(shape)},
            call_type="aembedding",
        )
    assert error.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["cohere/embed-v4.0", "cohere.embed-v4:0"], ids=["cohere", "bedrock_cohere"])
async def test_project_token_limit_counts_a_mixed_batch_as_text_when_the_backend_types_the_whole_batch(
    backend: str,
) -> None:
    handler, cache, auth, request = _project_limited_embedding_call("model_tpm_limit", (backend,))
    image_uri: Final = request["input"]

    await handler.async_pre_call_hook(
        user_api_key_dict=auth, cache=cache, data={**request, "input": [image_uri, image_uri]}, call_type="aembedding"
    )
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=auth,
            cache=cache,
            data={**request, "input": [image_uri, _EMBEDDING_CAPTION]},
            call_type="aembedding",
        )
    assert error.value.status_code == 429


@pytest.mark.asyncio
async def test_project_itpm_limit_reserves_media_tokens_for_an_image_embedding(
    rate_limiter: tuple[RateLimitHandler, DualCache],
) -> None:
    handler, cache = rate_limiter
    image_uri: Final = _embedding_media_uri("image", False)
    image_tokens, _ = estimate_input_and_output_tokens(
        handler, {"model": _EMBEDDING_MODEL, "input": image_uri}, call_type="aembedding"
    )
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-embedding-image-itpm-media"),
        project_id="proj-embedding-image-itpm-media",
        project_metadata={"model_itpm_limit": {_EMBEDDING_MODEL: image_tokens // 2}},
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=cache,
        data={"model": _EMBEDDING_MODEL, "input": _EMBEDDING_CAPTION},
        call_type="aembedding",
    )
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=auth,
            cache=cache,
            data={"model": _EMBEDDING_MODEL, "input": image_uri},
            call_type="aembedding",
        )
    assert error.value.status_code == 429


@pytest.mark.asyncio
async def test_ptu_pre_call_weights_media_input_and_output() -> None:
    cache: Final = DualCache()
    ceiling: Final = PTUTeamCeiling(
        model_group="gpt-6-astra", tpm_limit=10000, output_to_input_ratio=4, cached_input_ratio=0
    )

    def resolve_ceiling(team: str, model: str) -> PTUTeamCeiling | None:
        return ceiling if (team, model) == ("test-team", "gpt-6-astra") else None

    handler: Final = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(cache), ptu_team_ceiling_resolver=resolve_ceiling
    )
    request: Final = _quota_media_request("image", _quota_media_payload("image", True), "acompletion")
    await handler.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-media-ptu"), team_id="test-team"),
        cache=cache,
        data=request,
        call_type="acompletion",
    )
    stash: Final = get_request_stash()
    assert stash is not None
    assert (
        stash.ptu_reserved_tokens
        == estimate_image_tokens("gpt-6-astra", "data:image/png;base64," + _quota_media_payload("image", True)) + 4 * 16
    )


@pytest.mark.asyncio
async def test_project_fallback_keeps_legacy_audio_estimate_and_output_policy(
    rate_limiter: tuple[RateLimitHandler, DualCache],
) -> None:
    handler, cache = rate_limiter
    unsupported: Final = {"type": "unsupported_content"}
    with pytest.raises(UnsupportedContentTypeError):
        token_counter(model="gpt-6-astra", messages=[{"role": "user", "content": [unsupported]}])
    request: Final = {
        "model": "gpt-6-astra",
        "messages": [
            {
                "role": "user",
                "content": [
                    unsupported,
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _quota_media_payload("audio", True), "format": "wav"},
                    },
                ],
            }
        ],
    }
    auth: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-media-project"),
        project_id="media-project",
        project_metadata={"model_itpm_limit": {"gpt-6-astra": 1000}, "model_otpm_limit": {"gpt-6-astra": 1000}},
    )
    await handler.async_pre_call_hook(user_api_key_dict=auth, cache=cache, data=request, call_type="acompletion")
    stash: Final = get_request_stash()
    assert stash is not None
    assert stash.itpm_reserved_tokens == DEFAULT_AUDIO_TOKEN_ESTIMATE
    assert stash.otpm_reserved_tokens == 1


@pytest.mark.asyncio
async def test_single_anthropic_image_exceeds_tight_quota_with_explicit_output_cap(
    rate_limiter: tuple[RateLimitHandler, DualCache],
) -> None:
    handler, cache = rate_limiter
    request: Final = {
        "model": "anthropic/claude-haiku-5-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _quota_media_payload("image", True),
                        },
                    }
                ],
            }
        ],
        "max_tokens": 64,
    }
    with pytest.raises(HTTPException) as error:
        await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-claude-media-quota"), tpm_limit=500),
            cache=cache,
            data=request,
            call_type="anthropic_messages",
        )
    assert error.value.status_code == 429
