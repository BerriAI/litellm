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
from datetime import datetime
from typing import Any, Dict

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    ITPM_RESERVED_SCOPES_KEY,
    ITPM_RESERVED_TOKENS_KEY,
    OTPM_RESERVED_SCOPES_KEY,
    OTPM_RESERVED_TOKENS_KEY,
    RATE_LIMIT_DESCRIPTORS_KEY,
    TPM_RESERVATION_RELEASED_KEY,
    TPM_RESERVED_MODEL_KEY,
    TPM_RESERVED_SCOPES_KEY,
    TPM_RESERVED_TOKENS_KEY,
    _PROXY_MaxParallelRequestsHandler_v3 as RateLimitHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


@pytest.fixture
def rate_limiter():
    cache = DualCache()
    handler = RateLimitHandler(internal_usage_cache=InternalUsageCache(cache))
    return handler, cache


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
                "reserved_tokens": data.get(TPM_RESERVED_TOKENS_KEY, 0),
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
    counter_key = handler.create_rate_limit_keys(
        key="api_key", value=user_api_key_dict.api_key, rate_limit_type="tokens"
    )

    # Reservation will estimate >> 10 tokens, so this should be rejected.
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "x" * 200}],
        "max_tokens": 200,
    }

    estimated = handler._estimate_tokens_for_request(data=data)
    assert estimated > user_api_key_dict.tpm_limit, (
        "Test assumes the reservation amount blows past the limit; "
        f"estimated={estimated}, limit={user_api_key_dict.tpm_limit}"
    )

    with pytest.raises(Exception) as exc_info:
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
    cached_value = await cache.async_get_cache(key=counter_key, local_only=True)
    cached_int = int(cached_value or 0)
    assert cached_int < estimated, (
        f"Reservation leaked: counter={cached_int} after rejection of an "
        f"estimated_tokens={estimated} reservation."
    )


@pytest.mark.asyncio
async def test_token_adjustment_on_success(rate_limiter):
    """
    On success a reserved scope's counter is reconciled to actual via
    `actual - reserved`. With actual=50 and reserved=100, the api_key
    counter should see a -50 delta — and only because api_key was
    reserved against. Unreserved scopes get the full +actual instead.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-test-adjust")

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_SCOPES_KEY: [["api_key", api_key]],
            }
        },
        "model": "gpt-3.5-turbo",
    }

    mock_response = ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append(
                {
                    "key": op["key"],
                    "increment": op["increment_value"],
                }
            )

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_adjustments = [i for i in increments if "tokens" in i["key"]]

    assert any(i["increment"] == -50 for i in token_adjustments), (
        f"Expected a -50 token adjustment (50 actual - 100 reserved) but got: "
        f"{token_adjustments}"
    )


@pytest.mark.asyncio
async def test_token_release_on_failure(rate_limiter):
    """On failure the entire reservation must be refunded — but only against
    scopes that were actually charged at pre-call. Unreserved scopes were
    never incremented and must not receive a -reserved op (would drift
    negative)."""
    handler, _cache = rate_limiter

    api_key = hash_token("sk-test-fail")

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_SCOPES_KEY: [["api_key", api_key]],
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append(
                {
                    "key": op["key"],
                    "increment": op["increment_value"],
                }
            )

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_releases = [i for i in increments if "tokens" in i["key"]]

    assert any(
        i["increment"] == -100 for i in token_releases
    ), f"Expected the full reservation (-100) to be released, got: {token_releases}"


@pytest.mark.asyncio
async def test_model_scope_refund_targets_reserved_model(rate_limiter):
    """
    The pre-call reservation is charged against ``data["model"]`` but the
    router later writes ``model_group`` into ``litellm_params.metadata``,
    which can be ``None`` or a different value. Reconciliation MUST refund the
    same model-scoped counter that was incremented; otherwise model-level
    counters (model_per_team / model_per_key / etc.) drift up forever.

    This test makes ``model_group`` absent from kwargs (the failure mode in
    the Greptile P1) and asserts the refund still targets the model the
    reservation used.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-test-model-mismatch")
    team_id = "team-abc"
    reserved_model = "gpt-4o-mini"

    mock_kwargs = {
        # NOTE: no litellm_params.metadata.model_group — get_model_group_from_litellm_kwargs
        # returns None on this kwargs dict.
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_MODEL_KEY: reserved_model,
                TPM_RESERVED_SCOPES_KEY: [
                    ["model_per_team", f"{team_id}:{reserved_model}"]
                ],
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_model_per_team_key = handler.create_rate_limit_keys(
        key="model_per_team",
        value=f"{team_id}:{reserved_model}",
        rate_limit_type="tokens",
    )
    matching = [i for i in increments if i["key"] == expected_model_per_team_key]
    assert matching, (
        f"Expected a refund on the reserved model_per_team counter "
        f"({expected_model_per_team_key}) but got: "
        f"{[i['key'] for i in increments]}"
    )
    assert matching[0]["increment"] == -100, (
        f"Expected full -100 refund on model_per_team counter, got "
        f"{matching[0]['increment']}"
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

    tokens_counter_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }

    estimated = handler._estimate_tokens_for_request(data=data)

    await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="",
    )

    cached = int(
        await cache.async_get_cache(key=tokens_counter_key, local_only=True) or 0
    )

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

    estimated_per_request = handler._estimate_tokens_for_request(data=request_data)
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
async def test_org_scope_refund_on_failure(rate_limiter):
    """
    The plain `organization` scope is reserved upfront (it carries
    tokens_per_unit) — so on failure, the full reservation must be released
    against {organization:org_id}:tokens. Pre-fix this scope was missing
    from `_build_tpm_scope_pipeline_operations`, leaking forever.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-org-refund")
    org_id = "org-acme"

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_org_id": org_id,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_SCOPES_KEY: [["organization", org_id]],
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_org_key = handler.create_rate_limit_keys(
        key="organization", value=org_id, rate_limit_type="tokens"
    )
    matching = [i for i in increments if i["key"] == expected_org_key]
    assert matching, (
        f"Expected a refund on the org tokens counter ({expected_org_key}) "
        f"but got keys: {[i['key'] for i in increments]}"
    )
    assert (
        matching[0]["increment"] == -100
    ), f"Expected full -100 refund on org counter, got {matching[0]['increment']}"


@pytest.mark.asyncio
async def test_org_scope_reconciled_on_success(rate_limiter):
    """
    On success the org tokens counter must be reconciled to actual usage.
    With reserved=100 and actual=50, the org scope should see a -50 delta.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-org-success")
    org_id = "org-acme"

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_org_id": org_id,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_SCOPES_KEY: [["organization", org_id]],
            }
        },
        "model": "gpt-3.5-turbo",
    }

    mock_response = ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_org_key = handler.create_rate_limit_keys(
        key="organization", value=org_id, rate_limit_type="tokens"
    )
    matching = [i for i in increments if i["key"] == expected_org_key]
    assert matching, (
        f"Expected a reconciliation op on the org tokens counter "
        f"({expected_org_key}), got keys: {[i['key'] for i in increments]}"
    )
    assert matching[0]["increment"] == -50, (
        f"Expected -50 delta on org counter (50 actual - 100 reserved), got "
        f"{matching[0]['increment']}"
    )


@pytest.mark.asyncio
async def test_estimate_tokens_uses_max_tokens_when_explicit(rate_limiter):
    """When max_tokens is set explicitly, reservation should equal input + max_tokens."""
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={
            "messages": [
                {"role": "user", "content": "abcd" * 4}
            ],  # 16 chars ~ 4 tokens
            "max_tokens": 25,
        }
    )
    # input ~= 16/4 = 4 tokens; max_tokens = 25; total ~= 29
    assert estimate == 4 + 25


@pytest.mark.asyncio
async def test_estimate_tokens_zero_for_empty_embeddings(rate_limiter):
    """Embeddings have no output budget — reservation should equal input only."""
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
        data={"input": "hello world"}  # 11 chars
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

    counter_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

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
        assert (data.get("metadata") or {}).get(
            TPM_RESERVED_TOKENS_KEY
        ) == 1, "Contentless request should reserve the floor of 1 token"

    counter_after_two = int(
        await cache.async_get_cache(key=counter_key, local_only=True) or 0
    )
    assert counter_after_two == 2, (
        f"After two contentless requests at the floor, the api_key tokens "
        f"counter should be 2, got {counter_after_two}"
    )

    with pytest.raises(Exception) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data={"model": "gpt-3.5-turbo"},
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Third contentless request must be rate-limited; pre-fix it would "
        "have bypassed the TPM check entirely."
    )


@pytest.mark.asyncio
async def test_atomic_keys_share_hash_tag_per_descriptor(rate_limiter):
    """
    Cluster safety: every key in a single descriptor's Lua payload must
    share a `{key:value}` hash tag so the call lands on a single Redis
    Cluster slot. Otherwise the Lua script raises CROSSSLOT in cluster mode.
    """
    handler, _cache = rate_limiter

    descriptors = [
        {
            "key": "api_key",
            "value": "abc",
            "rate_limit": {
                "requests_per_unit": 10,
                "tokens_per_unit": 100,
                "window_size": 60,
            },
        },
        {
            "key": "user",
            "value": "xyz",
            "rate_limit": {"tokens_per_unit": 200, "window_size": 60},
        },
    ]
    increments = [{"requests": 1, "tokens": 10}, {"tokens": 10}]

    for descriptor, inc in zip(descriptors, increments):
        keys, _args, _meta = handler._build_descriptor_atomic_payload(
            descriptor=descriptor,
            increment_amounts=inc,
        )
        # All keys in a descriptor's payload must share the same {tag}
        # — that's the prefix between the first '{' and '}'.
        tags = {k[: k.index("}") + 1] for k in keys}
        assert len(tags) == 1, (
            f"Descriptor {descriptor['key']}:{descriptor['value']} produced "
            f"keys spanning multiple hash tags: {tags}. Redis Cluster would "
            f"reject this Lua call with CROSSSLOT."
        )
        expected_tag = f"{{{descriptor['key']}:{descriptor['value']}}}"
        assert tags == {expected_tag}, f"Expected hash tag {expected_tag}, got {tags}"


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
    reserved = (data.get("metadata") or {})[TPM_RESERVED_TOKENS_KEY]
    assert reserved > 0

    counter_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )
    counter_after_reserve = int(
        await cache.async_get_cache(key=counter_key, local_only=True) or 0
    )
    assert counter_after_reserve == reserved

    # Simulate a downstream guardrail rejecting the request.
    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    counter_after_release = int(
        await cache.async_get_cache(key=counter_key, local_only=True) or 0
    )
    assert counter_after_release == 0, (
        f"Reservation leaked: counter={counter_after_release} after "
        f"proxy-level rejection refund (expected 0)."
    )
    assert (data.get("metadata") or {}).get(TPM_RESERVATION_RELEASED_KEY) is True, (
        "Released marker must be stamped to prevent "
        "async_log_failure_event from double-refunding."
    )


@pytest.mark.asyncio
async def test_reservation_release_idempotent(rate_limiter):
    """
    If both `async_post_call_failure_hook` and `async_log_failure_event` end
    up firing for the same request, only the first refund applies — the
    second sees the released marker and no-ops.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-idempotent")

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    # Shared metadata dict simulates the propagation between
    # request_data["metadata"] and kwargs["litellm_params"]["metadata"] —
    # the post-call-failure-hook stamps the released marker there, and the
    # log-failure-event reads it.
    shared_metadata = {
        "user_api_key_hash": api_key,
        TPM_RESERVED_TOKENS_KEY: 100,
        RATE_LIMIT_DESCRIPTORS_KEY: [
            {
                "key": "api_key",
                "value": api_key,
                "rate_limit": {"tokens_per_unit": 10000, "window_size": 60},
            }
        ],
    }

    request_data = {
        "metadata": shared_metadata,
    }

    await handler.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("rejected"),
        user_api_key_dict=UserAPIKeyAuth(api_key=api_key),
    )

    first_refund_count = len([i for i in increments if "tokens" in i["key"]])
    assert first_refund_count > 0, "First refund should have applied"

    # Now simulate async_log_failure_event firing afterwards. It must see
    # the released marker (via shared metadata) and not double-refund.
    await handler.async_log_failure_event(
        kwargs={
            "litellm_params": {"metadata": shared_metadata},
            "standard_logging_object": {"metadata": shared_metadata},
        },
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    second_refund_count = len([i for i in increments if "tokens" in i["key"]])
    assert second_refund_count == first_refund_count, (
        f"Idempotency violated: refund count went from {first_refund_count} "
        f"to {second_refund_count} after second hook fired."
    )


@pytest.mark.asyncio
async def test_unreserved_scopes_charged_actual_not_delta_on_success(rate_limiter):
    """
    Counter-drift fix: a scope present in metadata but NOT reserved at
    pre-call (no configured TPM limit for it) must be charged the full
    `actual_tokens` on success — never the `delta = actual - reserved`.
    Otherwise that scope's counter goes negative whenever `actual < reserved`
    (the common case, since the reservation includes a conservative output
    pad).
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-mixed-scopes")
    team_id = "team-no-tpm-limit"

    # Reservation ONLY hit api_key — team had no TPM limit configured.
    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_SCOPES_KEY: [["api_key", api_key]],
            }
        },
        "model": "gpt-3.5-turbo",
    }

    mock_response = ModelResponse(
        id="t",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    api_key_token_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )
    team_token_key = handler.create_rate_limit_keys(
        key="team", value=team_id, rate_limit_type="tokens"
    )

    api_key_ops = [i for i in increments if i["key"] == api_key_token_key]
    team_ops = [i for i in increments if i["key"] == team_token_key]

    assert api_key_ops and api_key_ops[0]["increment"] == -50, (
        f"Reserved api_key scope must reconcile via delta (50-100=-50), "
        f"got {api_key_ops}"
    )
    assert team_ops and team_ops[0]["increment"] == 50, (
        f"Unreserved team scope must be charged full actual (+50), not the "
        f"-50 delta (which would drift its counter negative). Got {team_ops}"
    )


@pytest.mark.asyncio
async def test_unreserved_scopes_not_refunded_on_failure(rate_limiter):
    """
    Failure refund must only emit ops against scopes the reservation
    actually charged. Refunding an unreserved scope (which was never
    incremented at pre-call) would drive its counter to -reserved.
    """
    handler, _cache = rate_limiter

    api_key = hash_token("sk-mixed-fail")
    team_id = "team-no-tpm"

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
                TPM_RESERVED_TOKENS_KEY: 100,
                TPM_RESERVED_SCOPES_KEY: [["api_key", api_key]],
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = (
        mock_increment
    )

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    team_token_key = handler.create_rate_limit_keys(
        key="team", value=team_id, rate_limit_type="tokens"
    )
    api_key_token_key = handler.create_rate_limit_keys(
        key="api_key", value=api_key, rate_limit_type="tokens"
    )

    team_ops = [i for i in increments if i["key"] == team_token_key]
    api_key_ops = [i for i in increments if i["key"] == api_key_token_key]

    assert not team_ops, (
        f"Unreserved team scope must NOT be refunded (would drift negative), "
        f"got {team_ops}"
    )
    assert (
        api_key_ops and api_key_ops[0]["increment"] == -100
    ), f"Reserved api_key scope must be refunded -100, got {api_key_ops}"


@pytest.mark.asyncio
async def test_token_rate_limit_headers_present_in_stored_response(rate_limiter):
    """
    With `skip_tpm_check=True` on the RPM sliding-window pass, token statuses
    only come from `reserve_tpm_tokens`. They must be merged into
    `data["litellm_proxy_rate_limit_response"]` so the post-call hook can
    emit `x-ratelimit-{key}-remaining-tokens` / `-limit-tokens` headers to
    the client.
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

    response = data.get("litellm_proxy_rate_limit_response")
    assert isinstance(
        response, dict
    ), "Expected litellm_proxy_rate_limit_response to be set after pre-call"

    statuses = response.get("statuses") or []
    token_statuses = [s for s in statuses if s.get("rate_limit_type") == "tokens"]
    request_statuses = [s for s in statuses if s.get("rate_limit_type") == "requests"]

    assert token_statuses, (
        f"Token rate-limit status missing from stored response. Without it, "
        f"x-ratelimit-*-tokens headers never reach the client. Got "
        f"statuses: {[(s.get('descriptor_key'), s.get('rate_limit_type')) for s in statuses]}"
    )
    assert request_statuses, (
        "RPM rate-limit status was clobbered by the TPM merge — both must "
        "coexist in the stored response."
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

    estimate = handler._estimate_tokens_for_request(
        data={"messages": [{"role": "user", "content": "hello"}]},
        min_configured_tpm_limit=1000,
    )
    # input ~= 5//4 = 1 token; output floor capped at 1000//4 = 250;
    # total ~= 251 (well under 1000).
    assert (
        estimate <= 1000 // 2
    ), f"With TPM=1000, reservation must stay well under the limit; got {estimate}"
    assert estimate >= 1, "Estimate must be at least the call-site floor of 1"


@pytest.mark.asyncio
async def test_estimate_tokens_floor_unchanged_for_large_tpm(rate_limiter):
    """
    Large TPM budgets must keep the 1024-token floor so a stream of small
    concurrent requests can't collectively bypass the limit.
    """
    handler, _cache = rate_limiter

    estimate = handler._estimate_tokens_for_request(
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

    estimate = handler._estimate_tokens_for_request(
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

    reserved = (data.get("metadata") or {}).get(TPM_RESERVED_TOKENS_KEY)
    assert reserved is not None, "Reservation should have been stashed"
    assert reserved <= 1000 // 2, (
        f"Capped floor must keep the reservation well under the 1000 TPM "
        f"cap; got {reserved}"
    )


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
        f"Capped floor must be written to max_tokens to bound the actual "
        f"model output; got {data.get('max_tokens')}"
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

    assert "max_tokens" not in data, (
        f"Large TPM caps should leave max_tokens alone; got "
        f"{data.get('max_tokens')}"
    )


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

    assert len(rate_limited) > 0, (
        f"Expected some OTPM-rate-limited requests but all {len(successful)} succeeded."
    )


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

    itpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_itpm",
        value="proj-mantle-rollback:bedrock_mantle/claude-opus",
        rate_limit_type="tokens",
    )

    data = {
        "model": "bedrock_mantle/claude-opus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 500,  # blows past the 10-token OTPM limit
    }

    with pytest.raises(Exception) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    cached_value = await cache.async_get_cache(key=itpm_counter_key, local_only=True)
    assert int(cached_value or 0) == 0, (
        f"ITPM reservation leaked after OTPM rejection: counter={cached_value}"
    )


@pytest.mark.asyncio
async def test_project_itpm_reconciled_on_success_excludes_cached_tokens(rate_limiter):
    """
    On success, ITPM reconciles to billable input tokens (prompt_tokens
    minus cached_tokens) -- not raw prompt_tokens. Cached prompt-read tokens
    are free under Bedrock Mantle and must not count against the ITPM quota,
    even though they still appear in usage/cost logging elsewhere.
    """
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-mantle:model")
    otpm_scope = ("model_per_project_otpm", "proj-mantle:model")

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                ITPM_RESERVED_TOKENS_KEY: 100,
                ITPM_RESERVED_SCOPES_KEY: [list(itpm_scope)],
                OTPM_RESERVED_TOKENS_KEY: 60,
                OTPM_RESERVED_SCOPES_KEY: [list(otpm_scope)],
            }
        },
    }

    mock_response = ModelResponse(
        id="test",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model="bedrock_mantle/claude-opus",
        usage=Usage(
            prompt_tokens=80,
            completion_tokens=40,
            total_tokens=120,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=30),
        ),
        choices=[],
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = mock_increment

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [i for i in increments if "model_per_project_itpm" in i["key"]]
    otpm_adjustments = [i for i in increments if "model_per_project_otpm" in i["key"]]

    # billable_input = 80 - 30 cached = 50; delta = 50 - 100 reserved = -50
    assert any(i["increment"] == -50 for i in itpm_adjustments), (
        f"Expected a -50 ITPM adjustment (50 billable - 100 reserved), got: {itpm_adjustments}"
    )
    # delta = 40 actual completion - 60 reserved = -20
    assert any(i["increment"] == -20 for i in otpm_adjustments), (
        f"Expected a -20 OTPM adjustment (40 actual - 60 reserved), got: {otpm_adjustments}"
    )


@pytest.mark.asyncio
async def test_project_itpm_otpm_released_on_failure(rate_limiter):
    """On failure, the full ITPM and OTPM reservations must be refunded."""
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-mantle:model")
    otpm_scope = ("model_per_project_otpm", "proj-mantle:model")

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                ITPM_RESERVED_TOKENS_KEY: 100,
                ITPM_RESERVED_SCOPES_KEY: [list(itpm_scope)],
                OTPM_RESERVED_TOKENS_KEY: 60,
                OTPM_RESERVED_SCOPES_KEY: [list(otpm_scope)],
            }
        },
    }

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = mock_increment

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_releases = [i for i in increments if "model_per_project_itpm" in i["key"]]
    otpm_releases = [i for i in increments if "model_per_project_otpm" in i["key"]]

    assert any(i["increment"] == -100 for i in itpm_releases), itpm_releases
    assert any(i["increment"] == -60 for i in otpm_releases), otpm_releases


@pytest.mark.asyncio
async def test_proxy_rejection_refunds_itpm_otpm_by_their_own_amount_not_combined(rate_limiter):
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

    tpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project", value="proj-mixed:bedrock_mantle/claude-opus", rate_limit_type="tokens"
    )
    itpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_itpm", value="proj-mixed:bedrock_mantle/claude-opus", rate_limit_type="tokens"
    )
    otpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_otpm", value="proj-mixed:bedrock_mantle/claude-opus", rate_limit_type="tokens"
    )

    tpm_reserved = int(await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0)
    itpm_reserved = int(await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0)
    otpm_reserved = int(await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0)
    assert tpm_reserved > 0 and itpm_reserved > 0 and otpm_reserved > 0

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    tpm_after = int(await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0)
    itpm_after = int(await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0)
    otpm_after = int(await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0)

    assert tpm_after == 0, f"combined TPM counter leaked: {tpm_after}"
    assert itpm_after == 0, f"ITPM counter corrupted by combined-amount refund: {itpm_after}"
    assert otpm_after == 0, f"OTPM counter corrupted by combined-amount refund: {otpm_after}"


@pytest.mark.asyncio
async def test_proxy_rejection_refunds_itpm_otpm_only_reservation_with_no_combined_tpm(rate_limiter):
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

    itpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_itpm", value="proj-io-only:bedrock_mantle/claude-opus", rate_limit_type="tokens"
    )
    otpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project_otpm", value="proj-io-only:bedrock_mantle/claude-opus", rate_limit_type="tokens"
    )
    assert int(await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0) > 0
    assert int(await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0) > 0

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("guardrail rejected"),
        user_api_key_dict=user_api_key_dict,
    )

    itpm_after = int(await cache.async_get_cache(key=itpm_counter_key, local_only=True) or 0)
    otpm_after = int(await cache.async_get_cache(key=otpm_counter_key, local_only=True) or 0)
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

    tpm_counter_key = handler.create_rate_limit_keys(
        key="model_per_project", value="proj-double-refund:bedrock_mantle/claude-opus", rate_limit_type="tokens"
    )

    with pytest.raises(Exception) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429

    tpm_after_pre_call = int(await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0)
    assert tpm_after_pre_call == 0, f"combined TPM reservation not rolled back: {tpm_after_pre_call}"

    # In the real request lifecycle, async_post_call_failure_hook fires next
    # for a pre-call rejection. It must not refund the same reservation again.
    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=exc_info.value,
        user_api_key_dict=user_api_key_dict,
    )

    tpm_after_failure_hook = int(await cache.async_get_cache(key=tpm_counter_key, local_only=True) or 0)
    assert tpm_after_failure_hook == 0, (
        f"combined TPM counter went negative from a double refund: {tpm_after_failure_hook}"
    )


@pytest.mark.asyncio
async def test_itpm_reservation_accounts_for_image_content_not_just_text(rate_limiter):
    """
    Regression for a Medium-severity review finding: the ITPM reservation
    used to estimate input tokens from message text length alone, so a
    request with a tiny text prompt but real image content would reserve
    almost nothing, letting a burst of such requests blow past the
    configured ITPM limit before post-call usage reconciliation catches up.
    _estimate_precise_input_tokens uses litellm.token_counter (with
    use_default_image_token_count=True, so this test makes no network call)
    to account for image content instead of only text length.
    """
    handler, cache = rate_limiter

    api_key = hash_token("sk-multimodal-itpm")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-multimodal",
        project_metadata={
            # Tighter than the ~250-token default image estimate, but far
            # bigger than the handful of tokens the bare text "hi" would
            # cost under the old char-count-only estimate.
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
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/some-image.png"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(Exception) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="",
        )
    assert getattr(exc_info.value, "status_code", None) == 429, (
        "Expected the image content to push the ITPM reservation over the "
        "50-token limit; if this doesn't raise, the estimate is undercounting "
        "non-text content again."
    )


@pytest.mark.asyncio
async def test_responses_api_not_misclassified_as_embedding_for_output_estimate(rate_limiter):
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

    _, embedding_output_estimate = handler._estimate_input_and_output_tokens(data=data, call_type="aembedding")
    assert embedding_output_estimate == 0

    _, responses_output_estimate = handler._estimate_input_and_output_tokens(data=data, call_type="aresponses")
    assert responses_output_estimate > 0, (
        "Responses API call was misclassified as an embedding and reserved zero output tokens"
    )


@pytest.mark.asyncio
async def test_responses_api_usage_reconciles_using_input_output_tokens_fields(rate_limiter):
    """
    Regression for the other half of the same finding: ResponseAPIUsage
    exposes input_tokens/output_tokens, not prompt_tokens/completion_tokens.
    Before this fix, _resolve_io_token_reconcile_usage couldn't resolve
    Responses API usage at all, so the reservation was silently kept as-is
    instead of being trued up to the much larger actual usage.
    """
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-responses:model")
    otpm_scope = ("model_per_project_otpm", "proj-responses:model")

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                ITPM_RESERVED_TOKENS_KEY: 10,
                ITPM_RESERVED_SCOPES_KEY: [list(itpm_scope)],
                OTPM_RESERVED_TOKENS_KEY: 10,
                OTPM_RESERVED_SCOPES_KEY: [list(otpm_scope)],
            }
        },
    }

    mock_response = ResponsesAPIResponse(
        id="resp_test",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(input_tokens=80, output_tokens=400, total_tokens=480),
    )

    increments = []

    async def mock_increment(increment_list, **kwargs):
        for op in increment_list:
            increments.append({"key": op["key"], "increment": op["increment_value"]})

    handler.internal_usage_cache.dual_cache.async_increment_cache_pipeline = mock_increment

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [i for i in increments if "model_per_project_itpm" in i["key"]]
    otpm_adjustments = [i for i in increments if "model_per_project_otpm" in i["key"]]

    # delta = 80 actual input - 10 reserved = +70
    assert any(i["increment"] == 70 for i in itpm_adjustments), (
        f"ITPM reservation was never trued up to actual Responses API usage: {itpm_adjustments}"
    )
    # delta = 400 actual output - 10 reserved = +390
    assert any(i["increment"] == 390 for i in otpm_adjustments), (
        f"OTPM reservation was never trued up to actual Responses API usage: {otpm_adjustments}"
    )


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

    with pytest.raises(Exception) as exc_info:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
