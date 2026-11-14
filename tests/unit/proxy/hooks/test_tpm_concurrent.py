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

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Final

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _AUDIO_BYTES_PER_TOKEN,
    PROJECT_ITPM_DESCRIPTOR_KEY,
    PROJECT_OTPM_DESCRIPTOR_KEY,
    RateLimitedModel,
    WindowGuardedIncrement,
    _call_id_from_callback_kwargs,
    _request_stash,
    get_or_create_request_stash,
    get_request_stash,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3 as RateLimitHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.redis.stores.rate_limits import AtomicCounter, CounterIncrement, RateLimitScope
from litellm.types.llms.openai import (
    InputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)
from litellm.types.rerank import RerankResponse
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits


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
async def test_token_adjustment_on_success(recording_rate_limiter):
    """
    On success a reserved scope's counter is reconciled to actual via
    `actual - reserved`. With actual=50 and reserved=100, the api_key
    counter should see a -50 delta — and only because api_key was
    reserved against. Unreserved scopes get the full +actual instead.
    """
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-test-adjust")

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
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

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_adjustments = [i for i in increments if i.counter.kind == "tokens"]

    assert any(i.amount == -50 for i in token_adjustments), (
        f"Expected a -50 token adjustment (50 actual - 100 reserved) but got: {token_adjustments}"
    )


@pytest.mark.asyncio
async def test_token_release_on_failure(recording_rate_limiter):
    """On failure the entire reservation must be refunded — but only against
    scopes that were actually charged at pre-call. Unreserved scopes were
    never incremented and must not receive a -reserved op (would drift
    negative)."""
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-test-fail")

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
            }
        },
    }

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    token_releases = [i for i in increments if i.counter.kind == "tokens"]

    assert any(i.amount == -100 for i in token_releases), (
        f"Expected the full reservation (-100) to be released, got: {token_releases}"
    )


@pytest.mark.asyncio
async def test_model_scope_refund_targets_reserved_model(recording_rate_limiter):
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
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-test-model-mismatch")
    team_id = "team-abc"
    reserved_model = "gpt-4o-mini"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_model = RateLimitedModel(requested=reserved_model, group=reserved_model)
    stash.reserved_scopes = frozenset({("model_per_team", f"{team_id}:{reserved_model}")})

    mock_kwargs = {
        # NOTE: no litellm_params.metadata.model_group — get_model_group_from_litellm_kwargs
        # returns None on this kwargs dict.
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
            }
        },
    }

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_model_per_team_counter = handler.rate_limit_counter(
        key="model_per_team",
        value=f"{team_id}:{reserved_model}",
        rate_limit_type="tokens",
    )
    matching = [i for i in increments if i.counter == expected_model_per_team_counter]
    assert matching, (
        f"Expected a refund on the reserved model_per_team counter "
        f"({expected_model_per_team_counter}) but got: "
        f"{[i.counter for i in increments]}"
    )
    assert matching[0].amount == -100, f"Expected full -100 refund on model_per_team counter, got {matching[0].amount}"


@pytest.mark.asyncio
async def test_org_scope_refund_on_failure(recording_rate_limiter):
    """
    The plain `organization` scope is reserved upfront (it carries
    tokens_per_unit) — so on failure, the full reservation must be released
    against {organization:org_id}:tokens. Pre-fix this scope was missing
    from `_build_tpm_scope_pipeline_operations`, leaking forever.
    """
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-org-refund")
    org_id = "org-acme"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("organization", org_id)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_org_id": org_id,
            }
        },
    }

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_org_counter = handler.rate_limit_counter(key="organization", value=org_id, rate_limit_type="tokens")
    matching = [i for i in increments if i.counter == expected_org_counter]
    assert matching, (
        f"Expected a refund on the org tokens counter ({expected_org_counter}) "
        f"but got keys: {[i.counter for i in increments]}"
    )
    assert matching[0].amount == -100, f"Expected full -100 refund on org counter, got {matching[0].amount}"


@pytest.mark.asyncio
async def test_org_scope_reconciled_on_success(recording_rate_limiter):
    """
    On success the org tokens counter must be reconciled to actual usage.
    With reserved=100 and actual=50, the org scope should see a -50 delta.
    """
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-org-success")
    org_id = "org-acme"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("organization", org_id)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_org_id": org_id,
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

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    expected_org_counter = handler.rate_limit_counter(key="organization", value=org_id, rate_limit_type="tokens")
    matching = [i for i in increments if i.counter == expected_org_counter]
    assert matching, (
        f"Expected a reconciliation op on the org tokens counter "
        f"({expected_org_counter}), got keys: {[i.counter for i in increments]}"
    )
    assert matching[0].amount == -50, (
        f"Expected -50 delta on org counter (50 actual - 100 reserved), got {matching[0].amount}"
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
        counters, _meta = handler._build_descriptor_atomic_payload(
            descriptor=descriptor,
            increment_amounts=inc,
        )
        scopes = {counter.counter.scope for counter in counters}
        expected_scope = RateLimitScope(descriptor["key"], descriptor["value"])
        assert scopes == {expected_scope}, (
            f"Descriptor {descriptor['key']}:{descriptor['value']} produced "
            f"counters spanning multiple scopes: {scopes}. Redis Cluster would "
            f"reject this Lua call with CROSSSLOT."
        )


@pytest.mark.asyncio
async def test_reservation_release_idempotent(recording_rate_limiter):
    """
    If both `async_post_call_failure_hook` and `async_log_failure_event` end
    up firing for the same request, only the first refund applies — the
    second sees the released marker and no-ops.
    """
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-idempotent")

    # Both hooks read the same per-request ContextVar stash: the
    # post-call-failure-hook flips reservation_released on it, and the
    # log-failure-event observes the flip.
    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    await handler.async_post_call_failure_hook(
        request_data={},
        original_exception=Exception("rejected"),
        user_api_key_dict=UserAPIKeyAuth(api_key=api_key),
    )

    first_refund_count = len([i for i in increments if i.counter.kind == "tokens"])
    assert first_refund_count > 0, "First refund should have applied"

    # Now simulate async_log_failure_event firing afterwards. It must see
    # the released flag on the stash and not double-refund.
    await handler.async_log_failure_event(
        kwargs={
            "standard_logging_object": {"metadata": {"user_api_key_hash": api_key}},
        },
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    second_refund_count = len([i for i in increments if i.counter.kind == "tokens"])
    assert second_refund_count == first_refund_count, (
        f"Idempotency violated: refund count went from {first_refund_count} "
        f"to {second_refund_count} after second hook fired."
    )


@pytest.mark.asyncio
async def test_unreserved_scopes_charged_actual_not_delta_on_success(recording_rate_limiter):
    """
    Counter-drift fix: a scope present in metadata but NOT reserved at
    pre-call (no configured TPM limit for it) must be charged the full
    `actual_tokens` on success — never the `delta = actual - reserved`.
    Otherwise that scope's counter goes negative whenever `actual < reserved`
    (the common case, since the reservation includes a conservative output
    pad).
    """
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-mixed-scopes")
    team_id = "team-no-tpm-limit"

    # Reservation ONLY hit api_key — team had no TPM limit configured.
    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
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

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    api_key_tokens_counter = handler.rate_limit_counter(key="api_key", value=api_key, rate_limit_type="tokens")
    team_tokens_counter = handler.rate_limit_counter(key="team", value=team_id, rate_limit_type="tokens")

    api_key_ops = [i for i in increments if i.counter == api_key_tokens_counter]
    team_ops = [i for i in increments if i.counter == team_tokens_counter]

    assert api_key_ops and api_key_ops[0].amount == -50, (
        f"Reserved api_key scope must reconcile via delta (50-100=-50), got {api_key_ops}"
    )
    assert team_ops and team_ops[0].amount == 50, (
        f"Unreserved team scope must be charged full actual (+50), not the "
        f"-50 delta (which would drift its counter negative). Got {team_ops}"
    )


@pytest.mark.asyncio
async def test_unreserved_scopes_not_refunded_on_failure(recording_rate_limiter):
    """
    Failure refund must only emit ops against scopes the reservation
    actually charged. Refunding an unreserved scope (which was never
    incremented at pre-call) would drive its counter to -reserved.
    """
    handler, _cache, increments = recording_rate_limiter

    api_key = hash_token("sk-mixed-fail")
    team_id = "team-no-tpm"

    stash = get_or_create_request_stash()
    stash.reserved_tokens = 100
    stash.reserved_scopes = frozenset({("api_key", api_key)})

    mock_kwargs = {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": api_key,
                "user_api_key_team_id": team_id,
            }
        },
    }

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    team_tokens_counter = handler.rate_limit_counter(key="team", value=team_id, rate_limit_type="tokens")
    api_key_tokens_counter = handler.rate_limit_counter(key="api_key", value=api_key, rate_limit_type="tokens")

    team_ops = [i for i in increments if i.counter == team_tokens_counter]
    api_key_ops = [i for i in increments if i.counter == api_key_tokens_counter]

    assert not team_ops, f"Unreserved team scope must NOT be refunded (would drift negative), got {team_ops}"
    assert api_key_ops and api_key_ops[0].amount == -100, (
        f"Reserved api_key scope must be refunded -100, got {api_key_ops}"
    )


@pytest.mark.asyncio
async def test_project_itpm_reconciled_on_success_excludes_cached_tokens(recording_rate_limiter):
    """
    On success, ITPM reconciles to billable input tokens (prompt_tokens
    minus cached_tokens) -- not raw prompt_tokens. Cached prompt-read tokens
    are free under Bedrock Mantle and must not count against the ITPM quota,
    even though they still appear in usage/cost logging elsewhere.
    """
    handler, _cache, increments = recording_rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-mantle:model")
    otpm_scope = ("model_per_project_otpm", "proj-mantle:model")

    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    mock_kwargs = {}

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

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [i for i in increments if i.counter.scope.descriptor_key == "model_per_project_itpm"]
    otpm_adjustments = [i for i in increments if i.counter.scope.descriptor_key == "model_per_project_otpm"]

    # billable_input = 80 - 30 cached = 50; delta = 50 - 100 reserved = -50
    assert any(i.amount == -50 for i in itpm_adjustments), (
        f"Expected a -50 ITPM adjustment (50 billable - 100 reserved), got: {itpm_adjustments}"
    )
    # delta = 40 actual completion - 60 reserved = -20
    assert any(i.amount == -20 for i in otpm_adjustments), (
        f"Expected a -20 OTPM adjustment (40 actual - 60 reserved), got: {otpm_adjustments}"
    )


@pytest.mark.asyncio
async def test_redis_window_guard_uses_reservation_identity_and_never_falls_back_negative():
    calls = []

    async def failing_guard(*, keys, args):
        calls.append((keys, args))
        raise RuntimeError("redis unavailable")

    rate_limits = FakeRateLimits(window_guarded_token_increment_script=failing_guard)
    handler = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=rate_limits.store(),
    )
    counter = handler.rate_limit_counter(PROJECT_ITPM_DESCRIPTOR_KEY, "project:model", "tokens")
    await handler.async_increment_reservation_aware_tokens(
        pipeline_operations=[
            WindowGuardedIncrement(
                increment=CounterIncrement(counter=counter, amount=-90, ttl=60),
                expected_window_start="1234",
                reservation_backend="redis",
            )
        ]
    )

    assert calls == [
        (
            [
                "{model_per_project_itpm:project:model}:window",
                "{model_per_project_itpm:project:model}:tokens",
            ],
            ["1234", -90, 60],
        )
    ]
    assert rate_limits.redis_increments == []
    assert handler.rate_limits.local((counter,)) == (None,)


@pytest.mark.asyncio
async def test_atomic_lua_response_carries_redis_window_identity():
    async def successful_reservation(*, keys, args):
        return [0, 25, 1234]

    handler = RateLimitHandler(
        internal_usage_cache=InternalUsageCache(DualCache()),
        rate_limit_store=FakeRateLimits(check_and_increment_by_n_script=successful_reservation).store(),
    )
    counter = handler.rate_limit_counter(PROJECT_ITPM_DESCRIPTOR_KEY, "project:model", "tokens")
    meta = [
        {
            "descriptor_key": PROJECT_ITPM_DESCRIPTOR_KEY,
            "descriptor_value": "project:model",
            "current_limit": 100,
            "rate_limit_type": "tokens",
            "counter": counter,
            "increment": 25,
            "ttl": 60,
            "window_size": 60,
        }
    ]

    assert await handler._atomic_lua_per_descriptor([]) == {
        "overall_code": "OK",
        "statuses": [],
    }

    response = await handler._atomic_lua_per_descriptor(
        descriptor_groups=[
            (
                [AtomicCounter(counter=counter, limit=100, increment=25, ttl=60, window_size=60)],
                meta,
            )
        ]
    )

    assert response["statuses"][0]["limit_remaining"] == 75
    assert response["reservation_windows"] == frozenset({(counter, "1234", "redis")})


@pytest.mark.asyncio
async def test_project_itpm_otpm_released_on_failure(recording_rate_limiter):
    """On failure, the full ITPM and OTPM reservations must be refunded."""
    handler, _cache, increments = recording_rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-mantle:model")
    otpm_scope = ("model_per_project_otpm", "proj-mantle:model")

    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    mock_kwargs = {}

    await handler.async_log_failure_event(
        kwargs=mock_kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_releases = [i for i in increments if i.counter.scope.descriptor_key == "model_per_project_itpm"]
    otpm_releases = [i for i in increments if i.counter.scope.descriptor_key == "model_per_project_otpm"]

    assert any(i.amount == -100 for i in itpm_releases), itpm_releases
    assert any(i.amount == -60 for i in otpm_releases), otpm_releases


@pytest.mark.asyncio
async def test_responses_api_usage_reconciles_using_input_output_tokens_fields(
    recording_rate_limiter,
):
    """
    Regression for the other half of the same finding: ResponseAPIUsage
    exposes input_tokens/output_tokens, not prompt_tokens/completion_tokens.
    Before this fix, _resolve_io_token_reconcile_usage couldn't resolve
    Responses API usage at all, so the reservation was silently kept as-is
    instead of being trued up to the much larger actual usage.
    """
    handler, _cache, increments = recording_rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-responses:model")
    otpm_scope = ("model_per_project_otpm", "proj-responses:model")

    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 10
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 10
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    mock_kwargs = {}

    mock_response = ResponsesAPIResponse(
        id="resp_test",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(input_tokens=80, output_tokens=400, total_tokens=480),
    )

    await handler.async_log_success_event(
        kwargs=mock_kwargs,
        response_obj=mock_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [i for i in increments if i.counter.scope.descriptor_key == "model_per_project_itpm"]
    otpm_adjustments = [i for i in increments if i.counter.scope.descriptor_key == "model_per_project_otpm"]

    # delta = 80 actual input - 10 reserved = +70
    assert any(i.amount == 70 for i in itpm_adjustments), (
        f"ITPM reservation was never trued up to actual Responses API usage: {itpm_adjustments}"
    )
    # delta = 400 actual output - 10 reserved = +390
    assert any(i.amount == 390 for i in otpm_adjustments), (
        f"OTPM reservation was never trued up to actual Responses API usage: {otpm_adjustments}"
    )


def test_audio_token_estimate_scales_with_payload_size():
    """
    Regression for veria-ai Low finding: audio token reservation was flat
    300 per block regardless of duration. A short clip and a long clip both
    reserved the same amount, letting a caller hide long audio in one block
    to exhaust ITPM quota while reserving almost nothing.

    The estimate must now grow proportionally with the base64 payload size
    (len(b64) * 3 // 4 // _AUDIO_BYTES_PER_TOKEN), floored at
    DEFAULT_AUDIO_TOKEN_ESTIMATE so reference-only blocks and genuinely
    short clips still get a non-trivial reservation.

    To exceed the floor the decoded payload must be > 300 * 1600 = 480 000
    bytes. We synthesise a fake b64-length string of 650 000 chars
    (decoded ≈ 487 500 bytes → 304 tokens) to avoid actually allocating
    and encoding ~480 kB of audio in every test run.
    """
    large_b64 = "A" * 650_000
    very_large_b64 = "A" * 12_900_000
    small_b64 = "A" * 1_000

    large_block = {
        "type": "input_audio",
        "input_audio": {"data": large_b64, "format": "wav"},
    }
    small_block = {
        "type": "input_audio",
        "input_audio": {"data": small_b64, "format": "wav"},
    }
    very_large_block = {
        "type": "input_audio",
        "input_audio": {"data": very_large_b64, "format": "wav"},
    }
    no_data_block = {"type": "input_audio", "input_audio": {"format": "wav"}}

    large_estimate = RateLimitHandler._estimate_audio_block_tokens(large_block)
    very_large_estimate = RateLimitHandler._estimate_audio_block_tokens(very_large_block)
    small_estimate = RateLimitHandler._estimate_audio_block_tokens(small_block)
    no_data_estimate = RateLimitHandler._estimate_audio_block_tokens(no_data_block)

    assert large_estimate > small_estimate, (
        f"Large payload ({large_estimate}) must reserve more than small payload "
        f"({small_estimate}); flat-rate bug is back"
    )
    assert very_large_estimate == len(very_large_b64) * 3 // 4 // _AUDIO_BYTES_PER_TOKEN
    assert very_large_estimate > 6_000
    assert no_data_estimate >= 300, (
        f"Reference-only block (no data) must use the DEFAULT_AUDIO_TOKEN_ESTIMATE floor; got {no_data_estimate}"
    )
    assert small_estimate >= 300, (
        f"Small payload must be floored at DEFAULT_AUDIO_TOKEN_ESTIMATE=300; got {small_estimate}"
    )


@pytest.mark.asyncio
async def test_itpm_otpm_reservation_is_kept_on_stream_disconnect(recording_rate_limiter):
    handler, cache, increment_calls = recording_rate_limiter

    api_key = hash_token("sk-disconnect-test")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=api_key,
        project_id="proj-disconnect",
        project_metadata={
            "model_itpm_limit": {"bedrock_mantle/claude-opus": 1000},
            "model_otpm_limit": {"bedrock_mantle/claude-opus": 500},
        },
    )

    data: Dict[str, Any] = {
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
    assert stash.itpm_reserved_tokens > 0, "pre-call hook must stash an ITPM reservation"
    assert stash.otpm_reserved_tokens > 0, "pre-call hook must stash an OTPM reservation"

    increment_calls.clear()

    await handler.async_release_max_parallel_requests_on_disconnect(user_api_key_dict=user_api_key_dict)

    itpm_refunds = [
        c for c in increment_calls if c.counter.scope.descriptor_key == "model_per_project_itpm" and c.amount < 0
    ]
    otpm_refunds = [
        c for c in increment_calls if c.counter.scope.descriptor_key == "model_per_project_otpm" and c.amount < 0
    ]

    assert not itpm_refunds
    assert not otpm_refunds
    assert stash.reservation_released is False


@pytest.mark.asyncio
async def test_responses_api_multimodal_input_counts_image_content(rate_limiter):
    """
    Regression for a Low-severity veria-ai finding: the Responses API's
    `input` is commonly a list of message/content-block dicts, but
    litellm.token_counter's `text` argument only joins plain string entries
    in a list and silently drops everything else -- so an `input_image`
    block contributed ~0 tokens to the ITPM estimate instead of the real
    image token count. _estimate_precise_input_tokens now converts Responses
    `input` to chat messages first (via the standard
    transform_responses_api_input_to_messages helper) so image content is
    counted the same way a chat completion's image content already is.
    """
    handler, _cache = rate_limiter

    text_only_estimate = handler._estimate_precise_input_tokens(
        data={"input": "hi"},
        model="bedrock_mantle/claude-opus",
        call_type="aresponses",
    )

    multimodal_estimate = handler._estimate_precise_input_tokens(
        data={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hi"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/some-image.png",
                        },
                    ],
                }
            ],
        },
        model="bedrock_mantle/claude-opus",
        call_type="aresponses",
    )

    assert multimodal_estimate > text_only_estimate + 100, (
        "Responses API input_image content block was not counted; got "
        f"text_only={text_only_estimate}, multimodal={multimodal_estimate}"
    )


@pytest.mark.asyncio
async def test_refund_reserved_tokens_noop_when_amount_zero(rate_limiter):
    """_refund_reserved_tokens returns immediately without calling Redis when amount=0."""
    handler, _cache = rate_limiter

    calls = []

    async def mock_increment(pipeline_operations, **kwargs):
        calls.extend(pipeline_operations)

    handler.async_increment_tokens_with_ttl_preservation = mock_increment

    await handler._refund_reserved_tokens(
        scopes=[("api_key", "sk-test")],
        amount=0,
    )

    assert not calls, "No Redis ops expected when amount is zero"


def test_strip_audio_content_blocks_passthrough_non_list_messages():
    """Non-list input is returned unchanged (early return on line 2605)."""
    result = RateLimitHandler._strip_audio_content_blocks("not a list")
    assert result == "not a list"


def test_strip_audio_content_blocks_passthrough_non_dict_message():
    """Non-dict entries in the message list are appended unchanged."""
    messages = ["plain string message"]
    result = RateLimitHandler._strip_audio_content_blocks(messages)
    assert result == ["plain string message"]


def test_strip_audio_content_blocks_passthrough_non_list_content():
    """Messages with non-list content (e.g. plain string) pass through unchanged."""
    messages = [{"role": "user", "content": "hello"}]
    result = RateLimitHandler._strip_audio_content_blocks(messages)
    assert result == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_otpm_rejection_releases_stashed_parallel_slot(rate_limiter):
    """
    When OTPM is over limit and a parallel slot was already acquired, the
    disconnect cleanup path in _reserve_project_io_tokens_or_raise must
    release that slot. Exercises lines 2773-2777.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-otpm-slot"),
        project_id="proj-slot",
        project_metadata={"model_otpm_limit": {"m": 5}},
    )

    data: Dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 50,
    }

    slot_released = []

    async def mock_release(acquisition, parent_otel_span=None):
        slot_released.append(acquisition)

    handler._release_parallel_request_slots = mock_release

    stash = get_or_create_request_stash()
    stash.parallel_slot = {
        "slot_id": "test-slot-id",
        "gauges": (RateLimitScope("api_key", "some-key"),),
    }

    otpm_descriptor = {
        "key": PROJECT_OTPM_DESCRIPTOR_KEY,
        "value": "proj-slot:m",
        "rate_limit": {"tokens_per_unit": 5, "window_size": 60},
    }

    with pytest.raises(Exception, match="Rate limit exceeded for model_per_project_otpm") as exc_info:
        await handler._reserve_project_io_tokens_or_raise(
            descriptors=[otpm_descriptor],
            data=data,
            requested_model="m",
            user_api_key_dict=user_api_key_dict,
            tpm_reservation_scopes=[],
            tpm_reservation_amount=0,
        )
    assert getattr(exc_info.value, "status_code", None) == 429
    assert slot_released, "Parallel slot must be released when OTPM rejects"
    assert stash.parallel_slot is None


@pytest.mark.asyncio
async def test_itpm_only_status_stored_when_no_prior_rate_limit_response(rate_limiter):
    """
    When only ITPM is configured (no combined TPM/RPM to pre-populate
    the request stash), a successful ITPM reservation must store its status
    there so post-call headers can read it.
    """
    handler, cache = rate_limiter

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("sk-itpm-only-store"),
        project_id="proj-store",
    )

    data: Dict[str, Any] = {"model": "m", "messages": []}

    itpm_descriptor = {
        "key": PROJECT_ITPM_DESCRIPTOR_KEY,
        "value": "proj-store:m",
        "rate_limit": {"tokens_per_unit": 100000, "window_size": 60},
    }

    await handler._reserve_project_io_tokens_or_raise(
        descriptors=[itpm_descriptor],
        data=data,
        requested_model="m",
        user_api_key_dict=user_api_key_dict,
        tpm_reservation_scopes=[],
        tpm_reservation_amount=0,
    )

    stash = get_request_stash()
    assert stash is not None
    stored = stash.rate_limit_response
    assert stored is not None, "ITPM status must be stored in litellm_proxy_rate_limit_response"
    assert stored.get("statuses"), "Stored response must contain statuses"


def test_resolve_io_token_usage_responses_api_with_cached_tokens(rate_limiter):
    """
    ResponsesAPIResponse whose usage.input_tokens_details.cached_tokens is set
    subtracts the cached portion from billable input. Covers line 3501.
    """
    handler, _cache = rate_limiter

    response_obj = ResponsesAPIResponse(
        id="resp_cached",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=InputTokensDetails(cached_tokens=25),
        ),
    )
    billable_input, completion_tokens, resolved = handler._resolve_io_token_reconcile_usage(response_obj)

    assert resolved is True
    assert billable_input == 75, f"Expected 100 - 25 cached = 75, got {billable_input}"
    assert completion_tokens == 50


def test_resolve_io_token_usage_dict_format(rate_limiter):
    """
    Dict-shaped usage on a ModelResponse (older SDK versions or raw dicts in
    the usage field) is parsed correctly. Covers lines 3502-3506.
    """
    handler, _cache = rate_limiter

    response_obj = ModelResponse.model_construct(
        usage={
            "prompt_tokens": 80,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
    )
    billable_input, completion_tokens, resolved = handler._resolve_io_token_reconcile_usage(response_obj)

    assert resolved is True
    assert billable_input == 60, f"Expected 80 - 20 cached = 60, got {billable_input}"
    assert completion_tokens == 40


def test_resolve_io_token_usage_unknown_type_returns_unresolved(rate_limiter):
    """
    A ModelResponse whose usage attribute is not a Usage, ResponseAPIUsage,
    or dict (e.g. a plain int) returns (0, 0, False) so the reservation is
    kept rather than guessed. Covers lines 3507-3508.
    """
    handler, _cache = rate_limiter

    response_obj = ModelResponse.model_construct(usage=42)
    billable_input, completion_tokens, resolved = handler._resolve_io_token_reconcile_usage(response_obj)

    assert resolved is False
    assert billable_input == 0
    assert completion_tokens == 0


@pytest.mark.parametrize(
    ("combined_usage", "expected_increments"),
    [
        (None, ()),
        (
            Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            (-60, -45),
        ),
    ],
)
def test_zero_usage_keeps_reservations_unless_measured_fallback_exists(
    rate_limiter,
    combined_usage,
    expected_increments,
):
    handler, _cache = rate_limiter
    itpm_scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    otpm_scope = (PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    kwargs = {} if combined_usage is None else {"combined_usage_object": combined_usage}
    response_obj = ModelResponse(usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0))

    operations = handler._build_io_token_reservation_ops(kwargs, response_obj)

    assert tuple(operation.amount for operation in operations) == expected_increments


def test_a_responses_reply_reporting_no_tokens_keeps_the_reservations(rate_limiter):
    handler, _cache = rate_limiter
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({(PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({(PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")})
    response_obj = ResponsesAPIResponse(
        id="resp_zero",
        created_at=int(datetime.now().timestamp()),
        output=[],
        usage=ResponseAPIUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    )

    assert tuple(handler._build_io_token_reservation_ops({}, response_obj)) == ()


@pytest.mark.parametrize(
    ("usage", "expected_increments"),
    [
        (
            Usage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            (40, 15),
        ),
        (
            Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            (100, 60),
        ),
    ],
)
def test_retry_success_charges_released_project_io_reservations(
    rate_limiter,
    usage,
    expected_increments,
):
    handler, _cache = rate_limiter
    itpm_scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    otpm_scope = (PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    stash.mark_reservation_released()

    operations = handler._build_io_token_reservation_ops(
        {},
        ModelResponse(usage=usage),
    )

    assert tuple(operation.amount for operation in operations) == expected_increments


@pytest.mark.asyncio
async def test_build_io_token_reservation_ops_skips_unresolvable_usage(rate_limiter):
    """
    When response_obj has no parseable usage, _build_io_token_reservation_ops
    returns [] to keep the reservation as-is rather than zeroing it out on a
    bad guess. Covers line 3538.
    """
    handler, _cache = rate_limiter

    itpm_scope = ("model_per_project_itpm", "proj-b:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 50
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    mock_kwargs = {}

    ops = handler._build_io_token_reservation_ops(
        kwargs=mock_kwargs,
        response_obj=object(),
    )

    assert not ops, f"Expected empty ops for unresolvable usage, got {ops}"


def test_responses_input_is_counted_even_when_messages_is_present(rate_limiter):
    handler, _cache = rate_limiter
    small_estimate = handler._estimate_precise_input_tokens(
        data={"input": "short", "messages": [{"role": "user", "content": "ignored"}]},
        model="",
        call_type="aresponses",
    )
    large_estimate = handler._estimate_precise_input_tokens(
        data={"input": "large input " * 500, "messages": []},
        model="",
        call_type="aresponses",
    )

    assert large_estimate > small_estimate


def test_anthropic_messages_usage_reconciles_split_project_quota(rate_limiter):
    handler, _cache = rate_limiter

    billable_input, output_tokens, resolved = handler._resolve_io_token_reconcile_usage(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
                "cache_read_input_tokens": 30,
            }
        }
    )

    assert resolved is True
    assert billable_input == 70
    assert output_tokens == 25


def test_rerank_input_estimate_falls_back_to_character_count(
    rate_limiter,
    monkeypatch,
):
    handler, _cache = rate_limiter
    data = {
        "query": "query text",
        "documents": ["first document", "second document"],
    }

    def token_counter(**_kwargs):
        raise ValueError("tokenizer unavailable")

    monkeypatch.setattr("litellm.token_counter", token_counter)
    rerank_text = handler._rerank_input_to_text(data)

    assert (
        handler._estimate_precise_input_tokens(
            data,
            model="custom-rerank-model",
            call_type="rerank",
        )
        == len(rerank_text) // 4
    )


@pytest.mark.parametrize(
    ("response_obj", "expected"),
    [
        (
            RerankResponse(meta={"tokens": {"input_tokens": 42, "output_tokens": 3}}),
            (42, 3, True),
        ),
        (
            RerankResponse(
                meta={
                    "tokens": {"input_tokens": 0, "output_tokens": 0},
                    "billed_units": {"total_tokens": 57},
                }
            ),
            (57, 0, True),
        ),
        (
            RerankResponse(
                meta={
                    "tokens": {"input_tokens": 0, "output_tokens": 0},
                    "billed_units": {"total_tokens": 0},
                }
            ),
            (0, 0, False),
        ),
    ],
)
def test_rerank_usage_reconciles_project_split_token_quota(
    rate_limiter,
    response_obj,
    expected,
):
    handler, _cache = rate_limiter

    assert handler._resolve_io_token_reconcile_usage(response_obj) == expected


def test_split_quota_helpers_handle_non_mapping_inputs(rate_limiter):
    handler, _cache = rate_limiter

    assert _call_id_from_callback_kwargs(object()) is None
    assert handler._is_embedding_request(object(), None) is False
    assert handler._get_explicit_output_cap(object(), None) is None
    assert handler.get_output_candidate_count(object()) == 1
    assert handler.get_output_candidate_count({"n": 1e309}) == 1
    assert handler._get_explicit_output_cap({"max_output_tokens": []}, "responses") is None
    assert handler._apply_implicit_output_cap(object(), 100, "responses") is None
    assert handler._estimate_input_and_output_tokens(object()) == (0, 0)
    assert handler._build_io_token_reservation_ops(object(), object()) == ()


@pytest.mark.parametrize(
    ("data", "call_type", "expected"),
    [
        ({"max_tokens": "30.0"}, "", 30),
        ({"max_tokens": "not-a-number"}, "", None),
        ({"max_tokens": True}, "", None),
        ({"max_output_tokens": "30.0"}, "responses", 30),
        ({"max_output_tokens": "nan"}, "responses", None),
        ({"generationConfig": {"maxOutputTokens": "12.5"}}, "agenerate_content", 12),
        ({"generationConfig": {"maxOutputTokens": "oops"}}, "agenerate_content", None),
    ],
)
def test_get_explicit_output_cap_tolerates_unparseable_values(rate_limiter, data, call_type, expected):
    """A client-supplied cap the proxy cannot parse must fall back to the
    no-cap output estimate instead of raising ValueError and 500ing the
    request before it ever reaches the provider."""
    handler, _cache = rate_limiter

    assert handler._get_explicit_output_cap(data, call_type) == expected


def test_split_quota_multimodal_guards_handle_non_mapping_inputs(rate_limiter):
    handler, _cache = rate_limiter

    assert handler._estimate_audio_block_tokens(object()) == handler._estimate_audio_block_tokens({})
    assert handler._responses_input_to_chat_messages(object()) == ()
    assert handler._estimate_precise_input_tokens(object(), model=None) == 0


@pytest.mark.parametrize(
    ("call_type", "data", "expected_text"),
    [
        ("embedding", {"input": "embedding input"}, "embedding input"),
        (
            "embedding",
            {"input": ["first embedding", "second embedding"]},
            ["first embedding", "second embedding"],
        ),
        ("text_completion", {"prompt": "completion prompt"}, "completion prompt"),
    ],
)
def test_precise_input_estimate_selects_endpoint_text(
    rate_limiter,
    monkeypatch,
    call_type,
    data,
    expected_text,
):
    handler, _cache = rate_limiter
    captured = {}

    def token_counter(**kwargs):
        captured.update(kwargs)
        return 7

    monkeypatch.setattr("litellm.token_counter", token_counter)

    assert handler._estimate_precise_input_tokens(data, model="test", call_type=call_type) == 7
    assert captured["messages"] is None
    assert captured["text"] == expected_text


@pytest.mark.asyncio
async def test_project_io_reservation_ignores_non_mapping_request_data(rate_limiter):
    handler, _cache = rate_limiter

    await handler._reserve_project_io_tokens_or_raise(
        descriptors=[],
        data=object(),
        requested_model=None,
        user_api_key_dict=UserAPIKeyAuth(),
        tpm_reservation_scopes=(),
        tpm_reservation_amount=0,
    )


@pytest.mark.asyncio
async def test_streaming_combined_usage_reconciles_project_io_reservations(
    recording_rate_limiter,
):
    handler, _cache, increments = recording_rate_limiter
    itpm_scope = (PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")
    otpm_scope = (PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({itpm_scope})
    stash.otpm_reserved_tokens = 60
    stash.otpm_reserved_scopes = frozenset({otpm_scope})
    kwargs = {
        "combined_usage_object": Usage(
            prompt_tokens=40,
            completion_tokens=15,
            total_tokens=55,
        ),
    }

    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj={"response": "stream body"},
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    itpm_adjustments = [
        operation for operation in increments if operation.counter.scope.descriptor_key == PROJECT_ITPM_DESCRIPTOR_KEY
    ]
    otpm_adjustments = [
        operation for operation in increments if operation.counter.scope.descriptor_key == PROJECT_OTPM_DESCRIPTOR_KEY
    ]
    assert [operation.amount for operation in itpm_adjustments] == [-60]
    assert [operation.amount for operation in otpm_adjustments] == [-45]


def test_aggregate_only_combined_usage_reconciles_project_io_reservations(rate_limiter):
    handler, _cache = rate_limiter
    stash = get_or_create_request_stash()
    stash.itpm_reserved_tokens = 100
    stash.itpm_reserved_scopes = frozenset({(PROJECT_ITPM_DESCRIPTOR_KEY, "project:model")})
    stash.otpm_reserved_tokens = 80
    stash.otpm_reserved_scopes = frozenset({(PROJECT_OTPM_DESCRIPTOR_KEY, "project:model")})
    kwargs = {
        "combined_usage_object": Usage(total_tokens=55),
    }

    operations = handler._build_io_token_reservation_ops(kwargs, object())

    assert [operation.amount for operation in operations] == [-45, -25]


def test_raw_split_usage_dict_reconciles_project_io_tokens(rate_limiter):
    handler, _cache = rate_limiter

    assert handler._resolve_io_token_reconcile_usage(
        {
            "input_tokens": 30,
            "output_tokens": 12,
            "input_tokens_details": {"cached_tokens": 5},
        }
    ) == (25, 12, True)


@pytest.mark.asyncio
async def test_post_call_success_hook_contains_header_merge_failures(rate_limiter, monkeypatch):
    handler, _cache = rate_limiter
    response = ModelResponse()
    response._hidden_params = {}

    def raise_on_merge(**_kwargs):
        raise RuntimeError("header merge failed")

    monkeypatch.setattr(
        handler,
        "_merge_ratelimit_statuses_into_additional_headers",
        raise_on_merge,
    )

    await handler.async_post_call_success_hook(
        data={
            "litellm_proxy_rate_limit_response": {
                "overall_code": "OK",
                "statuses": (),
            }
        },
        user_api_key_dict=UserAPIKeyAuth(),
        response=response,
    )


@pytest.mark.asyncio
async def test_the_project_itpm_reservation_counts_the_request_off_the_event_loop(rate_limiter):
    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    handler, _cache = rate_limiter
    stash = get_or_create_request_stash()
    warm_tokenizer("claude-fable-5")
    data: dict[str, object] = {
        "model": "claude-fable-5",
        "messages": [{"role": "user", "content": text * 100}],
    }
    itpm_descriptor = {
        "key": PROJECT_ITPM_DESCRIPTOR_KEY,
        "value": "proj-loop:claude-fable-5",
        "rate_limit": {"tokens_per_unit": 10_000_000, "window_size": 60},
    }

    _, took, lags = await timed_with_loop_lags(
        lambda: handler._reserve_project_io_tokens_or_raise(
            descriptors=[itpm_descriptor],
            data=data,
            requested_model="claude-fable-5",
            user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-itpm-loop"), project_id="proj-loop"),
            tpm_reservation_scopes=[],
            tpm_reservation_amount=0,
        )
    )

    assert stash.rate_limit_response is not None
    assert_loop_stayed_free(took, lags)


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


def test_reservation_skips_the_history_a_compaction_item_replaces(rate_limiter):
    handler, _cache = rate_limiter
    compacted_request: Final = {"model": "gpt-4o", "input": _COMPACTED_INPUT, "max_output_tokens": 1}

    assert handler._estimate_input_and_output_tokens(
        data=compacted_request, call_type="aresponses"
    ) == handler._estimate_input_and_output_tokens(
        data={**compacted_request, "input": _COMPACTED_INPUT[2:]}, call_type="aresponses"
    )
    assert handler._estimate_precise_input_tokens(
        compacted_request, model="gpt-4o", call_type="aresponses"
    ) == handler._estimate_precise_input_tokens(
        {**compacted_request, "input": _SUMMARY_INPUT}, model="gpt-4o", call_type="aresponses"
    )
