"""
Tests for separate ITPM/OTPM deployment rate limits (enforce_model_rate_limits).
"""

import asyncio

import pytest

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    ITPM_CACHE_KEY,
    ITPM_RESERVED_KEY,
    OTPM_CACHE_KEY,
    OTPM_RESERVED_KEY,
    _reservation_value,
    _resolve_max_tokens,
    async_io_token_pre_call_check,
    async_io_token_reconcile_success,
    build_io_token_rate_limit_headers,
    deployment_has_io_token_limits,
    get_io_token_rate_limit_request_kwargs,
    io_token_reconcile_success,
    io_token_refund_failure,
    refund_stale_reservation_before_retry,
    set_io_token_rate_limit_request_kwargs,
)
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)
from litellm.types.utils import ModelResponse, Usage


class TestIOTokenRateLimitHelpers:
    def test_deployment_has_io_token_limits(self):
        assert deployment_has_io_token_limits({"litellm_params": {"itpm": 100, "otpm": 50}})
        assert not deployment_has_io_token_limits({"litellm_params": {"model": "x"}})

    def test_reservation_value_minimal_when_estimate_fails(self):
        # A failed/empty estimate (0) must reserve a minimal slot, not the
        # entire limit - otherwise one request whose estimate failed fills
        # the whole bucket and blocks every concurrent request until it
        # completes and reconciles.
        assert _reservation_value(0, 100) == 1
        assert _reservation_value(0, 1) == 1
        # A real non-zero estimate is reserved as-is.
        assert _reservation_value(42, 100) == 42

    def test_resolve_max_tokens_respects_explicit_zero(self):
        deployment = {"litellm_params": {"model": "openai/gpt-4o-mini"}}
        # An explicit max_tokens=0 is honored, not replaced by the model default.
        assert _resolve_max_tokens({"max_tokens": 0}, deployment) == 0
        # max_completion_tokens is the fallback only when max_tokens is absent.
        assert _resolve_max_tokens({"max_completion_tokens": 12}, deployment) == 12
        assert _resolve_max_tokens({"max_output_tokens": 9}, deployment) == 9

    def test_build_io_token_rate_limit_headers(self):
        headers = build_io_token_rate_limit_headers(
            itpm_limit=200,
            otpm_limit=40,
            current_itpm=15,
            current_otpm=4,
        )
        assert headers["x-ratelimit-limit-input-tokens"] == 200
        assert headers["x-ratelimit-remaining-input-tokens"] == 185
        assert headers["x-ratelimit-limit-output-tokens"] == 40
        assert headers["x-ratelimit-remaining-output-tokens"] == 36


class TestModelRateLimitingCheckIOTokens:
    @pytest.mark.asyncio
    async def test_itpm_reservation_and_reconcile(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": 100,
                "otpm": 50,
            },
            "model_info": {"id": "io-test-id"},
            "model_name": "opus",
        }

        request_kwargs = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(request_kwargs)
        await check.async_pre_call_check(deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-test-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        otpm_key = f"global_router:io-test-id:bedrock_mantle/anthropic.claude-opus-4-7:otpm:{minute}"

        kwargs = {
            "standard_logging_object": {
                "model_id": "io-test-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/anthropic.claude-opus-4-7"},
                "metadata": dict(request_kwargs["metadata"]),
            },
            "metadata": request_kwargs["metadata"],
        }
        response = ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        await check.async_log_success_event(kwargs, response, None, None)

        current_itpm = await dual_cache.async_get_cache(key=itpm_key)
        current_otpm = await dual_cache.async_get_cache(key=otpm_key)
        # ITPM tracks input tokens only (billable prompt tokens), not output.
        assert current_itpm == 5
        assert current_otpm == 3

    @pytest.mark.asyncio
    async def test_itpm_limit_raises_429(self):
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": 5,
            },
            "model_info": {"id": "io-limit-id"},
            "model_name": "opus",
        }

        # ITPM enforces input tokens only; the prompt alone must exceed the limit,
        # a large max_tokens must not contribute to the ITPM reservation.
        set_io_token_rate_limit_request_kwargs(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello world this is a longer prompt that exceeds the tiny itpm limit",
                    }
                ],
                "max_tokens": 10,
                "metadata": {},
            }
        )

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "ITPM limit=5" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_otpm_atomic_reservation_no_overshoot_under_concurrency(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        otpm_limit = 10
        max_tokens = 4
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "otpm": otpm_limit,
            },
            "model_info": {"id": "io-otpm-race-id"},
            "model_name": "opus",
        }

        set_io_token_rate_limit_request_kwargs(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": max_tokens,
                "metadata": {},
            }
        )

        async def _attempt():
            try:
                await check.async_pre_call_check(deployment)
                return True
            except litellm.RateLimitError:
                return False

        results = await asyncio.gather(*[_attempt() for _ in range(8)])
        successes = sum(1 for r in results if r)

        minute = get_utc_datetime().strftime("%H-%M")
        otpm_key = f"global_router:io-otpm-race-id:bedrock_mantle/anthropic.claude-opus-4-7:otpm:{minute}"
        current_otpm = await dual_cache.async_get_cache(key=otpm_key)

        # Atomic reservation must never let concurrent requests overshoot the limit.
        assert current_otpm is not None
        assert current_otpm <= otpm_limit
        assert successes == otpm_limit // max_tokens
        assert current_otpm == successes * max_tokens

    @pytest.mark.asyncio
    async def test_itpm_estimate_failure_reserves_minimal_not_full_limit(self):
        """
        When input-token estimation yields 0 (no messages/prompt/input field,
        unsupported model, tokenizer error), the reservation must be a
        minimal 1 token, not the entire itpm limit. Otherwise the first
        request whose estimate fails fills the whole bucket and every
        concurrent request is rejected until it completes - effectively
        serializing traffic to the deployment.
        """
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        itpm_limit = 5
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": itpm_limit,
            },
            "model_info": {"id": "io-itpm-estimate-fail-id"},
            "model_name": "opus",
        }

        # No messages/prompt/input field -> _estimate_input_tokens returns 0.
        set_io_token_rate_limit_request_kwargs(
            {
                "max_tokens": 5,
                "metadata": {},
            }
        )

        async def _attempt():
            try:
                await check.async_pre_call_check(deployment)
                return True
            except litellm.RateLimitError:
                return False

        results = await asyncio.gather(*[_attempt() for _ in range(8)])
        successes = sum(1 for r in results if r)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-itpm-estimate-fail-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        current_itpm = await dual_cache.async_get_cache(key=itpm_key)

        # A minimal 1-token reservation per request lets itpm_limit concurrent
        # requests through, instead of a single request starving the rest.
        assert current_itpm is not None
        assert current_itpm <= itpm_limit
        assert successes == itpm_limit

    @pytest.mark.asyncio
    async def test_reservation_read_prefers_top_level_metadata_over_litellm_params(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-lp-id:bedrock_mantle/test:itpm:{minute}"
        await dual_cache.async_increment_cache(key=itpm_key, value=20, ttl=60)

        # Production kwargs commonly carry litellm_params.metadata; the stashed
        # reservation lives in the top-level metadata and must still be found.
        kwargs = {
            "standard_logging_object": {
                "model_id": "io-lp-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {},
            },
            "metadata": {ITPM_RESERVED_KEY: 20, ITPM_CACHE_KEY: itpm_key},
            "litellm_params": {"metadata": {"user_api_key_hash": "abc123"}},
        }
        await check.async_log_failure_event(kwargs, None, None, None)

        current = await dual_cache.async_get_cache(key=itpm_key)
        assert current == 0

    @pytest.mark.asyncio
    async def test_reconcile_tracks_actual_usage_when_estimate_zero(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": 100,
            },
            "model_info": {"id": "io-zero-est-id"},
            "model_name": "opus",
        }

        request_kwargs = {"max_tokens": 5, "metadata": {}}
        set_io_token_rate_limit_request_kwargs(request_kwargs)
        await check.async_pre_call_check(deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-zero-est-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        # A failed/zero estimate reserves a minimal 1 token, not the full
        # itpm limit, so it doesn't starve concurrent requests.
        assert await dual_cache.async_get_cache(key=itpm_key) == 1

        kwargs = {
            "standard_logging_object": {
                "model_id": "io-zero-est-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/anthropic.claude-opus-4-7"},
                "metadata": dict(request_kwargs["metadata"]),
            },
            "metadata": request_kwargs["metadata"],
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=7, completion_tokens=0, total_tokens=7),
        )
        await check.async_log_success_event(kwargs, response, None, None)

        assert await dual_cache.async_get_cache(key=itpm_key) == 7

    @pytest.mark.asyncio
    async def test_zero_estimate_reserves_minimal_capacity_before_reconcile(self):
        """
        A zero/failed estimate reserves a minimal 1 token rather than the
        full itpm limit, so up to itpm_limit such calls are allowed
        concurrently instead of the first one claiming the entire bucket.
        """
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": 2,
            },
            "model_info": {"id": "io-zero-cap-id"},
            "model_name": "opus",
        }
        request_kwargs = {"max_tokens": 5, "metadata": {}}
        set_io_token_rate_limit_request_kwargs(request_kwargs)
        await check.async_pre_call_check(deployment)

        # Second zero-estimate call still fits within the itpm=2 limit.
        set_io_token_rate_limit_request_kwargs({"max_tokens": 5, "metadata": {}})
        await check.async_pre_call_check(deployment)

        # A third exceeds the limit and is rejected.
        set_io_token_rate_limit_request_kwargs({"max_tokens": 5, "metadata": {}})
        with pytest.raises(litellm.RateLimitError):
            await check.async_pre_call_check(deployment)

    @pytest.mark.asyncio
    async def test_explicit_zero_max_tokens_does_not_reserve_otpm(self):
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "otpm": 5,
            },
            "model_info": {"id": "io-zero-output-id"},
            "model_name": "opus",
        }
        zero_output_kwargs = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 0,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(zero_output_kwargs)
        await check.async_pre_call_check(deployment)

        zero_output_otpm_key = zero_output_kwargs["metadata"][OTPM_CACHE_KEY]
        assert zero_output_kwargs["metadata"][OTPM_RESERVED_KEY] == 0
        assert (await dual_cache.async_get_cache(key=zero_output_otpm_key) or 0) == 0

        normal_output_kwargs = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(normal_output_kwargs)
        await check.async_pre_call_check(deployment)

        normal_output_otpm_key = normal_output_kwargs["metadata"][OTPM_CACHE_KEY]
        assert await dual_cache.async_get_cache(key=normal_output_otpm_key) == 5

    def test_sync_io_pre_call_reserves_and_reconciles(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": 100,
                "otpm": 50,
            },
            "model_info": {"id": "io-sync-id"},
            "model_name": "opus",
        }
        request_kwargs = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(request_kwargs)
        check.pre_call_check(deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-sync-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        otpm_key = f"global_router:io-sync-id:bedrock_mantle/anthropic.claude-opus-4-7:otpm:{minute}"
        kwargs = {
            "standard_logging_object": {
                "model_id": "io-sync-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/anthropic.claude-opus-4-7"},
                "metadata": dict(request_kwargs["metadata"]),
            },
            "metadata": request_kwargs["metadata"],
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        check.log_success_event(kwargs, response, None, None)

        assert dual_cache.get_cache(key=itpm_key) == 5
        assert dual_cache.get_cache(key=otpm_key) == 3

    @pytest.mark.asyncio
    async def test_reconcile_runs_via_success_event_without_model_id(self):
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        itpm_key = "global_router:io-noid:bedrock_mantle/test:itpm:12-34"
        await dual_cache.async_increment_cache(key=itpm_key, value=10, ttl=60)

        # standard_logging_object has no model_id (only the TPM path needs it);
        # IO reconciliation must still run off the stashed cache key.
        kwargs = {
            "standard_logging_object": {
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {},
            },
            "metadata": {ITPM_RESERVED_KEY: 10, ITPM_CACHE_KEY: itpm_key},
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )
        await check.async_log_success_event(kwargs, response, None, None)

        assert await dual_cache.async_get_cache(key=itpm_key) == 3

    @pytest.mark.asyncio
    async def test_failure_clears_reservation_so_retry_is_not_poisoned(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-first:bedrock_mantle/test:itpm:{minute}"
        await dual_cache.async_increment_cache(key=itpm_key, value=8, ttl=60)

        # Shared request metadata carrying the first (IO) deployment's reservation.
        metadata = {ITPM_RESERVED_KEY: 8, ITPM_CACHE_KEY: itpm_key}
        fail_kwargs = {
            "metadata": metadata,
            "standard_logging_object": {
                "model_id": "io-first",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {},
            },
        }
        await check.async_log_failure_event(fail_kwargs, None, None, None)

        assert await dual_cache.async_get_cache(key=itpm_key) == 0
        assert ITPM_RESERVED_KEY not in metadata
        assert ITPM_CACHE_KEY not in metadata

        # Retry succeeds on a non-IO fallback deployment reusing the same metadata.
        retry_kwargs = {
            "metadata": metadata,
            "standard_logging_object": {
                "model_id": "non-io-second",
                "hidden_params": {"litellm_model_name": "openai/gpt-4o-mini"},
                "metadata": {},
                "total_tokens": 12,
            },
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=6, completion_tokens=6, total_tokens=12),
        )
        await check.async_log_success_event(retry_kwargs, response, None, None)

        # The first deployment's ITPM counter is not driven negative...
        assert await dual_cache.async_get_cache(key=itpm_key) == 0
        # ...and the non-IO deployment's TPM usage is tracked normally.
        tpm_key = f"non-io-second:openai/gpt-4o-mini:tpm:{minute}"
        assert await dual_cache.async_get_cache(key=tpm_key) == 12

    @pytest.mark.asyncio
    async def test_stale_reservation_refunded_before_retry_overwrites_it(self):
        """
        A retry reuses the same mutable kwargs dict for the next deployment.
        If deployment A's failure event hasn't run yet (e.g. it was scheduled
        as a background task) when the retry calls
        set_io_token_rate_limit_request_kwargs for deployment B, the router
        must first synchronously refund + clear A's reservation via
        refund_stale_reservation_before_retry - otherwise A's counter stays
        elevated by the reservation until its TTL expires, and the
        now-orphaned sentinels must not leak into B's accounting either.
        """
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key_a = f"global_router:io-retry-a:bedrock_mantle/test-a:itpm:{minute}"
        await dual_cache.async_increment_cache(key=itpm_key_a, value=9, ttl=60)

        # Deployment A's still-unreconciled reservation, stashed on the shared
        # kwargs dict the retry loop reuses.
        shared_kwargs = {"metadata": {ITPM_RESERVED_KEY: 9, ITPM_CACHE_KEY: itpm_key_a}}

        # Router calls this before overwriting kwargs for deployment B's attempt -
        # simulating the fix landing ahead of set_io_token_rate_limit_request_kwargs.
        refund_stale_reservation_before_retry(dual_cache, shared_kwargs)

        # A's reservation is refunded immediately, not left stranded for a
        # background failure task that may run arbitrarily later (or never,
        # if the sentinels get cleared out from under it first).
        assert await dual_cache.async_get_cache(key=itpm_key_a) == 0
        assert ITPM_RESERVED_KEY not in shared_kwargs["metadata"]
        assert ITPM_CACHE_KEY not in shared_kwargs["metadata"]

        # A's own (now-late) failure event finds nothing left to refund and
        # is a safe no-op, since the sentinels were already cleared above.
        io_token_refund_failure(dual_cache, shared_kwargs)
        assert await dual_cache.async_get_cache(key=itpm_key_a) == 0

        # The retry proceeds to stash deployment B's own reservation on the
        # same dict; it starts clean, unaffected by A's cleared sentinels.
        set_io_token_rate_limit_request_kwargs(shared_kwargs)
        itpm_key_b = f"global_router:io-retry-b:bedrock_mantle/test-b:itpm:{minute}"
        shared_kwargs["metadata"][ITPM_RESERVED_KEY] = 4
        shared_kwargs["metadata"][ITPM_CACHE_KEY] = itpm_key_b
        await dual_cache.async_increment_cache(key=itpm_key_b, value=4, ttl=60)
        assert await dual_cache.async_get_cache(key=itpm_key_b) == 4

    @pytest.mark.asyncio
    async def test_client_supplied_reservation_keys_are_stripped(self):
        # metadata is caller-controlled; the server-only reservation sentinels
        # must be removed before the router captures the request kwargs.
        forged = {
            "metadata": {ITPM_RESERVED_KEY: 999999, ITPM_CACHE_KEY: "attacker:key:itpm:00-00"},
            "litellm_metadata": {OTPM_RESERVED_KEY: 7},
            "litellm_params": {"metadata": {OTPM_CACHE_KEY: "attacker:key:otpm:00-00"}},
        }
        set_io_token_rate_limit_request_kwargs(forged)
        stored = get_io_token_rate_limit_request_kwargs()

        assert ITPM_RESERVED_KEY not in stored["metadata"]
        assert ITPM_CACHE_KEY not in stored["metadata"]
        assert OTPM_RESERVED_KEY not in stored["litellm_metadata"]
        assert OTPM_CACHE_KEY not in stored["litellm_params"]["metadata"]

    @pytest.mark.asyncio
    async def test_forged_reservation_cannot_decrement_counter(self):
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        victim_key = "global_router:victim:model:itpm:00-00"
        await dual_cache.async_increment_cache(key=victim_key, value=100, ttl=60)

        # A caller forges a reservation pointing at another deployment's counter.
        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 100, ITPM_CACHE_KEY: victim_key},
            "standard_logging_object": {
                "model_id": "m",
                "hidden_params": {"litellm_model_name": "model"},
                "metadata": {},
                "total_tokens": 2,
            },
        }
        # The router sanitizes the request kwargs before the call runs.
        set_io_token_rate_limit_request_kwargs(kwargs)
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        await check.async_log_success_event(kwargs, response, None, None)

        # The forged reservation was stripped, so the victim counter is untouched.
        assert await dual_cache.async_get_cache(key=victim_key) == 100

    @pytest.mark.asyncio
    async def test_otpm_reservation_error_rolls_back_itpm(self):
        from litellm.utils import get_utc_datetime

        class _OtpmFailCache(DualCache):
            async def async_increment_cache(self, key, **kwargs):
                if ":otpm:" in key:
                    raise RuntimeError("transient cache error")
                return await super().async_increment_cache(key=key, **kwargs)

        dual_cache = _OtpmFailCache()
        deployment = {
            "litellm_params": {
                "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                "itpm": 1000,
                "otpm": 1000,
            },
            "model_info": {"id": "io-rollback-id"},
            "model_name": "opus",
        }
        set_io_token_rate_limit_request_kwargs(
            {
                "messages": [{"role": "user", "content": "hello world"}],
                "max_tokens": 5,
                "metadata": {},
            }
        )

        with pytest.raises(RuntimeError):
            await async_io_token_pre_call_check(dual_cache, deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-rollback-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        # A transient OTPM error must release the ITPM reservation, not leak it.
        assert (await dual_cache.async_get_cache(key=itpm_key) or 0) == 0

    @pytest.mark.asyncio
    async def test_reconcile_clears_stash_even_when_increment_errors(self):
        class _ItpmFailCache(DualCache):
            async def async_increment_cache(self, key, **kwargs):
                if ":itpm:" in key:
                    raise RuntimeError("transient cache error")
                return await super().async_increment_cache(key=key, **kwargs)

        dual_cache = _ItpmFailCache()
        metadata = {ITPM_RESERVED_KEY: 5, ITPM_CACHE_KEY: "global_router:x:model:itpm:00-00"}
        kwargs = {"metadata": metadata}
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )

        with pytest.raises(RuntimeError):
            await async_io_token_reconcile_success(dual_cache, kwargs, response)

        # The stash is cleared even though reconciliation raised, so a duplicate
        # success event can't re-process it.
        assert ITPM_RESERVED_KEY not in metadata
        assert ITPM_CACHE_KEY not in metadata

    @pytest.mark.asyncio
    async def test_io_conflict_warning_not_collapsed_for_missing_model_id(self, caplog):
        import logging

        check = ModelRateLimitingCheck(dual_cache=DualCache())
        deployment = {
            "litellm_params": {"model": "openai/gpt-4o-mini", "itpm": 100, "tpm": 1000},
            "model_info": {},
        }
        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            check._warn_io_token_and_tpm_rpm_coexist_once(deployment)
            check._warn_io_token_and_tpm_rpm_coexist_once(deployment)

        warnings = [r for r in caplog.records if "both limit types are enforced" in r.message]
        # id-less deployments are not collapsed onto a single dedup key.
        assert len(warnings) == 2

    @pytest.mark.asyncio
    async def test_missing_deployment_id_skips_io_reservation(self):
        dual_cache = DualCache()
        deployment = {
            "litellm_params": {"model": "openai/gpt-4o-mini", "itpm": 100},
            "model_info": {},  # no id -> cannot build a per-deployment cache key
            "model_name": "opus",
        }
        request_kwargs = {
            "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 5,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(request_kwargs)

        result = await async_io_token_pre_call_check(dual_cache, deployment)

        assert result is deployment
        # No reservation is stashed, so nothing lands in a shared None:None bucket.
        assert ITPM_RESERVED_KEY not in request_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_reconcile_uses_reservation_minute_key(self):
        dual_cache = DualCache()
        # Reservation was made on a fixed minute key; a call that finishes in a
        # later minute must reconcile against that same key, never a key built
        # from the response-time minute.
        itpm_key = "global_router:io-min-id:bedrock_mantle/test:itpm:99-99"
        await dual_cache.async_increment_cache(key=itpm_key, value=10, ttl=60)

        kwargs = {"metadata": {ITPM_RESERVED_KEY: 10, ITPM_CACHE_KEY: itpm_key}}
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=4, completion_tokens=0, total_tokens=4),
        )
        await async_io_token_reconcile_success(dual_cache, kwargs, response)

        assert await dual_cache.async_get_cache(key=itpm_key) == 4

    @pytest.mark.asyncio
    async def test_reconcile_missing_usage_keeps_reservation(self):
        dual_cache = DualCache()
        itpm_key = "global_router:io-missing-usage:bedrock_mantle/test:itpm:00-00"
        otpm_key = "global_router:io-missing-usage:bedrock_mantle/test:otpm:00-00"
        await dual_cache.async_increment_cache(key=itpm_key, value=8, ttl=60)
        await dual_cache.async_increment_cache(key=otpm_key, value=5, ttl=60)

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 8,
                OTPM_RESERVED_KEY: 5,
                ITPM_CACHE_KEY: itpm_key,
                OTPM_CACHE_KEY: otpm_key,
            }
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
        )

        await async_io_token_reconcile_success(dual_cache, kwargs, response)

        assert await dual_cache.async_get_cache(key=itpm_key) == 8
        assert await dual_cache.async_get_cache(key=otpm_key) == 5
        assert ITPM_RESERVED_KEY not in kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_reconcile_total_tokens_only_keeps_reservation(self):
        """
        A response usage object with only total_tokens (no prompt/completion
        breakdown) can't be split into input/output, so it must be treated the
        same as missing usage: keep the reservation instead of resolving to
        (0, 0) and refunding it in full.
        """
        dual_cache = DualCache()
        itpm_key = "global_router:io-total-only:bedrock_mantle/test:itpm:00-00"
        otpm_key = "global_router:io-total-only:bedrock_mantle/test:otpm:00-00"
        await dual_cache.async_increment_cache(key=itpm_key, value=8, ttl=60)
        await dual_cache.async_increment_cache(key=otpm_key, value=5, ttl=60)

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 8,
                OTPM_RESERVED_KEY: 5,
                ITPM_CACHE_KEY: itpm_key,
                OTPM_CACHE_KEY: otpm_key,
            }
        }
        response = {"type": "message", "usage": {"total_tokens": 13}}

        await async_io_token_reconcile_success(dual_cache, kwargs, response)

        assert await dual_cache.async_get_cache(key=itpm_key) == 8
        assert await dual_cache.async_get_cache(key=otpm_key) == 5

    def test_reconcile_standard_logging_total_tokens_only_keeps_reservation(self):
        dual_cache = DualCache()
        itpm_key = "global_router:io-slo-total-only:bedrock_mantle/test:itpm:00-00"
        dual_cache.set_cache(key=itpm_key, value=10, ttl=60)

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 10, ITPM_CACHE_KEY: itpm_key},
            "standard_logging_object": {"total_tokens": 4},
        }
        response = {"type": "message", "role": "assistant", "content": []}

        io_token_reconcile_success(dual_cache, kwargs, response)

        assert dual_cache.get_cache(key=itpm_key) == 10

    @pytest.mark.asyncio
    async def test_reconcile_falls_back_to_standard_logging_object(self):
        dual_cache = DualCache()
        itpm_key = "global_router:io-slo-fallback:bedrock_mantle/test:itpm:00-00"
        await dual_cache.async_increment_cache(key=itpm_key, value=10, ttl=60)

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 10, ITPM_CACHE_KEY: itpm_key},
            "standard_logging_object": {
                "prompt_tokens": 4,
                "completion_tokens": 0,
                "total_tokens": 4,
            },
        }
        response = {"type": "message", "role": "assistant", "content": []}

        await async_io_token_reconcile_success(dual_cache, kwargs, response)

        assert await dual_cache.async_get_cache(key=itpm_key) == 4

    def test_sync_reconcile_anthropic_dict_usage(self):
        dual_cache = DualCache()
        itpm_key = "global_router:io-anthropic:bedrock_mantle/test:itpm:00-00"
        otpm_key = "global_router:io-anthropic:bedrock_mantle/test:otpm:00-00"
        dual_cache.set_cache(key=itpm_key, value=6, ttl=60)
        dual_cache.set_cache(key=otpm_key, value=4, ttl=60)

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 6,
                OTPM_RESERVED_KEY: 4,
                ITPM_CACHE_KEY: itpm_key,
                OTPM_CACHE_KEY: otpm_key,
            }
        }
        response = {
            "type": "message",
            "usage": {"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 1},
        }

        io_token_reconcile_success(dual_cache, kwargs, response)

        assert dual_cache.get_cache(key=itpm_key) == 2
        assert dual_cache.get_cache(key=otpm_key) == 2

    @pytest.mark.asyncio
    async def test_io_and_tpm_rpm_limits_both_enforced_with_warning(self, caplog):
        import logging

        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        model_id = "io-mixed-id"
        deployment_name = "bedrock_mantle/anthropic.claude-opus-4-7"
        deployment = {
            "litellm_params": {
                "model": deployment_name,
                "itpm": 100,
                "rpm": 1,
            },
            "model_info": {"id": model_id},
            "model_name": "opus",
        }

        minute = get_utc_datetime().strftime("%H-%M")
        rpm_key = f"{model_id}:{deployment_name}:rpm:{minute}"
        itpm_key = f"global_router:{model_id}:{deployment_name}:itpm:{minute}"
        await dual_cache.async_increment_cache(key=rpm_key, value=5, ttl=60)

        request_kwargs = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(request_kwargs)

        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            with pytest.raises(litellm.RateLimitError):
                await check.async_pre_call_check(deployment)

        assert await dual_cache.async_get_cache(key=rpm_key) == 6
        assert (await dual_cache.async_get_cache(key=itpm_key) or 0) == 0
        assert ITPM_RESERVED_KEY not in request_kwargs["metadata"]
        assert any("both limit types are enforced" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_io_success_still_tracks_tpm_for_mixed_deployment(self):
        """
        A deployment with itpm/otpm AND tpm/rpm must have BOTH counters updated on
        success, otherwise the tpm_key the pre-call check reads is never written
        and the tpm_limit can never be enforced.
        """
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        model_id = "io-tpm-mixed-id"
        deployment_name = "bedrock_mantle/anthropic.claude-opus-4-7"
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:{model_id}:{deployment_name}:itpm:{minute}"
        tpm_key = f"{model_id}:{deployment_name}:tpm:{minute}"
        await dual_cache.async_increment_cache(key=itpm_key, value=5, ttl=60)

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 5, ITPM_CACHE_KEY: itpm_key},
            "standard_logging_object": {
                "model_id": model_id,
                "total_tokens": 7,
                "hidden_params": {"litellm_model_name": deployment_name},
            },
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )

        await check.async_log_success_event(kwargs, response, None, None)

        # ITPM reconciled down from the 5-token reservation to actual usage (3).
        assert await dual_cache.async_get_cache(key=itpm_key) == 3
        # TPM tracking must still run so the tpm/rpm pre-call path can enforce it.
        assert await dual_cache.async_get_cache(key=tpm_key) == 7

    def test_io_success_still_tracks_tpm_for_mixed_deployment_sync(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        model_id = "io-tpm-mixed-sync-id"
        deployment_name = "bedrock_mantle/anthropic.claude-opus-4-7"
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:{model_id}:{deployment_name}:itpm:{minute}"
        tpm_key = f"{model_id}:{deployment_name}:tpm:{minute}"
        dual_cache.set_cache(key=itpm_key, value=5, ttl=60)

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 5, ITPM_CACHE_KEY: itpm_key},
            "standard_logging_object": {
                "model_id": model_id,
                "total_tokens": 7,
                "hidden_params": {"litellm_model_name": deployment_name},
            },
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )

        check.log_success_event(kwargs, response, None, None)

        assert dual_cache.get_cache(key=itpm_key) == 3
        assert dual_cache.get_cache(key=tpm_key) == 7

    @pytest.mark.asyncio
    async def test_failure_refunds_itpm_reservation(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-refund-id:bedrock_mantle/test:itpm:{minute}"
        await dual_cache.async_increment_cache(key=itpm_key, value=20, ttl=60)

        reservation = {ITPM_RESERVED_KEY: 20, ITPM_CACHE_KEY: itpm_key}
        kwargs = {
            "standard_logging_object": {
                "model_id": "io-refund-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": dict(reservation),
            },
            "metadata": dict(reservation),
        }
        await check.async_log_failure_event(kwargs, None, None, None)

        current = await dual_cache.async_get_cache(key=itpm_key)
        assert current == 0


class TestRouterIOTokenIntegration:
    @pytest.mark.asyncio
    async def test_model_group_info_aggregates_io_limits(self):
        router = Router(
            model_list=[
                {
                    "model_name": "opus",
                    "litellm_params": {
                        "model": "bedrock_mantle/anthropic.claude-opus-4-7",
                        "itpm": 100,
                        "otpm": 20,
                    },
                }
            ],
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )
        info = router.get_model_group_info("opus")
        assert info is not None
        assert info.itpm == 100
        assert info.otpm == 20
