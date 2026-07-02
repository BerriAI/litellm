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
    async_io_token_reconcile_success,
    build_io_token_rate_limit_headers,
    deployment_has_io_token_limits,
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

        # No messages/prompt -> pre-call estimate is 0, so nothing is reserved,
        # but the actual billable input must still be tracked post-call.
        request_kwargs = {"max_tokens": 5, "metadata": {}}
        set_io_token_rate_limit_request_kwargs(request_kwargs)
        await check.async_pre_call_check(deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-zero-est-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        assert (await dual_cache.async_get_cache(key=itpm_key) or 0) == 0

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
    async def test_io_limits_supersede_tpm_rpm_with_warning(self, caplog):
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

        # rpm counter is already over the rpm=1 limit; the legacy path would
        # atomically increment and raise, the io path must never touch it.
        minute = get_utc_datetime().strftime("%H-%M")
        rpm_key = f"{model_id}:{deployment_name}:rpm:{minute}"
        await dual_cache.async_increment_cache(key=rpm_key, value=5, ttl=60)

        set_io_token_rate_limit_request_kwargs(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "metadata": {},
            }
        )

        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            result = await check.async_pre_call_check(deployment)

        # io path taken, rpm limit not enforced.
        assert result is deployment
        assert await dual_cache.async_get_cache(key=rpm_key) == 5
        # the conflict is surfaced, not silent.
        assert any("take precedence" in record.message for record in caplog.records)

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
