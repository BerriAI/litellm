"""
Tests for separate ITPM/OTPM deployment rate limits (enforce_model_rate_limits).
"""

import asyncio

import pytest

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    ITPM_RESERVED_KEY,
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
            "metadata": {ITPM_RESERVED_KEY: 20},
            "litellm_params": {"metadata": {"user_api_key_hash": "abc123"}},
        }
        await check.async_log_failure_event(kwargs, None, None, None)

        current = await dual_cache.async_get_cache(key=itpm_key)
        assert current == 0

    @pytest.mark.asyncio
    async def test_failure_refunds_itpm_reservation(self):
        from litellm.utils import get_utc_datetime

        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-refund-id:bedrock_mantle/test:itpm:{minute}"
        await dual_cache.async_increment_cache(key=itpm_key, value=20, ttl=60)

        kwargs = {
            "standard_logging_object": {
                "model_id": "io-refund-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {ITPM_RESERVED_KEY: 20},
            },
            "metadata": {ITPM_RESERVED_KEY: 20},
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
