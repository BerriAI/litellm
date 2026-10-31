"""
Tests for separate ITPM/OTPM deployment rate limits (enforce_model_rate_limits).
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeRedis, FakeServer
from fakeredis.aioredis import FakeRedis as FakeAsyncRedis

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.constants import NO_MAX_TOKENS_TPM_RESERVATION_FRACTION
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    ITPM_CACHE_KEY,
    ITPM_RESERVED_KEY,
    OTPM_CACHE_KEY,
    OTPM_RESERVED_KEY,
    _explicit_max_tokens,
    _reservation_value,
    async_admit_io_token_turn,
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
from litellm.types.llms.openai import InputTokensDetails, ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import RouterCacheEnum
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import get_utc_datetime

MANTLE_OPUS = "bedrock_mantle/anthropic.claude-opus-4-7"
MANTLE_OPUS_MAX_OUTPUT = litellm.get_model_info(model=MANTLE_OPUS)["max_output_tokens"]
UNMAPPED_MODEL = "openai/litellm-io-unmapped-model"
UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT = 4096
AZURE_BASE_MODEL = "azure/gpt-5"
AZURE_BASE_MODEL_MAX_OUTPUT = litellm.get_model_info(model=AZURE_BASE_MODEL)["max_output_tokens"]


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

    @pytest.mark.parametrize(
        "request_kwargs, expected",
        [
            ({"max_tokens": 0}, 0),
            ({"max_completion_tokens": 12}, 12),
            ({"max_output_tokens": 9}, 9),
            ({"max_tokens": 5, "max_completion_tokens": 12}, 5),
            ({"max_tokens": None, "max_output_tokens": 9}, 9),
            ({"max_tokens": [100]}, None),
            ({"messages": []}, None),
            (None, None),
        ],
    )
    def test_explicit_max_tokens_reads_every_output_cap_key(
        self, request_kwargs: dict[str, object] | None, expected: int | None
    ):
        assert _explicit_max_tokens(request_kwargs) == expected

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
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
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
            },
            metadata_channel="metadata",
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
            },
            metadata_channel="metadata",
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
            },
            metadata_channel="metadata",
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
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
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
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
        await check.async_pre_call_check(deployment)

        # Second zero-estimate call still fits within the itpm=2 limit.
        set_io_token_rate_limit_request_kwargs({"max_tokens": 5, "metadata": {}}, metadata_channel="metadata")
        await check.async_pre_call_check(deployment)

        # A third exceeds the limit and is rejected.
        set_io_token_rate_limit_request_kwargs({"max_tokens": 5, "metadata": {}}, metadata_channel="metadata")
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
        set_io_token_rate_limit_request_kwargs(zero_output_kwargs, metadata_channel="metadata")
        await check.async_pre_call_check(deployment)

        zero_output_otpm_key = zero_output_kwargs["metadata"][OTPM_CACHE_KEY]
        assert zero_output_kwargs["metadata"][OTPM_RESERVED_KEY] == 0
        assert (await dual_cache.async_get_cache(key=zero_output_otpm_key) or 0) == 0

        normal_output_kwargs = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(normal_output_kwargs, metadata_channel="metadata")
        await check.async_pre_call_check(deployment)

        normal_output_otpm_key = normal_output_kwargs["metadata"][OTPM_CACHE_KEY]
        assert await dual_cache.async_get_cache(key=normal_output_otpm_key) == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_check", [False, True])
    @pytest.mark.parametrize(
        "model, otpm_limit",
        [
            (MANTLE_OPUS, MANTLE_OPUS_MAX_OUTPUT // 2),
            (MANTLE_OPUS, MANTLE_OPUS_MAX_OUTPUT - 1),
            (UNMAPPED_MODEL, 1000),
            (UNMAPPED_MODEL, 3),
        ],
    )
    async def test_request_without_max_tokens_is_admitted_below_model_max_output(
        self, model: str, otpm_limit: int, use_sync_check: bool
    ):
        dual_cache: Final = DualCache()
        check: Final = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment: Final = {
            "litellm_params": {"model": model, "otpm": otpm_limit},
            "model_info": {"id": f"io-no-max-tokens-{otpm_limit}"},
            "model_name": "no-max-tokens",
        }
        request_kwargs: Final = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")

        if use_sync_check:
            check.pre_call_check(deployment)
        else:
            await check.async_pre_call_check(deployment)

        reserved: Final = request_kwargs["metadata"][OTPM_RESERVED_KEY]
        assert isinstance(reserved, int)
        assert 0 < reserved < otpm_limit
        assert await dual_cache.async_get_cache(key=request_kwargs["metadata"][OTPM_CACHE_KEY]) == reserved

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, otpm_limit",
        [(MANTLE_OPUS, MANTLE_OPUS_MAX_OUTPUT), (UNMAPPED_MODEL, UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT)],
    )
    async def test_concurrent_requests_without_max_tokens_share_otpm_without_overshoot(
        self, model: str, otpm_limit: int
    ):
        dual_cache: Final = DualCache()
        check: Final = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment: Final = {
            "litellm_params": {"model": model, "otpm": otpm_limit},
            "model_info": {"id": "io-no-max-tokens-race-id"},
            "model_name": "no-max-tokens",
        }

        async def _attempt() -> dict[str, object] | None:
            request_kwargs: Final = {"messages": [{"role": "user", "content": "hi"}], "metadata": {}}
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            try:
                await check.async_pre_call_check(deployment)
            except litellm.RateLimitError:
                return None
            return request_kwargs["metadata"]

        admitted: Final = [m for m in await asyncio.gather(*[_attempt() for _ in range(8)]) if m is not None]
        otpm_key: Final = admitted[0][OTPM_CACHE_KEY]

        assert len(admitted) == round(1 / NO_MAX_TOKENS_TPM_RESERVATION_FRACTION)
        assert {m[OTPM_CACHE_KEY] for m in admitted} == {otpm_key}
        assert await dual_cache.async_get_cache(key=otpm_key) == sum(m[OTPM_RESERVED_KEY] for m in admitted)
        assert await dual_cache.async_get_cache(key=otpm_key) <= otpm_limit

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, model_max_output",
        [(MANTLE_OPUS, MANTLE_OPUS_MAX_OUTPUT), (UNMAPPED_MODEL, UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT)],
    )
    async def test_request_without_max_tokens_reserves_model_max_output_when_otpm_allows(
        self, model: str, model_max_output: int
    ):
        dual_cache: Final = DualCache()
        check: Final = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment: Final = {
            "litellm_params": {"model": model, "otpm": model_max_output * 10},
            "model_info": {"id": "io-no-max-tokens-roomy-id"},
            "model_name": "roomy",
        }
        request_kwargs: Final = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")

        await check.async_pre_call_check(deployment)

        assert request_kwargs["metadata"][OTPM_RESERVED_KEY] == model_max_output

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_check", [False, True])
    async def test_base_model_does_not_change_the_default_otpm_reservation(self, use_sync_check: bool):
        check: Final = ModelRateLimitingCheck(dual_cache=DualCache())
        otpm_limit: Final = AZURE_BASE_MODEL_MAX_OUTPUT * 10

        async def _reserved(model_info: dict[str, str]) -> int:
            request_kwargs: Final = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            deployment: Final = {
                "litellm_params": {"model": "azure/litellm-io-custom-deployment", "otpm": otpm_limit},
                "model_info": model_info,
                "model_name": "custom-name",
            }
            if use_sync_check:
                check.pre_call_check(deployment)
            else:
                await check.async_pre_call_check(deployment)
            return request_kwargs["metadata"][OTPM_RESERVED_KEY]

        with_base_model: Final = await _reserved({"id": "io-with-base-model", "base_model": AZURE_BASE_MODEL})
        without_base_model: Final = await _reserved({"id": "io-without-base-model"})

        assert with_base_model == without_base_model

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_check", [False, True])
    async def test_unparseable_max_tokens_leaves_itpm_uncharged(self, use_sync_check: bool):
        dual_cache: Final = DualCache()
        check: Final = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment: Final = {
            "litellm_params": {"model": MANTLE_OPUS, "itpm": 2000, "otpm": 2000},
            "model_info": {"id": "io-bad-max-tokens-id"},
            "model_name": "opus",
        }

        async def _pre_call(request_kwargs: dict[str, object]) -> None:
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            if use_sync_check:
                check.pre_call_check(deployment)
            else:
                await check.async_pre_call_check(deployment)

        await _pre_call({"messages": [{"role": "user", "content": "hello there"}], "max_tokens": "abc", "metadata": {}})
        valid_kwargs: Final = {
            "messages": [{"role": "user", "content": "hello there"}],
            "max_tokens": 1,
            "metadata": {},
        }
        await _pre_call(valid_kwargs)

        itpm_counter: Final = await dual_cache.async_get_cache(key=valid_kwargs["metadata"][ITPM_CACHE_KEY])
        assert itpm_counter == valid_kwargs["metadata"][ITPM_RESERVED_KEY]

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_above_default_share_is_reserved_in_full(self):
        dual_cache: Final = DualCache()
        check: Final = ModelRateLimitingCheck(dual_cache=dual_cache)
        deployment: Final = {
            "litellm_params": {"model": MANTLE_OPUS, "otpm": 100},
            "model_info": {"id": "io-explicit-max-tokens-id"},
            "model_name": "opus",
        }
        request_kwargs: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100, "metadata": {}}
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")

        await check.async_pre_call_check(deployment)

        assert request_kwargs["metadata"][OTPM_RESERVED_KEY] == 100
        set_io_token_rate_limit_request_kwargs(
            {"messages": [{"role": "user", "content": "hi"}], "metadata": {}}, metadata_channel="metadata"
        )
        with pytest.raises(litellm.RateLimitError):
            await check.async_pre_call_check(deployment)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_check", [False, True])
    async def test_deployment_max_tokens_sets_the_reservation_unless_the_request_sets_one(self, use_sync_check: bool):
        check: Final = ModelRateLimitingCheck(dual_cache=DualCache())
        deployment: Final = {
            "litellm_params": {"model": MANTLE_OPUS, "otpm": 4000, "max_tokens": 100},
            "model_info": {"id": "io-deployment-max-tokens-id"},
            "model_name": "opus",
        }
        without_cap: Final = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
        with_cap: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 300, "metadata": {}}

        for request_kwargs in (without_cap, with_cap):
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            if use_sync_check:
                check.pre_call_check(deployment)
            else:
                await check.async_pre_call_check(deployment)

        assert without_cap["metadata"][OTPM_RESERVED_KEY] == 100
        assert with_cap["metadata"][OTPM_RESERVED_KEY] == 300

    @pytest.mark.asyncio
    async def test_deployment_cap_sets_the_reservation_only_through_the_param_its_api_sends(self):
        check: Final = ModelRateLimitingCheck(dual_cache=DualCache())
        chat: Final = {"messages": [{"role": "user", "content": "hello"}]}
        responses: Final = {"input": "hello"}

        async def reserved(deployment_cap: Mapping[str, int], request_shape: Mapping[str, object]) -> int:
            request_kwargs: Final = {**request_shape, "metadata": {}}
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            await check.async_pre_call_check(
                {
                    "litellm_params": {"model": MANTLE_OPUS, "otpm": 4000, **deployment_cap},
                    "model_info": {"id": f"io-sent-cap-{sorted(deployment_cap)}-{sorted(request_shape)}"},
                    "model_name": "opus",
                }
            )
            return request_kwargs["metadata"][OTPM_RESERVED_KEY]

        chat_default: Final = await reserved({}, chat)
        responses_default: Final = await reserved({}, responses)

        assert chat_default == responses_default > 100
        assert await reserved({"max_tokens": 100}, chat) == 100
        assert await reserved({"max_completion_tokens": 100}, chat) == 100
        assert await reserved({"max_output_tokens": 100}, chat) == chat_default
        assert await reserved({"max_output_tokens": 100}, responses) == 100
        assert await reserved({"max_tokens": 100}, responses) == responses_default
        assert await reserved({"max_completion_tokens": 100}, responses) == responses_default

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_check", [False, True])
    async def test_deployment_cap_above_the_default_reserves_the_default(self, use_sync_check: bool):
        check: Final = ModelRateLimitingCheck(dual_cache=DualCache())
        otpm_limit: Final = UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT * 5

        async def reserved(deployment_cap: Mapping[str, int]) -> int:
            request_kwargs: Final = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            deployment: Final = {
                "litellm_params": {"model": UNMAPPED_MODEL, "otpm": otpm_limit, **deployment_cap},
                "model_info": {"id": f"io-deployment-cap-above-default-{deployment_cap}"},
                "model_name": "unmapped",
            }
            if use_sync_check:
                check.pre_call_check(deployment)
            else:
                await check.async_pre_call_check(deployment)
            return request_kwargs["metadata"][OTPM_RESERVED_KEY]

        default: Final = await reserved({})

        assert await reserved({"max_tokens": default + 1}) == default
        assert await reserved({"max_tokens": otpm_limit * 2}) == default

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
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
        check.pre_call_check(deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = f"global_router:io-sync-id:bedrock_mantle/anthropic.claude-opus-4-7:itpm:{minute}"
        otpm_key = f"global_router:io-sync-id:bedrock_mantle/anthropic.claude-opus-4-7:otpm:{minute}"
        kwargs = {
            "standard_logging_object": {
                "model_id": "io-sync-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/anthropic.claude-opus-4-7"},
                "metadata": {},
            },
            "litellm_params": {"metadata": request_kwargs["metadata"]},
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
        set_io_token_rate_limit_request_kwargs(shared_kwargs, metadata_channel="metadata")
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
        set_io_token_rate_limit_request_kwargs(forged, metadata_channel="metadata")
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
        set_io_token_rate_limit_request_kwargs(kwargs, metadata_channel="metadata")
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
            },
            metadata_channel="metadata",
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
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")

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
    @pytest.mark.parametrize("use_sync_reconcile", [False, True])
    async def test_reconcile_charges_usage_above_the_reservation_to_the_minute_it_finished_in(
        self, use_sync_reconcile: bool
    ):
        dual_cache: Final = DualCache()
        counter: Final = "global_router:io-late-id:bedrock_mantle/test:otpm"
        reserved_minute_key: Final = f"{counter}:99-99"
        await dual_cache.async_increment_cache(key=reserved_minute_key, value=10, ttl=60)
        kwargs: Final = {"metadata": {ITPM_RESERVED_KEY: 0, OTPM_RESERVED_KEY: 10, OTPM_CACHE_KEY: reserved_minute_key}}
        response: Final = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=4, completion_tokens=25, total_tokens=29),
        )
        minute_before: Final = get_utc_datetime().strftime("%H-%M")

        if use_sync_reconcile:
            io_token_reconcile_success(dual_cache, kwargs, response)
        else:
            await async_io_token_reconcile_success(dual_cache, kwargs, response)

        finished_minutes: Final = {minute_before, get_utc_datetime().strftime("%H-%M")}
        charged: Final = [await dual_cache.async_get_cache(key=f"{counter}:{minute}") for minute in finished_minutes]
        assert await dual_cache.async_get_cache(key=reserved_minute_key) == 10
        assert sum(value or 0 for value in charged) == 25 - 10

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

    def test_sync_reconcile_anthropic_dict_usage_counts_uncached_input_once(self):
        dual_cache = DualCache()
        itpm_key = "global_router:io-anthropic:bedrock_mantle/test:itpm:00-00"
        otpm_key = "global_router:io-anthropic:bedrock_mantle/test:otpm:00-00"
        dual_cache.set_cache(key=itpm_key, value=10, ttl=60)
        dual_cache.set_cache(key=otpm_key, value=4, ttl=60)

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 10,
                OTPM_RESERVED_KEY: 4,
                ITPM_CACHE_KEY: itpm_key,
                OTPM_CACHE_KEY: otpm_key,
            }
        }
        response = {
            "type": "message",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "cache_read_input_tokens": 1,
                "cache_creation_input_tokens": 4,
            },
        }

        io_token_reconcile_success(dual_cache, kwargs, response)

        assert dual_cache.get_cache(key=itpm_key) == 3 + 4
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
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")

        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            with pytest.raises(litellm.RateLimitError):
                await check.async_pre_call_check(deployment)

        assert await dual_cache.async_get_cache(key=rpm_key) == 5, "the rejected request hands its rpm count back"
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


ROUTER_IO_MODEL: Final = "anthropic/claude-haiku-5-5"
RouterCall = Callable[[Router, str | Exception], Awaitable[tuple[int, int]]]


def _router_with_io_limits() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "limited",
                "litellm_params": {"model": ROUTER_IO_MODEL, "api_key": "sk-test", "itpm": 50000, "otpm": 50000},
            }
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=1,
    )


def _current_minute() -> str:
    return get_utc_datetime().strftime("%H-%M")


def _counter(router: Router, template: RouterCacheEnum, minutes: frozenset[str]) -> int:
    deployment: Final = router.model_list[0]
    return sum(
        router.cache.get_cache(
            key=template.value.format(
                id=deployment["model_info"]["id"],
                model=deployment["litellm_params"]["model"],
                current_minute=minute,
            )
        )
        or 0
        for minute in minutes
    )


async def _io_counters_since(router: Router, first_minute: str) -> tuple[int, int]:
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    minutes: Final = frozenset({first_minute, _current_minute()})
    return _counter(router, RouterCacheEnum.ITPM, minutes), _counter(router, RouterCacheEnum.OTPM, minutes)


async def _sdk_chat(router: Router, mock_response: str | Exception) -> tuple[int, int]:
    response: Final = await router.acompletion(
        model="limited",
        messages=[{"role": "user", "content": "hello there"}],
        max_tokens=100,
        mock_response=mock_response,
    )
    return response.usage.prompt_tokens, response.usage.completion_tokens


async def _proxy_chat(router: Router, mock_response: str | Exception) -> tuple[int, int]:
    response: Final = await router.acompletion(
        model="limited",
        messages=[{"role": "user", "content": "hello there"}],
        max_tokens=100,
        metadata={"user_api_key_hash": "hashed-key"},
        mock_response=mock_response,
    )
    return response.usage.prompt_tokens, response.usage.completion_tokens


async def _proxy_responses(router: Router, mock_response: str | Exception) -> tuple[int, int]:
    response: Final = await router.aresponses(
        model="limited",
        input="hello there",
        max_output_tokens=100,
        litellm_metadata={"user_api_key_hash": "hashed-key"},
        mock_response=mock_response,
    )
    return response.usage.input_tokens, response.usage.output_tokens


async def _proxy_messages(router: Router, mock_response: str | Exception) -> tuple[int, int]:
    response: Final = await router.aanthropic_messages(
        model="limited",
        messages=[{"role": "user", "content": "hello there"}],
        max_tokens=100,
        litellm_metadata={"user_api_key_hash": "hashed-key"},
        mock_response=mock_response,
    )
    return response["usage"]["input_tokens"], response["usage"]["output_tokens"]


ProxyCallSharingLogging = Callable[[Router, str | Exception, LiteLLMLoggingObject], Awaitable[object]]
PROXY_MESSAGES: Final = [{"role": "user", "content": "hello there"}]


def _logging_obj_shared_across_attempts(original_function: str) -> LiteLLMLoggingObject:
    logging_obj, _ = litellm.utils.function_setup(
        original_function,
        litellm.utils.Rules(),
        datetime.now(),
        model="limited",
        messages=PROXY_MESSAGES,
        litellm_call_id="proxy-call",
    )
    return logging_obj


async def _proxy_chat_sharing_logging(
    router: Router, mock_error: str | Exception, logging_obj: LiteLLMLoggingObject
) -> object:
    return await router.acompletion(
        model="limited",
        messages=PROXY_MESSAGES,
        max_tokens=100,
        metadata={"user_api_key_hash": "hashed-key"},
        mock_response=mock_error,
        litellm_call_id="proxy-call",
        litellm_logging_obj=logging_obj,
    )


async def _proxy_responses_sharing_logging(
    router: Router, mock_error: str | Exception, logging_obj: LiteLLMLoggingObject
) -> object:
    return await router.aresponses(
        model="limited",
        input="hello there",
        max_output_tokens=100,
        litellm_metadata={"user_api_key_hash": "hashed-key"},
        mock_response=mock_error,
        litellm_call_id="proxy-call",
        litellm_logging_obj=logging_obj,
    )


async def _proxy_messages_sharing_logging(
    router: Router, mock_error: str | Exception, logging_obj: LiteLLMLoggingObject
) -> object:
    return await router.aanthropic_messages(
        model="limited",
        messages=PROXY_MESSAGES,
        max_tokens=100,
        litellm_metadata={"user_api_key_hash": "hashed-key"},
        mock_response=mock_error,
        litellm_call_id="proxy-call",
        litellm_logging_obj=logging_obj,
    )


class TestRouterIOTokenIntegration:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call",
        [_sdk_chat, _proxy_chat, _proxy_responses, _proxy_messages],
        ids=["sdk_chat", "proxy_chat_metadata", "proxy_responses_litellm_metadata", "proxy_messages_litellm_metadata"],
    )
    async def test_success_reconciles_io_counters_to_real_usage(self, call: RouterCall):
        router: Final = _router_with_io_limits()
        first_minute: Final = _current_minute()

        usage: Final = await call(router, "hi")

        assert await _io_counters_since(router, first_minute) == usage

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call", "mock_error"),
        [
            (
                _proxy_responses,
                litellm.InternalServerError(message="mock 500", llm_provider="anthropic", model=ROUTER_IO_MODEL),
            ),
            (_proxy_messages, "litellm.InternalServerError"),
        ],
        ids=["proxy_responses_litellm_metadata", "proxy_messages_litellm_metadata"],
    )
    async def test_retried_failure_refunds_each_io_reservation_once(
        self, call: RouterCall, mock_error: str | Exception
    ):
        router: Final = _router_with_io_limits()
        first_minute: Final = _current_minute()

        with pytest.raises(litellm.InternalServerError):
            await call(router, mock_error)

        assert await _io_counters_since(router, first_minute) == (0, 0)


    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call", "original_function", "mock_error"),
        [
            (_proxy_chat_sharing_logging, "acompletion", "litellm.InternalServerError"),
            (
                _proxy_responses_sharing_logging,
                "aresponses",
                litellm.InternalServerError(message="mock 500", llm_provider="anthropic", model=ROUTER_IO_MODEL),
            ),
            (_proxy_messages_sharing_logging, "anthropic_messages", "litellm.InternalServerError"),
        ],
        ids=["proxy_chat", "proxy_responses", "proxy_messages"],
    )
    async def test_retried_failure_sharing_one_logging_object_refunds_every_attempt(
        self, call: ProxyCallSharingLogging, original_function: str, mock_error: str | Exception
    ):
        router: Final = _router_with_io_limits()
        first_minute: Final = _current_minute()

        with pytest.raises(litellm.InternalServerError):
            await call(router, mock_error, _logging_obj_shared_across_attempts(original_function))

        assert await _io_counters_since(router, first_minute) == (0, 0)

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


class TestContextSlotRetention:
    def test_setter_stores_kwargs_only_for_io_limited_deployments(self):
        """
        The context slot pins the entire request kwargs (messages included)
        for the lifetime of the surrounding asyncio context, and pooled
        resources created mid-request (e.g. redis connections) capture that
        context, extending the pin far past the request. Only ITPM/OTPM
        pre-call checks read the slot, so the setter must store None for
        deployments without io token limits and still clear reservation
        sentinels from kwargs either way.
        """
        kwargs = {
            "messages": [{"role": "user", "content": "x" * 1000}],
            "metadata": {ITPM_RESERVED_KEY: 999, ITPM_CACHE_KEY: "forged"},
        }
        set_io_token_rate_limit_request_kwargs(kwargs, store_in_context=False, metadata_channel="metadata")
        assert get_io_token_rate_limit_request_kwargs() is None
        assert ITPM_RESERVED_KEY not in kwargs["metadata"]
        assert ITPM_CACHE_KEY not in kwargs["metadata"]

        set_io_token_rate_limit_request_kwargs(kwargs, store_in_context=True, metadata_channel="metadata")
        assert get_io_token_rate_limit_request_kwargs() is kwargs

        set_io_token_rate_limit_request_kwargs(kwargs, store_in_context=False, metadata_channel="metadata")
        assert get_io_token_rate_limit_request_kwargs() is None

    @pytest.mark.asyncio
    async def test_router_does_not_pin_kwargs_without_io_limits(self):
        router = Router(
            model_list=[
                {
                    "model_name": "plain",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                }
            ]
        )
        set_io_token_rate_limit_request_kwargs(None, metadata_channel="metadata")
        kwargs = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
        deployment = router.get_deployment_by_model_group_name("plain")
        assert deployment is not None
        router._update_kwargs_with_deployment(deployment=deployment.model_dump(), kwargs=kwargs)
        assert get_io_token_rate_limit_request_kwargs() is None

    @pytest.mark.asyncio
    async def test_router_pins_kwargs_for_io_limited_deployment(self):
        router = Router(
            model_list=[
                {
                    "model_name": "limited",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test", "itpm": 100},
                }
            ],
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )
        set_io_token_rate_limit_request_kwargs(None, metadata_channel="metadata")
        kwargs = {"messages": [{"role": "user", "content": "hello"}], "metadata": {}}
        deployment = router.get_deployment_by_model_group_name("limited")
        assert deployment is not None
        router._update_kwargs_with_deployment(deployment=deployment.model_dump(), kwargs=kwargs)
        assert get_io_token_rate_limit_request_kwargs() is kwargs


@pytest.mark.asyncio
async def test_custom_azure_deployment_name_reserves_itpm_on_its_base_models_token_count() -> None:
    async def reserved_input(deployment: dict[str, object]) -> object:
        metadata: Final[dict[str, object]] = {}
        request_kwargs: Final[dict[str, object]] = {
            "messages": [{"role": "user", "content": "日本語のテキストとहिन्दी पाठ और русский текст. " * 20}],
            "max_tokens": 5,
            "metadata": metadata,
        }
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
        await async_io_token_pre_call_check(DualCache(), deployment)
        return metadata[ITPM_RESERVED_KEY]

    named = await reserved_input(
        {
            "litellm_params": {"model": "azure/my-gpt-6", "base_model": "azure/gpt-6.1-sol", "itpm": 100_000},
            "model_info": {"id": "named-deployment"},
            "model_name": "named",
        }
    )
    served = await reserved_input(
        {
            "litellm_params": {"model": "azure/gpt-6.1-sol", "itpm": 100_000},
            "model_info": {"id": "served-deployment"},
            "model_name": "served",
        }
    )

    assert named == served


def _reply_using(prompt_tokens: int, completion_tokens: int) -> ModelResponse:
    return ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _proxy_success_kwargs(reservation: Mapping[str, object]) -> dict[str, object]:
    return {
        "litellm_params": {"metadata": {**reservation, "user_api_key_hash": "hashed-key"}},
        "standard_logging_object": {
            "hidden_params": {"litellm_model_name": "openai/gpt-5-mini"},
            "metadata": {"user_api_key_hash": "hashed-key"},
        },
    }


async def _deliver_success(
    check: ModelRateLimitingCheck, delivery: str, kwargs: dict[str, object], reply: object
) -> None:
    if delivery == "async":
        await check.async_log_success_event(kwargs, reply, None, None)
        return
    check.log_success_event(kwargs, reply, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deliveries",
    [("async",), ("sync",), ("async", "async"), ("async", "sync"), ("sync", "async"), ("sync", "sync")],
)
async def test_success_settles_a_reservation_held_only_in_litellm_params_metadata_exactly_once(
    deliveries: tuple[str, ...],
) -> None:
    dual_cache: Final = DualCache()
    check: Final = ModelRateLimitingCheck(dual_cache=dual_cache)
    itpm_key: Final = "global_router:proxy-shape:openai/gpt-5-mini:itpm:12-34"
    otpm_key: Final = "global_router:proxy-shape:openai/gpt-5-mini:otpm:12-34"
    await dual_cache.async_increment_cache(key=itpm_key, value=40, ttl=60)
    await dual_cache.async_increment_cache(key=otpm_key, value=4096, ttl=60)
    kwargs: Final = _proxy_success_kwargs(
        {ITPM_RESERVED_KEY: 40, OTPM_RESERVED_KEY: 4096, ITPM_CACHE_KEY: itpm_key, OTPM_CACHE_KEY: otpm_key}
    )

    for delivery in deliveries:
        await _deliver_success(check, delivery, kwargs, _reply_using(prompt_tokens=7, completion_tokens=3))

    assert await dual_cache.async_get_cache(key=itpm_key) == 7
    assert await dual_cache.async_get_cache(key=otpm_key) == 3


_TURN_AT: Final = datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
_ACTUAL_TOKENS: Final = {"itpm": 5, "otpm": 2}


def _turn_deployment(kind: str) -> dict[str, object]:
    return {"litellm_params": {"model": "openai/gpt-5-mini", kind: 100}, "model_info": {"id": f"ws-turn-{kind}"}}


def _turn_key(kind: str, minute: str = "12-34") -> str:
    return f"global_router:ws-turn-{kind}:openai/gpt-5-mini:{kind}:{minute}"


@pytest.fixture
def settled_on_the_turn_s_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "litellm.router_utils.pre_call_checks.io_token_rate_limit_check.get_utc_datetime", lambda: _TURN_AT
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("used", [100, 101, 250])
async def test_turn_admission_refuses_once_the_minutes_tokens_reach_the_limit(kind: str, used: int) -> None:
    dual_cache: Final = DualCache()
    await dual_cache.async_increment_cache(key=_turn_key(kind), value=used, ttl=60)

    with pytest.raises(litellm.RateLimitError, match=f"{kind.upper()} limit=100, current usage={used}"):
        await async_admit_io_token_turn(dual_cache, _turn_deployment(kind), _TURN_AT)


@pytest.mark.asyncio
@pytest.mark.usefixtures("settled_on_the_turn_s_minute")
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("used", [0, 1, 99])
async def test_an_admitted_turn_reserves_nothing_and_counts_its_actual_tokens(kind: str, used: int) -> None:
    dual_cache: Final = DualCache()
    await dual_cache.async_increment_cache(key=_turn_key(kind), value=used, ttl=60)
    actual: Final = _ACTUAL_TOKENS

    reservation: Final = await async_admit_io_token_turn(dual_cache, _turn_deployment(kind), _TURN_AT)
    counter_key: Final = reservation[ITPM_CACHE_KEY if kind == "itpm" else OTPM_CACHE_KEY]
    assert isinstance(counter_key, str)
    assert await dual_cache.async_get_cache(key=counter_key) == used
    await async_io_token_reconcile_success(
        dual_cache, _proxy_success_kwargs(reservation), _reply_using(actual["itpm"], actual["otpm"])
    )

    assert await dual_cache.async_get_cache(key=counter_key) == used + actual[kind]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    [
        {"litellm_params": {"model": "openai/gpt-5-mini", "tpm": 1}, "model_info": {"id": "ws-turn-unlimited"}},
        {"litellm_params": {"model": "openai/gpt-5-mini", "otpm": 1}},
    ],
)
async def test_a_turn_without_io_token_counters_is_admitted_with_nothing_to_settle(
    deployment: dict[str, object],
) -> None:
    assert await async_admit_io_token_turn(DualCache(), deployment, _TURN_AT) == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("settled_on_the_turn_s_minute")
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("used_by_others", [0, 1, 99])
async def test_a_socket_open_reserves_nothing_and_its_first_turn_counts_its_actual_tokens_on_its_own_minute(
    kind: str, used_by_others: int
) -> None:
    dual_cache: Final = DualCache()
    await dual_cache.async_increment_cache(key=_turn_key(kind), value=used_by_others, ttl=60)
    socket_open: Final[dict[str, object]] = {"metadata": {}}
    set_io_token_rate_limit_request_kwargs(socket_open, opens_responses_socket=True, metadata_channel="metadata")

    assert await async_io_token_pre_call_check(dual_cache, _turn_deployment(kind)) == _turn_deployment(kind)
    assert await dual_cache.async_get_cache(key=_turn_key(kind)) == used_by_others

    opened_with: Final = socket_open["metadata"]
    assert isinstance(opened_with, dict)
    await async_io_token_reconcile_success(
        dual_cache, _proxy_success_kwargs(opened_with), _reply_using(_ACTUAL_TOKENS["itpm"], _ACTUAL_TOKENS["otpm"])
    )

    assert await dual_cache.async_get_cache(key=_turn_key(kind)) == used_by_others + _ACTUAL_TOKENS[kind]


class _RedisCacheOnFakeServer(RedisCache):
    def __init__(self, server: FakeServer) -> None:
        super().__init__(host="io-token-turn-redis", connection_pool=FakeRedis(server=server).connection_pool)
        self._fake_async_client: Final = FakeAsyncRedis(server=server)

    def init_async_client(self) -> FakeAsyncRedis:
        return self._fake_async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_turn_is_refused_on_this_instance_s_count_when_reading_the_shared_counter_fails(
    kind: str,
) -> None:
    server: Final = FakeServer()
    dual_cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=_RedisCacheOnFakeServer(server))
    server.connected = False
    await dual_cache.in_memory_cache.async_increment(key=_turn_key(kind), value=100)

    with pytest.raises(litellm.RateLimitError, match=f"{kind.upper()} limit=100, current usage=100"):
        await async_admit_io_token_turn(dual_cache, _turn_deployment(kind), _TURN_AT)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("delivery", ["async", "sync"])
async def test_a_turn_finishing_in_the_next_minute_counts_its_actual_tokens_on_that_minute(
    kind: str, delivery: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "litellm.router_utils.pre_call_checks.io_token_rate_limit_check.get_utc_datetime",
        lambda: datetime(2026, 1, 1, 12, 35, 10, tzinfo=timezone.utc),
    )
    dual_cache: Final = DualCache()
    reservation: Final = await async_admit_io_token_turn(dual_cache, _turn_deployment(kind), _TURN_AT)

    await _deliver_success(
        ModelRateLimitingCheck(dual_cache=dual_cache),
        delivery,
        _proxy_success_kwargs(reservation),
        _reply_using(_ACTUAL_TOKENS["itpm"], _ACTUAL_TOKENS["otpm"]),
    )

    assert [await dual_cache.async_get_cache(key=_turn_key(kind, minute)) for minute in ("12-34", "12-35")] == [
        None,
        _ACTUAL_TOKENS[kind],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_turn_is_admitted_once_the_shared_counter_drops_below_the_limit_this_instance_last_read(
    kind: str,
) -> None:
    redis_cache: Final = _RedisCacheOnFakeServer(FakeServer())
    dual_cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)
    await dual_cache.in_memory_cache.async_increment(key=_turn_key(kind), value=100)
    await redis_cache.async_increment(key=_turn_key(kind), value=99)

    admitted: Final = await async_admit_io_token_turn(dual_cache, _turn_deployment(kind), _TURN_AT)

    assert admitted[ITPM_CACHE_KEY if kind == "itpm" else OTPM_CACHE_KEY] == _turn_key(kind)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses_reply",
    [
        pytest.param(
            ResponsesAPIResponse(
                id="resp_cached",
                created_at=0,
                output=[],
                usage=ResponseAPIUsage(
                    input_tokens=50,
                    output_tokens=5,
                    total_tokens=55,
                    input_tokens_details=InputTokensDetails(cached_tokens=30),
                ),
            ),
            id="usage_object",
        ),
        pytest.param(
            {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 5,
                    "total_tokens": 55,
                    "input_tokens_details": {"cached_tokens": 30},
                }
            },
            id="usage_dict",
        ),
    ],
)
@pytest.mark.usefixtures("settled_on_the_turn_s_minute")
async def test_a_turn_counts_only_its_uncached_responses_input_toward_itpm(responses_reply: object) -> None:
    dual_cache: Final = DualCache()
    await dual_cache.async_increment_cache(key=_turn_key("itpm"), value=10, ttl=60)

    reservation: Final = await async_admit_io_token_turn(dual_cache, _turn_deployment("itpm"), _TURN_AT)
    await async_io_token_reconcile_success(dual_cache, _proxy_success_kwargs(reservation), responses_reply)

    assert await dual_cache.async_get_cache(key=_turn_key("itpm")) == 10 + 50 - 30


def _shared_counter_answering(answer: int | Exception) -> DualCache:
    redis_cache: Final = MagicMock(spec=RedisCache)
    redis_cache.async_get_cache = AsyncMock(side_effect=[answer])
    return DualCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_later_turn_is_refused_when_another_instance_filled_the_shared_minute(kind: str) -> None:
    with pytest.raises(litellm.RateLimitError, match=f"{kind.upper()} limit=100, current usage=100"):
        await async_admit_io_token_turn(_shared_counter_answering(100), _turn_deployment(kind), _TURN_AT)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_later_turn_is_admitted_on_this_instance_s_count_when_the_shared_counter_is_unreachable(
    kind: str,
) -> None:
    dual_cache: Final = _shared_counter_answering(RedisCircuitBreakerOpenError("circuit open"))
    await dual_cache.in_memory_cache.async_increment(key=_turn_key(kind), value=99)

    admitted: Final = await async_admit_io_token_turn(dual_cache, _turn_deployment(kind), _TURN_AT)

    assert admitted[ITPM_CACHE_KEY if kind == "itpm" else OTPM_CACHE_KEY] == _turn_key(kind)
