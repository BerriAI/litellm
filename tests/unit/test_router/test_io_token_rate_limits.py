"""
Tests for separate ITPM/OTPM deployment rate limits (enforce_model_rate_limits).
"""

import asyncio
import contextvars
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Final, Literal

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.constants import NO_MAX_TOKENS_TPM_RESERVATION_FRACTION
from litellm.litellm_core_utils.attempt_settlement import routed_attempt
from litellm.litellm_core_utils.internal_call_metadata import IO_TOKEN_RESERVATION_ID_KEY
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_usage import CounterIncrement, RouterUsageStore, UsageCounter
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    ITPM_COUNTER_KEY,
    ITPM_RESERVED_KEY,
    OTPM_COUNTER_KEY,
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
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import get_utc_datetime
from tests.unit.redis.stores.router_usage_double import FakeUsageHandle, usage_store

MANTLE_OPUS = "bedrock_mantle/anthropic.claude-opus-4-7"
MANTLE_OPUS_MAX_OUTPUT = litellm.get_model_info(model=MANTLE_OPUS)["max_output_tokens"]
UNMAPPED_MODEL = "openai/litellm-io-unmapped-model"
UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT = 4096
AZURE_BASE_MODEL = "azure/gpt-5"
AZURE_BASE_MODEL_MAX_OUTPUT = litellm.get_model_info(model=AZURE_BASE_MODEL)["max_output_tokens"]


def _memory_usage() -> RouterUsageStore:
    return RouterUsageStore(connect(None).router_usage())


def _io(deployment_id: str, model: str, kind: Literal["itpm", "otpm"], window: str) -> UsageCounter:
    return UsageCounter("global_router", deployment_id, model, window, kind)


class _FailingKind(FakeUsageHandle):
    """A usage store whose increments of one counter kind fail, as a transient Redis error would."""

    def __init__(self, failing: str) -> None:
        super().__init__()
        self.failing: Final = failing

    def increment_sync(self, counter: UsageCounter, value: float, ttl_seconds: int) -> float:
        if counter.kind == self.failing:
            raise RuntimeError("transient cache error")
        return super().increment_sync(counter, value, ttl_seconds)


def _counter_in(value: object) -> UsageCounter:
    """The counter a reservation stashed in the request metadata."""
    assert isinstance(value, dict)
    return UsageCounter(**value)


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

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        itpm_key = _io("io-test-id", "bedrock_mantle/anthropic.claude-opus-4-7", "itpm", minute)
        otpm_key = _io("io-test-id", "bedrock_mantle/anthropic.claude-opus-4-7", "otpm", minute)

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

        current_itpm = usage.get_local(itpm_key)
        current_otpm = usage.get_local(otpm_key)
        # ITPM tracks input tokens only (billable prompt tokens), not output.
        assert current_itpm == 5
        assert current_otpm == 3

    @pytest.mark.asyncio
    async def test_itpm_limit_raises_429(self):
        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        otpm_key = _io("io-otpm-race-id", "bedrock_mantle/anthropic.claude-opus-4-7", "otpm", minute)
        current_otpm = usage.get_local(otpm_key)

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

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        itpm_key = _io("io-itpm-estimate-fail-id", "bedrock_mantle/anthropic.claude-opus-4-7", "itpm", minute)
        current_itpm = usage.get_local(itpm_key)

        # A minimal 1-token reservation per request lets itpm_limit concurrent
        # requests through, instead of a single request starving the rest.
        assert current_itpm is not None
        assert current_itpm <= itpm_limit
        assert successes == itpm_limit

    @pytest.mark.asyncio
    async def test_reservation_read_prefers_top_level_metadata_over_litellm_params(self):
        from litellm.utils import get_utc_datetime

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = _io("io-lp-id", "bedrock_mantle/test", "itpm", minute)
        usage.increment_local(CounterIncrement(itpm_key, 20, 60))

        # Production kwargs commonly carry litellm_params.metadata; the stashed
        # reservation lives in the top-level metadata and must still be found.
        kwargs = {
            "standard_logging_object": {
                "model_id": "io-lp-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {},
            },
            "metadata": {ITPM_RESERVED_KEY: 20, ITPM_COUNTER_KEY: asdict(itpm_key)},
            "litellm_params": {"metadata": {"user_api_key_hash": "abc123"}},
        }
        await check.async_log_failure_event(kwargs, None, None, None)

        current = usage.get_local(itpm_key)
        assert current == 0

    @pytest.mark.asyncio
    async def test_reconcile_tracks_actual_usage_when_estimate_zero(self):
        from litellm.utils import get_utc_datetime

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        itpm_key = _io("io-zero-est-id", "bedrock_mantle/anthropic.claude-opus-4-7", "itpm", minute)
        # A failed/zero estimate reserves a minimal 1 token, not the full
        # itpm limit, so it doesn't starve concurrent requests.
        assert usage.get_local(itpm_key) == 1

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

        assert usage.get_local(itpm_key) == 7

    @pytest.mark.asyncio
    async def test_zero_estimate_reserves_minimal_capacity_before_reconcile(self):
        """
        A zero/failed estimate reserves a minimal 1 token rather than the
        full itpm limit, so up to itpm_limit such calls are allowed
        concurrently instead of the first one claiming the entire bucket.
        """
        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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

        zero_output_otpm_key = _counter_in(zero_output_kwargs["metadata"][OTPM_COUNTER_KEY])
        assert zero_output_kwargs["metadata"][OTPM_RESERVED_KEY] == 0
        assert (usage.get_local(zero_output_otpm_key) or 0) == 0

        normal_output_kwargs = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(normal_output_kwargs, metadata_channel="metadata")
        await check.async_pre_call_check(deployment)

        normal_output_otpm_key = _counter_in(normal_output_kwargs["metadata"][OTPM_COUNTER_KEY])
        assert usage.get_local(normal_output_otpm_key) == 5

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
        usage: Final = _memory_usage()
        check: Final = ModelRateLimitingCheck(usage=usage)
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
        assert usage.get_local(_counter_in(request_kwargs["metadata"][OTPM_COUNTER_KEY])) == reserved

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, otpm_limit",
        [(MANTLE_OPUS, MANTLE_OPUS_MAX_OUTPUT), (UNMAPPED_MODEL, UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT)],
    )
    async def test_concurrent_requests_without_max_tokens_share_otpm_without_overshoot(
        self, model: str, otpm_limit: int
    ):
        usage: Final = _memory_usage()
        check: Final = ModelRateLimitingCheck(usage=usage)
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
        otpm_key: Final = _counter_in(admitted[0][OTPM_COUNTER_KEY])

        assert len(admitted) == round(1 / NO_MAX_TOKENS_TPM_RESERVATION_FRACTION)
        assert {_counter_in(m[OTPM_COUNTER_KEY]) for m in admitted} == {otpm_key}
        assert usage.get_local(otpm_key) == sum(m[OTPM_RESERVED_KEY] for m in admitted)
        assert usage.get_local(otpm_key) <= otpm_limit

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, model_max_output",
        [(MANTLE_OPUS, MANTLE_OPUS_MAX_OUTPUT), (UNMAPPED_MODEL, UNMAPPED_MODEL_DEFAULT_MAX_OUTPUT)],
    )
    async def test_request_without_max_tokens_reserves_model_max_output_when_otpm_allows(
        self, model: str, model_max_output: int
    ):
        usage: Final = _memory_usage()
        check: Final = ModelRateLimitingCheck(usage=usage)
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
    async def test_otpm_reservation_uses_each_deployments_output_limit(self) -> None:
        backend_model: Final = "openai/gpt-4o-mini"
        catalog_max_output: Final = litellm.get_model_info(model=backend_model)["max_output_tokens"]
        assert isinstance(catalog_max_output, int)
        otpm_limit: Final = int(max(catalog_max_output, 987) / NO_MAX_TOKENS_TPM_RESERVATION_FRACTION) + 1
        check: Final = ModelRateLimitingCheck(usage=_memory_usage())

        async def _reserved(deployment_id: str, model_info: dict[str, object]) -> int:
            metadata: Final[dict[str, object]] = {}
            request_kwargs: Final[dict[str, object]] = {
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": metadata,
            }
            set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")
            deployment: Final = {
                "litellm_params": {"model": backend_model, "otpm": otpm_limit},
                "model_info": {"id": deployment_id, **model_info},
            }
            await check.async_pre_call_check(deployment)
            reservation: Final = metadata.get(OTPM_RESERVED_KEY)
            assert isinstance(reservation, int)
            return reservation

        plain: Final = await _reserved("io-plain-gpt-4o-mini", {})
        override: Final = await _reserved("io-overridden-gpt-4o-mini", {"max_output_tokens": 987})

        assert plain == catalog_max_output
        assert override == 987

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_check", [False, True])
    async def test_base_model_does_not_change_the_default_otpm_reservation(self, use_sync_check: bool):
        check: Final = ModelRateLimitingCheck(usage=_memory_usage())
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
        usage: Final = _memory_usage()
        check: Final = ModelRateLimitingCheck(usage=usage)
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

        itpm_counter: Final = usage.get_local(_counter_in(valid_kwargs["metadata"][ITPM_COUNTER_KEY]))
        assert itpm_counter == valid_kwargs["metadata"][ITPM_RESERVED_KEY]

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_above_default_share_is_reserved_in_full(self):
        usage: Final = _memory_usage()
        check: Final = ModelRateLimitingCheck(usage=usage)
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
        check: Final = ModelRateLimitingCheck(usage=_memory_usage())
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
        check: Final = ModelRateLimitingCheck(usage=_memory_usage())
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
        check: Final = ModelRateLimitingCheck(usage=_memory_usage())
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

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        itpm_key = _io("io-sync-id", "bedrock_mantle/anthropic.claude-opus-4-7", "itpm", minute)
        otpm_key = _io("io-sync-id", "bedrock_mantle/anthropic.claude-opus-4-7", "otpm", minute)
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

        assert usage.get_local(itpm_key) == 5
        assert usage.get_local(otpm_key) == 3

    @pytest.mark.asyncio
    async def test_reconcile_runs_via_success_event_without_model_id(self):
        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        itpm_key = _io("io-noid", "bedrock_mantle/test", "itpm", "12-34")
        usage.increment_local(CounterIncrement(itpm_key, 10, 60))

        # standard_logging_object has no model_id (only the TPM path needs it);
        # IO reconciliation must still run off the stashed cache key.
        kwargs = {
            "standard_logging_object": {
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {},
            },
            "metadata": {ITPM_RESERVED_KEY: 10, ITPM_COUNTER_KEY: asdict(itpm_key)},
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )
        await check.async_log_success_event(kwargs, response, None, None)

        assert usage.get_local(itpm_key) == 3

    @pytest.mark.asyncio
    async def test_failure_clears_reservation_so_retry_is_not_poisoned(self):
        from litellm.utils import get_utc_datetime

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = _io("io-first", "bedrock_mantle/test", "itpm", minute)
        usage.increment_local(CounterIncrement(itpm_key, 8, 60))

        # Shared request metadata carrying the first (IO) deployment's reservation.
        metadata = {ITPM_RESERVED_KEY: 8, ITPM_COUNTER_KEY: asdict(itpm_key)}
        fail_kwargs = {
            "metadata": metadata,
            "standard_logging_object": {
                "model_id": "io-first",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": {},
            },
        }
        await check.async_log_failure_event(fail_kwargs, None, None, None)

        assert usage.get_local(itpm_key) == 0
        assert ITPM_RESERVED_KEY not in metadata
        assert ITPM_COUNTER_KEY not in metadata

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
        assert usage.get_local(itpm_key) == 0
        # ...and the non-IO deployment's TPM usage is tracked normally.
        tpm = UsageCounter.for_deployment("non-io-second", "openai/gpt-4o-mini", "tpm", minute)
        assert usage.get_local(tpm) == 12

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

        usage = _memory_usage()
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key_a = _io("io-retry-a", "bedrock_mantle/test-a", "itpm", minute)
        usage.increment_local(CounterIncrement(itpm_key_a, 9, 60))

        # Deployment A's still-unreconciled reservation, stashed on the shared
        # kwargs dict the retry loop reuses.
        shared_kwargs = {"metadata": {ITPM_RESERVED_KEY: 9, ITPM_COUNTER_KEY: asdict(itpm_key_a)}}

        # Router calls this before overwriting kwargs for deployment B's attempt -
        # simulating the fix landing ahead of set_io_token_rate_limit_request_kwargs.
        refund_stale_reservation_before_retry(usage, shared_kwargs)

        # A's reservation is refunded immediately, not left stranded for a
        # background failure task that may run arbitrarily later (or never,
        # if the sentinels get cleared out from under it first).
        assert usage.get_local(itpm_key_a) == 0
        assert ITPM_RESERVED_KEY not in shared_kwargs["metadata"]
        assert ITPM_COUNTER_KEY not in shared_kwargs["metadata"]

        # A's own (now-late) failure event finds nothing left to refund and
        # is a safe no-op, since the sentinels were already cleared above.
        io_token_refund_failure(usage, shared_kwargs)
        assert usage.get_local(itpm_key_a) == 0

        # The retry proceeds to stash deployment B's own reservation on the
        # same dict; it starts clean, unaffected by A's cleared sentinels.
        set_io_token_rate_limit_request_kwargs(shared_kwargs, metadata_channel="metadata")
        itpm_key_b = _io("io-retry-b", "bedrock_mantle/test-b", "itpm", minute)
        shared_kwargs["metadata"][ITPM_RESERVED_KEY] = 4
        shared_kwargs["metadata"][ITPM_COUNTER_KEY] = asdict(itpm_key_b)
        usage.increment_local(CounterIncrement(itpm_key_b, 4, 60))
        assert usage.get_local(itpm_key_b) == 4

    @pytest.mark.asyncio
    async def test_client_supplied_reservation_keys_are_stripped(self):
        # metadata is caller-controlled; the server-only reservation sentinels
        # must be removed before the router captures the request kwargs.
        forged = {
            "metadata": {ITPM_RESERVED_KEY: 999999, ITPM_COUNTER_KEY: asdict(_io("attacker", "key", "itpm", "00-00"))},
            "litellm_metadata": {OTPM_RESERVED_KEY: 7},
            "litellm_params": {"metadata": {OTPM_COUNTER_KEY: asdict(_io("attacker", "key", "otpm", "00-00"))}},
        }
        set_io_token_rate_limit_request_kwargs(forged, metadata_channel="metadata")
        stored = get_io_token_rate_limit_request_kwargs()

        assert ITPM_RESERVED_KEY not in stored["metadata"]
        assert ITPM_COUNTER_KEY not in stored["metadata"]
        assert OTPM_RESERVED_KEY not in stored["litellm_metadata"]
        assert OTPM_COUNTER_KEY not in stored["litellm_params"]["metadata"]

    @pytest.mark.asyncio
    async def test_forged_reservation_cannot_decrement_counter(self):
        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        victim_key = _io("victim", "model", "itpm", "00-00")
        usage.increment_local(CounterIncrement(victim_key, 100, 60))

        # A caller forges a reservation pointing at another deployment's counter.
        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 100, ITPM_COUNTER_KEY: asdict(victim_key)},
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
        assert usage.get_local(victim_key) == 100

    @pytest.mark.asyncio
    async def test_otpm_reservation_error_rolls_back_itpm(self):
        from litellm.utils import get_utc_datetime

        usage = usage_store(_FailingKind("otpm"))
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
            await async_io_token_pre_call_check(usage, deployment)

        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = _io("io-rollback-id", "bedrock_mantle/anthropic.claude-opus-4-7", "itpm", minute)
        # A transient OTPM error must release the ITPM reservation, not leak it.
        assert (usage.get_local(itpm_key) or 0) == 0

    @pytest.mark.asyncio
    async def test_reconcile_clears_stash_even_when_increment_errors(self):
        usage = usage_store(_FailingKind("itpm"))
        metadata = {ITPM_RESERVED_KEY: 5, ITPM_COUNTER_KEY: asdict(_io("x", "model", "itpm", "00-00"))}
        kwargs = {"metadata": metadata}
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=3, completion_tokens=0, total_tokens=3),
        )

        await async_io_token_reconcile_success(usage, kwargs, response)

        assert ITPM_RESERVED_KEY not in metadata
        assert ITPM_COUNTER_KEY not in metadata

    @pytest.mark.asyncio
    async def test_io_conflict_warning_not_collapsed_for_missing_model_id(self, caplog):
        import logging

        check = ModelRateLimitingCheck(usage=_memory_usage())
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
        usage = _memory_usage()
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

        result = await async_io_token_pre_call_check(usage, deployment)

        assert result is deployment
        # No reservation is stashed, so nothing lands in a shared None:None bucket.
        assert ITPM_RESERVED_KEY not in request_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_reconcile_uses_reservation_minute_key(self):
        usage = _memory_usage()
        # Reservation was made on a fixed minute key; a call that finishes in a
        # later minute must reconcile against that same key, never a key built
        # from the response-time minute.
        itpm_key = _io("io-min-id", "bedrock_mantle/test", "itpm", "99-99")
        usage.increment_local(CounterIncrement(itpm_key, 10, 60))

        kwargs = {"metadata": {ITPM_RESERVED_KEY: 10, ITPM_COUNTER_KEY: asdict(itpm_key)}}
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=4, completion_tokens=0, total_tokens=4),
        )
        await async_io_token_reconcile_success(usage, kwargs, response)

        assert usage.get_local(itpm_key) == 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_sync_reconcile", [False, True])
    async def test_reconcile_charges_usage_above_the_reservation_to_the_minute_it_finished_in(
        self, use_sync_reconcile: bool
    ):
        usage: Final = _memory_usage()
        reserved_minute_key: Final = _io("io-late-id", "bedrock_mantle/test", "otpm", "99-99")
        usage.increment_local(CounterIncrement(reserved_minute_key, 10, 60))
        kwargs: Final = {
            "metadata": {ITPM_RESERVED_KEY: 0, OTPM_RESERVED_KEY: 10, OTPM_COUNTER_KEY: asdict(reserved_minute_key)}
        }
        response: Final = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
            usage=Usage(prompt_tokens=4, completion_tokens=25, total_tokens=29),
        )
        minute_before: Final = get_utc_datetime().strftime("%H-%M")

        if use_sync_reconcile:
            io_token_reconcile_success(usage, kwargs, response)
        else:
            await async_io_token_reconcile_success(usage, kwargs, response)

        finished_minutes: Final = {minute_before, get_utc_datetime().strftime("%H-%M")}
        charged: Final = [
            usage.get_local(_io("io-late-id", "bedrock_mantle/test", "otpm", minute)) for minute in finished_minutes
        ]
        assert usage.get_local(reserved_minute_key) == 10
        assert sum(value or 0 for value in charged) == 25 - 10

    @pytest.mark.asyncio
    async def test_reconcile_missing_usage_keeps_reservation(self):
        usage = _memory_usage()
        itpm_key = _io("io-missing-usage", "bedrock_mantle/test", "itpm", "00-00")
        otpm_key = _io("io-missing-usage", "bedrock_mantle/test", "otpm", "00-00")
        usage.increment_local(CounterIncrement(itpm_key, 8, 60))
        usage.increment_local(CounterIncrement(otpm_key, 5, 60))

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 8,
                OTPM_RESERVED_KEY: 5,
                ITPM_COUNTER_KEY: asdict(itpm_key),
                OTPM_COUNTER_KEY: asdict(otpm_key),
            }
        }
        response = ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
        )

        await async_io_token_reconcile_success(usage, kwargs, response)

        assert usage.get_local(itpm_key) == 8
        assert usage.get_local(otpm_key) == 5
        assert ITPM_RESERVED_KEY not in kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_reconcile_total_tokens_only_keeps_reservation(self):
        """
        A response usage object with only total_tokens (no prompt/completion
        breakdown) can't be split into input/output, so it must be treated the
        same as missing usage: keep the reservation instead of resolving to
        (0, 0) and refunding it in full.
        """
        usage = _memory_usage()
        itpm_key = _io("io-total-only", "bedrock_mantle/test", "itpm", "00-00")
        otpm_key = _io("io-total-only", "bedrock_mantle/test", "otpm", "00-00")
        usage.increment_local(CounterIncrement(itpm_key, 8, 60))
        usage.increment_local(CounterIncrement(otpm_key, 5, 60))

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 8,
                OTPM_RESERVED_KEY: 5,
                ITPM_COUNTER_KEY: asdict(itpm_key),
                OTPM_COUNTER_KEY: asdict(otpm_key),
            }
        }
        response = {"type": "message", "usage": {"total_tokens": 13}}

        await async_io_token_reconcile_success(usage, kwargs, response)

        assert usage.get_local(itpm_key) == 8
        assert usage.get_local(otpm_key) == 5

    def test_reconcile_standard_logging_total_tokens_only_keeps_reservation(self):
        usage = _memory_usage()
        itpm_key = _io("io-slo-total-only", "bedrock_mantle/test", "itpm", "00-00")
        usage.set_local(itpm_key, 10)

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 10, ITPM_COUNTER_KEY: asdict(itpm_key)},
            "standard_logging_object": {"total_tokens": 4},
        }
        response = {"type": "message", "role": "assistant", "content": []}

        io_token_reconcile_success(usage, kwargs, response)

        assert usage.get_local(itpm_key) == 10

    @pytest.mark.asyncio
    async def test_reconcile_falls_back_to_standard_logging_object(self):
        usage = _memory_usage()
        itpm_key = _io("io-slo-fallback", "bedrock_mantle/test", "itpm", "00-00")
        usage.increment_local(CounterIncrement(itpm_key, 10, 60))

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 10, ITPM_COUNTER_KEY: asdict(itpm_key)},
            "standard_logging_object": {
                "prompt_tokens": 4,
                "completion_tokens": 0,
                "total_tokens": 4,
            },
        }
        response = {"type": "message", "role": "assistant", "content": []}

        await async_io_token_reconcile_success(usage, kwargs, response)

        assert usage.get_local(itpm_key) == 4

    def test_sync_reconcile_anthropic_dict_usage_counts_uncached_input_once(self):
        usage = _memory_usage()
        itpm_key = _io("io-anthropic", "bedrock_mantle/test", "itpm", "00-00")
        otpm_key = _io("io-anthropic", "bedrock_mantle/test", "otpm", "00-00")
        usage.set_local(itpm_key, 10)
        usage.set_local(otpm_key, 4)

        kwargs = {
            "metadata": {
                ITPM_RESERVED_KEY: 10,
                OTPM_RESERVED_KEY: 4,
                ITPM_COUNTER_KEY: asdict(itpm_key),
                OTPM_COUNTER_KEY: asdict(otpm_key),
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

        io_token_reconcile_success(usage, kwargs, response)

        assert usage.get_local(itpm_key) == 3 + 4
        assert usage.get_local(otpm_key) == 2

    @pytest.mark.asyncio
    async def test_io_and_tpm_rpm_limits_both_enforced_with_warning(self, caplog):
        import logging

        from litellm.utils import get_utc_datetime

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
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
        rpm = UsageCounter.for_deployment(model_id, deployment_name, "rpm", minute)
        itpm_key = _io(model_id, deployment_name, "itpm", minute)
        usage.increment_local(CounterIncrement(rpm, 5, 60))

        request_kwargs = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "metadata": {},
        }
        set_io_token_rate_limit_request_kwargs(request_kwargs, metadata_channel="metadata")

        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            with pytest.raises(litellm.RateLimitError):
                await check.async_pre_call_check(deployment)

        assert usage.get_local(rpm) == 5, "the rejected request hands its rpm count back"
        assert (usage.get_local(itpm_key) or 0) == 0
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

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        model_id = "io-tpm-mixed-id"
        deployment_name = "bedrock_mantle/anthropic.claude-opus-4-7"
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = _io(model_id, deployment_name, "itpm", minute)
        tpm = UsageCounter.for_deployment(model_id, deployment_name, "tpm", minute)
        usage.increment_local(CounterIncrement(itpm_key, 5, 60))

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 5, ITPM_COUNTER_KEY: asdict(itpm_key)},
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
        assert usage.get_local(itpm_key) == 3
        # TPM tracking must still run so the tpm/rpm pre-call path can enforce it.
        assert usage.get_local(tpm) == 7

    def test_io_success_still_tracks_tpm_for_mixed_deployment_sync(self):
        from litellm.utils import get_utc_datetime

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        model_id = "io-tpm-mixed-sync-id"
        deployment_name = "bedrock_mantle/anthropic.claude-opus-4-7"
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = _io(model_id, deployment_name, "itpm", minute)
        tpm = UsageCounter.for_deployment(model_id, deployment_name, "tpm", minute)
        usage.set_local(itpm_key, 5)

        kwargs = {
            "metadata": {ITPM_RESERVED_KEY: 5, ITPM_COUNTER_KEY: asdict(itpm_key)},
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

        assert usage.get_local(itpm_key) == 3
        assert usage.get_local(tpm) == 7

    @pytest.mark.asyncio
    async def test_failure_refunds_itpm_reservation(self):
        from litellm.utils import get_utc_datetime

        usage = _memory_usage()
        check = ModelRateLimitingCheck(usage=usage)
        minute = get_utc_datetime().strftime("%H-%M")
        itpm_key = _io("io-refund-id", "bedrock_mantle/test", "itpm", minute)
        usage.increment_local(CounterIncrement(itpm_key, 20, 60))

        reservation = {ITPM_RESERVED_KEY: 20, ITPM_COUNTER_KEY: asdict(itpm_key)}
        kwargs = {
            "standard_logging_object": {
                "model_id": "io-refund-id",
                "hidden_params": {"litellm_model_name": "bedrock_mantle/test"},
                "metadata": dict(reservation),
            },
            "metadata": dict(reservation),
        }
        await check.async_log_failure_event(kwargs, None, None, None)

        current = usage.get_local(itpm_key)
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


def _counter(router: Router, kind: Literal["itpm", "otpm"], minutes: frozenset[str]) -> int:
    deployment: Final = router.model_list[0]
    usage: Final = router._core._usage  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # the store the router counts io tokens in
    return sum(
        int(
            usage.get_local(_io(deployment["model_info"]["id"], deployment["litellm_params"]["model"], kind, minute))
            or 0
        )
        for minute in minutes
    )


async def _io_counters_since(router: Router, first_minute: str) -> tuple[int, int]:
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    minutes: Final = frozenset({first_minute, _current_minute()})
    return _counter(router, "itpm", minutes), _counter(router, "otpm", minutes)


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
            "metadata": {ITPM_RESERVED_KEY: 999, ITPM_COUNTER_KEY: asdict(_io("forged", "model", "itpm", "00-00"))},
        }
        set_io_token_rate_limit_request_kwargs(kwargs, store_in_context=False, metadata_channel="metadata")
        assert get_io_token_rate_limit_request_kwargs() is None
        assert ITPM_RESERVED_KEY not in kwargs["metadata"]
        assert ITPM_COUNTER_KEY not in kwargs["metadata"]

        set_io_token_rate_limit_request_kwargs(kwargs, store_in_context=True, metadata_channel="metadata")
        assert get_io_token_rate_limit_request_kwargs() is kwargs

        set_io_token_rate_limit_request_kwargs(kwargs, store_in_context=False, metadata_channel="metadata")
        assert get_io_token_rate_limit_request_kwargs() is None


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
        await async_io_token_pre_call_check(_memory_usage(), deployment)
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
    usage: Final = _memory_usage()
    check: Final = ModelRateLimitingCheck(usage=usage)
    itpm_key: Final = _io("proxy-shape", "openai/gpt-5-mini", "itpm", "12-34")
    otpm_key: Final = _io("proxy-shape", "openai/gpt-5-mini", "otpm", "12-34")
    usage.increment_local(CounterIncrement(itpm_key, 40, 60))
    usage.increment_local(CounterIncrement(otpm_key, 4096, 60))
    kwargs: Final = _proxy_success_kwargs(
        {
            ITPM_RESERVED_KEY: 40,
            OTPM_RESERVED_KEY: 4096,
            ITPM_COUNTER_KEY: asdict(itpm_key),
            OTPM_COUNTER_KEY: asdict(otpm_key),
        }
    )

    for delivery in deliveries:
        await _deliver_success(check, delivery, kwargs, _reply_using(prompt_tokens=7, completion_tokens=3))

    assert usage.get_local(itpm_key) == 7
    assert usage.get_local(otpm_key) == 3


_TURN_AT: Final = datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
_ACTUAL_TOKENS: Final = {"itpm": 5, "otpm": 2}


def _turn_deployment(kind: str) -> dict[str, object]:
    return {"litellm_params": {"model": "openai/gpt-5-mini", kind: 100}, "model_info": {"id": f"ws-turn-{kind}"}}


def _turn_key(kind: Literal["itpm", "otpm"], minute: str = "12-34") -> UsageCounter:
    return _io(f"ws-turn-{kind}", "openai/gpt-5-mini", kind, minute)


@pytest.fixture
def settled_on_the_turn_s_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "litellm.router_utils.pre_call_checks.io_token_rate_limit_check.get_utc_datetime", lambda: _TURN_AT
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("used", [100, 101, 250])
async def test_turn_admission_refuses_once_the_minutes_tokens_reach_the_limit(kind: str, used: int) -> None:
    usage: Final = _memory_usage()
    usage.increment_local(CounterIncrement(_turn_key(kind), used, 60))

    with pytest.raises(litellm.RateLimitError, match=f"{kind.upper()} limit=100, current usage={used}"):
        await async_admit_io_token_turn(usage, _turn_deployment(kind), _TURN_AT)


@pytest.mark.asyncio
@pytest.mark.usefixtures("settled_on_the_turn_s_minute")
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("used", [0, 1, 99])
async def test_an_admitted_turn_reserves_nothing_and_counts_its_actual_tokens(kind: str, used: int) -> None:
    usage: Final = _memory_usage()
    usage.increment_local(CounterIncrement(_turn_key(kind), used, 60))
    actual: Final = _ACTUAL_TOKENS

    reservation: Final = await async_admit_io_token_turn(usage, _turn_deployment(kind), _TURN_AT)
    counter_key: Final = _counter_in(reservation[ITPM_COUNTER_KEY if kind == "itpm" else OTPM_COUNTER_KEY])
    assert usage.get_local(counter_key) == used
    await async_io_token_reconcile_success(
        usage, _proxy_success_kwargs(reservation), _reply_using(actual["itpm"], actual["otpm"])
    )

    assert usage.get_local(counter_key) == used + actual[kind]


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
    assert await async_admit_io_token_turn(_memory_usage(), deployment, _TURN_AT) == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("settled_on_the_turn_s_minute")
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
@pytest.mark.parametrize("used_by_others", [0, 1, 99])
async def test_a_socket_open_reserves_nothing_and_its_first_turn_counts_its_actual_tokens_on_its_own_minute(
    kind: str, used_by_others: int
) -> None:
    usage: Final = _memory_usage()
    usage.increment_local(CounterIncrement(_turn_key(kind), used_by_others, 60))
    socket_open: Final[dict[str, object]] = {"metadata": {}}
    set_io_token_rate_limit_request_kwargs(socket_open, opens_responses_socket=True, metadata_channel="metadata")

    assert await async_io_token_pre_call_check(usage, _turn_deployment(kind)) == _turn_deployment(kind)
    assert usage.get_local(_turn_key(kind)) == used_by_others

    opened_with: Final = socket_open["metadata"]
    assert isinstance(opened_with, dict)
    await async_io_token_reconcile_success(
        usage, _proxy_success_kwargs(opened_with), _reply_using(_ACTUAL_TOKENS["itpm"], _ACTUAL_TOKENS["otpm"])
    )

    assert usage.get_local(_turn_key(kind)) == used_by_others + _ACTUAL_TOKENS[kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_turn_is_refused_on_this_instance_s_count_when_reading_the_shared_counter_fails(
    kind: Literal["itpm", "otpm"],
) -> None:
    usage: Final = usage_store(FakeUsageHandle(local={_turn_key(kind): 100}, redis={}, redis_down=True))

    with pytest.raises(litellm.RateLimitError, match=f"{kind.upper()} limit=100, current usage=100"):
        await async_admit_io_token_turn(usage, _turn_deployment(kind), _TURN_AT)


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
    usage: Final = _memory_usage()
    reservation: Final = await async_admit_io_token_turn(usage, _turn_deployment(kind), _TURN_AT)

    await _deliver_success(
        ModelRateLimitingCheck(usage=usage),
        delivery,
        _proxy_success_kwargs(reservation),
        _reply_using(_ACTUAL_TOKENS["itpm"], _ACTUAL_TOKENS["otpm"]),
    )

    assert [usage.get_local(_turn_key(kind, minute)) for minute in ("12-34", "12-35")] == [
        None,
        _ACTUAL_TOKENS[kind],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_turn_is_admitted_once_the_shared_counter_drops_below_the_limit_this_instance_last_read(
    kind: str,
) -> None:
    usage: Final = usage_store(FakeUsageHandle(local={_turn_key(kind): 100}, redis={_turn_key(kind): 99}))

    admitted: Final = await async_admit_io_token_turn(usage, _turn_deployment(kind), _TURN_AT)

    assert _counter_in(admitted[ITPM_COUNTER_KEY if kind == "itpm" else OTPM_COUNTER_KEY]) == _turn_key(kind)


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
    usage: Final = _memory_usage()
    usage.increment_local(CounterIncrement(_turn_key("itpm"), 10, 60))

    reservation: Final = await async_admit_io_token_turn(usage, _turn_deployment("itpm"), _TURN_AT)
    await async_io_token_reconcile_success(usage, _proxy_success_kwargs(reservation), responses_reply)

    assert usage.get_local(_turn_key("itpm")) == 10 + 50 - 30


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_later_turn_is_refused_when_another_instance_filled_the_shared_minute(kind: str) -> None:
    with pytest.raises(litellm.RateLimitError, match=f"{kind.upper()} limit=100, current usage=100"):
        await async_admit_io_token_turn(
            usage_store(FakeUsageHandle(redis={_turn_key(kind): 100})), _turn_deployment(kind), _TURN_AT
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["itpm", "otpm"])
async def test_a_later_turn_is_admitted_on_this_instance_s_count_when_the_shared_counter_is_unreachable(
    kind: str,
) -> None:
    usage: Final = usage_store(FakeUsageHandle(local={_turn_key(kind): 99}, redis={}, breaker_open=True))

    admitted: Final = await async_admit_io_token_turn(usage, _turn_deployment(kind), _TURN_AT)

    assert _counter_in(admitted[ITPM_COUNTER_KEY if kind == "itpm" else OTPM_COUNTER_KEY]) == _turn_key(kind)


_HELD_DEPLOYMENT: Final = {
    "litellm_params": {"model": ROUTER_IO_MODEL, "itpm": 50000, "otpm": 50000},
    "model_info": {"id": "held-id"},
    "model_name": "limited",
}


def _held_keys() -> tuple[UsageCounter, UsageCounter]:
    minute: Final = _current_minute()
    return (
        _io("held-id", ROUTER_IO_MODEL, "itpm", minute),
        _io("held-id", ROUTER_IO_MODEL, "otpm", minute),
    )


class _HeldIncrement(FakeUsageHandle):
    """A connected usage store holding the first increment matching ``holds`` until ``release`` is set."""

    def __init__(self, holds: Callable[[UsageCounter, float], bool]) -> None:
        super().__init__(redis={})
        self._holds: Final = holds
        self.holding: Final = asyncio.Event()
        self.release: Final = asyncio.Event()

    async def increment(self, counter: UsageCounter, value: float, ttl_seconds: int) -> float:
        if not self.holding.is_set() and self._holds(counter, value):
            self.holding.set()
            await self.release.wait()
        return self.increment_sync(counter, value, ttl_seconds)


async def _counts(usage: RouterUsageStore) -> tuple[int, int]:
    for _ in range(10):
        await asyncio.sleep(0)
    itpm_key, otpm_key = _held_keys()
    itpm: Final = usage.get_local(itpm_key)
    otpm: Final = usage.get_local(otpm_key)
    return (itpm if isinstance(itpm, int) else 0), (otpm if isinstance(otpm, int) else 0)


async def _cancel_twice_while_held(held_store: _HeldIncrement, held: "asyncio.Task[object]") -> None:
    await held_store.holding.wait()
    held.cancel()
    held.cancel()
    with pytest.raises(asyncio.CancelledError):
        await held
    held_store.release.set()


@pytest.mark.asyncio
async def test_a_request_cancelled_mid_reservation_hands_back_what_it_reserved() -> None:
    held_store: Final = _HeldIncrement(lambda counter, value: counter.kind == "otpm" and value > 0)
    cache: Final = usage_store(held_store)
    request: Final[dict[str, object]] = {"messages": PROXY_MESSAGES, "max_tokens": 100, "metadata": {}}

    async def reserve() -> object:
        set_io_token_rate_limit_request_kwargs(request, metadata_channel="metadata")
        return await async_io_token_pre_call_check(cache, dict(_HELD_DEPLOYMENT))

    await _cancel_twice_while_held(held_store, asyncio.create_task(reserve()))

    assert await _counts(cache) == (0, 0)
    assert ITPM_RESERVED_KEY not in request["metadata"]


def _stashed(itpm_reserved: int, otpm_reserved: int) -> dict[str, object]:
    itpm_key, otpm_key = _held_keys()
    return {
        "metadata": {
            ITPM_RESERVED_KEY: itpm_reserved,
            OTPM_RESERVED_KEY: otpm_reserved,
            ITPM_COUNTER_KEY: asdict(itpm_key),
            OTPM_COUNTER_KEY: asdict(otpm_key),
        }
    }


async def _reserved(cache: RouterUsageStore, itpm_reserved: int, otpm_reserved: int) -> dict[str, object]:
    itpm_key, otpm_key = _held_keys()
    await cache.increment(itpm_key, itpm_reserved, 60)
    await cache.increment(otpm_key, otpm_reserved, 60)
    return _stashed(itpm_reserved, otpm_reserved)


@pytest.mark.asyncio
async def test_a_refund_cancelled_between_its_decrements_still_hands_back_both() -> None:
    held_store: Final = _HeldIncrement(lambda counter, value: counter.kind == "itpm" and value < 0)
    cache: Final = usage_store(held_store)
    kwargs: Final = await _reserved(cache, 9, 100)

    await _cancel_twice_while_held(
        held_store, asyncio.create_task(ModelRateLimitingCheck(cache).async_log_failure_event(kwargs, None, None, None))
    )

    assert await _counts(cache) == (0, 0)


class _YieldingIncrement(FakeUsageHandle):
    """A connected usage store whose increments yield to the event loop first."""

    def __init__(self) -> None:
        super().__init__(redis={})

    async def increment(self, counter: UsageCounter, value: float, ttl_seconds: int) -> float:
        await asyncio.sleep(0)
        return self.increment_sync(counter, value, ttl_seconds)


@pytest.mark.asyncio
async def test_competing_failure_callbacks_hand_a_reservation_back_once() -> None:
    cache: Final = usage_store(_YieldingIncrement())
    kwargs: Final = await _reserved(cache, 9, 100)
    check: Final = ModelRateLimitingCheck(cache)

    await asyncio.gather(*(check.async_log_failure_event(kwargs, None, None, None) for _ in range(2)))

    assert await _counts(cache) == (0, 0)


@pytest.mark.asyncio
async def test_a_late_failure_callback_of_one_attempt_leaves_the_next_attempts_reservation_held() -> None:
    cache: Final = _memory_usage()
    check: Final = ModelRateLimitingCheck(cache)
    request: Final[dict[str, object]] = {"messages": PROXY_MESSAGES, "max_tokens": 100, "metadata": {}}

    async def attempt() -> contextvars.Context:
        with routed_attempt():
            set_io_token_rate_limit_request_kwargs(request, metadata_channel="metadata")
            await async_io_token_pre_call_check(cache, dict(_HELD_DEPLOYMENT))
            return contextvars.copy_context()

    def failure_logged_in(context: contextvars.Context) -> "asyncio.Task[None]":
        return context.run(asyncio.create_task, check.async_log_failure_event(request, None, None, None))

    first: Final = await attempt()
    await failure_logged_in(first)
    _ = await attempt()
    held_by_second: Final = await _counts(cache)

    await failure_logged_in(first)

    assert held_by_second[1] == 100
    assert await _counts(cache) == held_by_second


def _anthropic_reply() -> dict[str, object]:
    return {
        "id": "msg_held",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


@pytest.mark.parametrize(
    "attempts",
    [("hang",), ("503", "hang")],
    ids=["cancelled_while_waiting_on_the_provider", "cancelled_on_its_retry"],
)
@pytest.mark.asyncio
async def test_a_cancelled_request_sharing_one_logging_object_hands_back_every_reservation(
    attempts: tuple[str, ...], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    waiting_on_provider: Final = asyncio.Event()
    script: Final = iter((*attempts, "ok"))

    async def upstream(_: httpx.Request) -> httpx.Response:
        match next(script):
            case "503":
                return httpx.Response(
                    503, json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}}
                )
            case "hang":
                waiting_on_provider.set()
                await asyncio.Event().wait()
        return httpx.Response(200, json=_anthropic_reply())

    respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=upstream)
    router: Final = _router_with_io_limits()
    first_minute: Final = _current_minute()
    call: Final["asyncio.Task[object]"] = asyncio.create_task(
        router.acompletion(
            model="limited",
            messages=PROXY_MESSAGES,
            max_tokens=100,
            metadata={"user_api_key_hash": "hashed-key"},
            litellm_call_id="proxy-call",
            litellm_logging_obj=_logging_obj_shared_across_attempts("acompletion"),
        )
    )
    await waiting_on_provider.wait()
    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call

    assert await _io_counters_since(router, first_minute) == (0, 0)
    assert await _proxy_chat(router, "hi") == await _io_counters_since(router, first_minute)


@pytest.mark.asyncio
async def test_a_reservation_refunded_through_a_logged_copy_of_its_metadata_is_refunded_once() -> None:
    cache: Final = _memory_usage()
    check: Final = ModelRateLimitingCheck(cache)
    request: Final[dict[str, object]] = {"messages": PROXY_MESSAGES, "max_tokens": 100, "litellm_metadata": {}}

    async def cancelled_attempt() -> None:
        with routed_attempt():
            set_io_token_rate_limit_request_kwargs(request, metadata_channel="litellm_metadata")
            await async_io_token_pre_call_check(cache, dict(_HELD_DEPLOYMENT))
            logged.update({"litellm_params": {"litellm_metadata": dict(request["litellm_metadata"])}})  # pyright: ignore[reportArgumentType]  # the stashed metadata dict
            await check.async_log_failure_event(request, None, None, None)

    logged: Final[dict[str, object]] = {}
    await asyncio.create_task(cancelled_attempt())

    await check.async_log_failure_event(logged, None, None, None)

    assert await _counts(cache) == (0, 0)


@pytest.mark.asyncio
async def test_a_retry_refunds_the_reservation_its_previous_attempt_left_behind() -> None:
    cache: Final = _memory_usage()
    request: Final[dict[str, object]] = {"messages": PROXY_MESSAGES, "max_tokens": 100, "metadata": {}}

    async def first_attempt() -> None:
        with routed_attempt():
            set_io_token_rate_limit_request_kwargs(request, metadata_channel="metadata")
            await async_io_token_pre_call_check(cache, dict(_HELD_DEPLOYMENT))

    def retry() -> None:
        with routed_attempt():
            refund_stale_reservation_before_retry(cache, request)

    await asyncio.create_task(first_attempt())
    contextvars.copy_context().run(retry)

    assert await _counts(cache) == (0, 0)


@pytest.mark.asyncio
async def test_a_request_cancelled_after_its_provider_answered_keeps_the_tokens_it_spent_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.integrations.custom_logger import CustomLogger

    answered: Final = asyncio.Event()

    class SlowSuccessHook(CustomLogger):
        async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
            answered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, SlowSuccessHook()])
    router: Final = _router_with_io_limits()
    first_minute: Final = _current_minute()
    call: Final["asyncio.Task[tuple[int, int]]"] = asyncio.create_task(_proxy_chat(router, "hi"))
    await answered.wait()
    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call

    _, otpm = await _io_counters_since(router, first_minute)
    assert otpm == 100


def _stashed_as(reservation_id: str, itpm_reserved: int, otpm_reserved: int) -> dict[str, object]:
    stashed: Final = _stashed(itpm_reserved, otpm_reserved)
    return {"metadata": {**stashed["metadata"], IO_TOKEN_RESERVATION_ID_KEY: reservation_id}}  # pyright: ignore[reportGeneralTypeIssues]  # the stash is a dict


@pytest.mark.asyncio
async def test_a_settled_reservation_stays_settled_however_many_others_settle_after_it() -> None:
    cache: Final = _memory_usage()
    check: Final = ModelRateLimitingCheck(cache)
    itpm_key, otpm_key = _held_keys()
    await cache.increment(itpm_key, 9, 60)
    await cache.increment(otpm_key, 100, 60)
    first: Final = _stashed_as("first-reservation", 9, 100)
    logged_copy: Final = {"litellm_params": {"metadata": dict(first["metadata"])}}  # pyright: ignore[reportArgumentType]  # the stash is a dict
    await check.async_log_failure_event(first, None, None, None)
    for n in range(10_001):
        io_token_refund_failure(cache, _stashed_as(f"later-reservation-{n}", 0, 0))

    await check.async_log_failure_event(logged_copy, None, None, None)

    assert await _counts(cache) == (0, 0)


class _FirstChunkNeverArrives(httpx.AsyncByteStream):
    def __init__(self, reading: asyncio.Event) -> None:
        self._reading: Final = reading

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._reading.set()
        await asyncio.Event().wait()
        yield b""


@pytest.mark.asyncio
async def test_a_stream_cancelled_before_its_first_chunk_hands_back_its_reservation(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    reading_first_chunk: Final = asyncio.Event()
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_FirstChunkNeverArrives(reading_first_chunk)
        )
    )
    router: Final = _router_with_io_limits()
    first_minute: Final = _current_minute()
    call: Final["asyncio.Task[object]"] = asyncio.create_task(
        router.acompletion(
            model="limited",
            messages=PROXY_MESSAGES,
            max_tokens=100,
            stream=True,
            metadata={"user_api_key_hash": "hashed-key"},
        )
    )
    await reading_first_chunk.wait()
    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call

    assert await _io_counters_since(router, first_minute) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("settled_on_the_turn_s_minute")
async def test_a_stashed_reservation_counts_under_mains_redis_keys() -> None:
    usage: Final = _memory_usage()
    deployment: Final = {
        "litellm_params": {"model": "bedrock/claude", "itpm": 100, "otpm": 100},
        "model_info": {"id": "dep-1"},
    }

    reservation: Final = await async_admit_io_token_turn(usage, deployment, _TURN_AT)
    await async_io_token_reconcile_success(usage, _proxy_success_kwargs(reservation), _reply_using(5, 2))

    assert usage.snapshot() == {
        "global_router:dep-1:bedrock/claude:itpm:12-34": 5,
        "global_router:dep-1:bedrock/claude:otpm:12-34": 2,
    }


def _otpm_limited_deployment() -> dict[str, object]:
    return {
        "litellm_params": {"model": MANTLE_OPUS, "otpm": 50},
        "model_info": {"id": "oversized-cap-id"},
        "model_name": "opus",
    }


@pytest.mark.asyncio
async def test_an_output_cap_too_large_for_the_store_to_count_is_refused_instead_of_admitted() -> None:
    check: Final = ModelRateLimitingCheck(usage=_memory_usage())
    set_io_token_rate_limit_request_kwargs({"max_tokens": 2**63, "metadata": {}}, metadata_channel="metadata")

    with pytest.raises(litellm.RateLimitError, match="OTPM limit=50"):
        await check.async_pre_call_check(_otpm_limited_deployment())


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [True, False], ids=["async", "sync"])
async def test_an_output_cap_past_the_limit_is_refused_without_touching_what_was_counted(asynchronous: bool) -> None:
    usage: Final = _memory_usage()
    check: Final = ModelRateLimitingCheck(usage=usage)

    async def pre_call(max_tokens: int) -> None:
        set_io_token_rate_limit_request_kwargs({"max_tokens": max_tokens, "metadata": {}}, metadata_channel="metadata")
        if asynchronous:
            await check.async_pre_call_check(_otpm_limited_deployment())
            return
        check.pre_call_check(_otpm_limited_deployment())

    await pre_call(40)
    counted: Final = usage.snapshot()

    with pytest.raises(litellm.RateLimitError, match="OTPM limit=50"):
        await pre_call(2**63 - 1)

    assert usage.snapshot() == counted


def test_a_sync_output_cap_too_large_for_the_store_to_count_is_refused_instead_of_admitted() -> None:
    check: Final = ModelRateLimitingCheck(usage=_memory_usage())
    set_io_token_rate_limit_request_kwargs({"max_tokens": 2**63, "metadata": {}}, metadata_channel="metadata")

    with pytest.raises(litellm.RateLimitError, match="OTPM limit=50"):
        check.pre_call_check(_otpm_limited_deployment())
