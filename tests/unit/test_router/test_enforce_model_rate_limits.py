"""
Tests for enforce_model_rate_limits feature.

This feature allows users to enforce TPM/RPM limits set on model deployments
regardless of the routing strategy being used.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)

TPM_DEPLOYMENT = {
    "tpm": 1000,
    "litellm_params": {"model": "gpt-4"},
    "model_info": {"id": "replica-test-id"},
    "model_name": "test-model",
}


def _dual_cache_with_local_tpm(local_tpm: int, redis_cache: MagicMock | None) -> DualCache:
    dual_cache = DualCache(redis_cache=redis_cache)
    check = ModelRateLimitingCheck(dual_cache=dual_cache)
    now = litellm.utils.get_utc_datetime()
    for minute in (now, now + timedelta(minutes=1)):
        tpm_key, _ = check._get_cache_keys(TPM_DEPLOYMENT, minute.strftime("%H-%M"))
        dual_cache.set_cache(key=tpm_key, value=local_tpm, local_only=True)
    return dual_cache


class TestModelRateLimitingCheck:
    """Test the ModelRateLimitingCheck class directly."""

    def test_get_deployment_limits_from_top_level(self):
        """Test extracting limits from top-level deployment config."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "tpm": 1000,
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 1000
        assert rpm == 10

    def test_get_deployment_limits_from_litellm_params(self):
        """Test extracting limits from litellm_params."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4", "tpm": 2000, "rpm": 20},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 2000
        assert rpm == 20

    def test_get_deployment_limits_from_model_info(self):
        """Test extracting limits from model_info."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id", "tpm": 3000, "rpm": 30},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 3000
        assert rpm == 30

    def test_get_deployment_limits_none_when_not_set(self):
        """Test that None is returned when limits are not set."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm is None
        assert rpm is None

    def test_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded."""
        mock_cache = MagicMock()
        mock_cache.increment_cache.return_value = 11  # Over limit after increment

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "RPM limit=10" in str(exc_info.value)
        assert "current usage=11" in str(exc_info.value)

    def test_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit."""
        mock_cache = MagicMock()
        mock_cache.increment_cache.return_value = 6

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded."""
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = 1000  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "tpm": 1000,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "TPM limit=1000" in str(exc_info.value)
        assert "current usage=1000" in str(exc_info.value)

    def test_pre_call_check_rejects_when_shared_tpm_is_over_limit_but_local_is_under(self):
        redis_cache = MagicMock()
        redis_cache.get_cache.return_value = 1000
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.parametrize(
        "redis_get", [MagicMock(return_value=None), MagicMock(side_effect=RedisCircuitBreakerOpenError())]
    )
    def test_pre_call_check_keeps_rejecting_on_local_usage_when_redis_read_fails(self, redis_get):
        redis_cache = MagicMock()
        redis_cache.get_cache = redis_get
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(1000, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    def test_pre_call_check_falls_back_to_local_tpm_and_still_checks_rpm_when_redis_circuit_is_open(self):
        redis_cache = MagicMock()
        redis_cache.get_cache.side_effect = RedisCircuitBreakerOpenError()
        redis_cache.increment_cache.return_value = 2
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check({**TPM_DEPLOYMENT, "rpm": 1})

        assert "RPM limit=1" in str(exc_info.value)

    def test_pre_call_check_without_redis_enforces_local_tpm_and_rpm(self):
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, None))
        deployment = {**TPM_DEPLOYMENT, "rpm": 1}

        assert check.pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "RPM limit=1" in str(exc_info.value)

    def test_log_success_event_increments_cache(self):
        """Test that log_success_event correctly increments the cache."""
        mock_cache = MagicMock()
        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        kwargs = {
            "standard_logging_object": {
                "model_id": "test-id",
                "total_tokens": 50,
                "hidden_params": {"litellm_model_name": "gpt-4"},
            }
        }

        check.log_success_event(kwargs, None, None, None)

        # Verify increment_cache was called
        mock_cache.increment_cache.assert_called_once()
        _, kwarg_params = mock_cache.increment_cache.call_args
        assert "test-id:gpt-4:tpm:" in kwarg_params["key"]
        assert kwarg_params["value"] == 50


class TestModelRateLimitingCheckAsync:
    """Test async methods of ModelRateLimitingCheck."""

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_increment_cache = AsyncMock(return_value=11)  # Over limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "RPM limit=10" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_increment_cache = AsyncMock(return_value=6)

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=1000)  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "tpm": 1000,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "TPM limit=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_rejects_when_shared_tpm_is_over_limit_but_local_is_under(self):
        redis_cache = MagicMock()
        redis_cache.async_get_cache = AsyncMock(return_value=1000)
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "redis_get", [AsyncMock(return_value=None), AsyncMock(side_effect=RedisCircuitBreakerOpenError())]
    )
    async def test_async_pre_call_check_keeps_rejecting_on_local_usage_when_redis_read_fails(self, redis_get):
        redis_cache = MagicMock()
        redis_cache.async_get_cache = redis_get
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(1000, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(TPM_DEPLOYMENT)

        assert "current usage=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_falls_back_to_local_tpm_and_still_checks_rpm_when_redis_circuit_is_open(
        self,
    ):
        redis_cache = MagicMock()
        redis_cache.async_get_cache = AsyncMock(side_effect=RedisCircuitBreakerOpenError())
        redis_cache.async_increment = AsyncMock(return_value=2)
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, redis_cache))

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check({**TPM_DEPLOYMENT, "rpm": 1})

        assert "RPM limit=1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_without_redis_enforces_local_tpm_and_rpm(self):
        check = ModelRateLimitingCheck(dual_cache=_dual_cache_with_local_tpm(5, None))
        deployment = {**TPM_DEPLOYMENT, "rpm": 1}

        assert await check.async_pre_call_check(deployment) == deployment
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "RPM limit=1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_log_success_event_increments_cache(self):
        """Test that async_log_success_event correctly increments the cache."""
        mock_cache = MagicMock()
        mock_cache.async_increment_cache = AsyncMock()
        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        kwargs = {
            "standard_logging_object": {
                "model_id": "test-id",
                "total_tokens": 50,
                "hidden_params": {"litellm_model_name": "gpt-4"},
            }
        }

        await check.async_log_success_event(kwargs, None, None, None)

        # Verify async_increment_cache was called
        mock_cache.async_increment_cache.assert_called_once()
        _, kwarg_params = mock_cache.async_increment_cache.call_args
        assert "test-id:gpt-4:tpm:" in kwarg_params["key"]
        assert kwarg_params["value"] == 50


class TestRouterWithEnforceModelRateLimits:
    """Test Router integration with enforce_model_rate_limits."""

    def test_router_initializes_with_enforce_model_rate_limits(self):
        """Test that Router properly initializes the ModelRateLimitingCheck."""
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test"},
                "rpm": 10,
            }
        ]

        router = Router(
            model_list=model_list,
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )

        # Check that the callback was added
        assert router.optional_callbacks is not None
        assert len(router.optional_callbacks) == 1
        assert isinstance(router.optional_callbacks[0], ModelRateLimitingCheck)

    def test_router_optional_callbacks_contains_model_rate_limiting(self):
        """Test that ModelRateLimitingCheck is in the callbacks list."""
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test"},
                "rpm": 10,
            }
        ]

        Router(
            model_list=model_list,
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )

        # Find the ModelRateLimitingCheck in litellm.callbacks
        found = False
        for callback in litellm.callbacks:
            if isinstance(callback, ModelRateLimitingCheck):
                found = True
                break

        assert found, "ModelRateLimitingCheck should be in litellm.callbacks"


class TestModelRateLimitConcurrency:
    """Test that RPM rate limiting is atomic under concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_rpm_limit(self):
        """
        Fire 4 concurrent async requests with RPM limit of 2.
        Exactly 2 should succeed and 2 should raise RateLimitError.

        This test validates the atomic increment-first pattern:
        the old check-then-increment pattern would let 3+ through
        due to a race condition on the local cache read.
        """
        dual_cache = DualCache()
        check = ModelRateLimitingCheck(dual_cache=dual_cache)

        deployment = {
            "rpm": 2,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "concurrent-test-id"},
            "model_name": "test-model",
        }

        async def attempt_request():
            return await check.async_pre_call_check(deployment)

        results = await asyncio.gather(
            *[attempt_request() for _ in range(4)],
            return_exceptions=True,
        )

        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, litellm.RateLimitError)]

        assert len(successes) == 2, f"Expected 2 successes, got {len(successes)}"
        assert len(failures) == 2, f"Expected 2 rate limit errors, got {len(failures)}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit,answer",
    [
        pytest.param(
            {"tpm": 1000}, {"finishReason": "IMAGE_PROHIBITED_CONTENT", "content": {"role": "model"}}, id="tpm-refusal"
        ),
        pytest.param({"tpm": 1000}, {"finishReason": "STOP", "content": {"parts": [{"text": "a dot"}]}}, id="tpm-no_image"),
        pytest.param(
            {"itpm": 700}, {"finishReason": "IMAGE_PROHIBITED_CONTENT", "content": {"role": "model"}}, id="itpm-refusal"
        ),
        pytest.param(
            {"tpm": 1000, "input_cost_per_token": 0, "output_cost_per_token": 0},
            {"finishReason": "IMAGE_PROHIBITED_CONTENT", "content": {"role": "model"}},
            id="tpm-zero_priced_refusal",
        ),
    ],
)
async def test_an_image_answer_without_an_image_counts_its_answered_tokens_toward_the_deployment_limits(
    limit: dict[str, int], answer: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[
            {
                "model_name": "image",
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-image",
                    "api_key": "test-key",
                    "api_base": "https://image.invalid/v1",
                    **limit,
                },
                "model_info": {"id": "image"},
            }
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=0,
    )
    billed: Final = {"promptTokenCount": 777, "candidatesTokenCount": 333, "totalTokenCount": 1110}

    try:
        with respx.mock(assert_all_called=False) as upstream:
            route = upstream.post(url__startswith="https://").mock(
                return_value=httpx.Response(200, json={"candidates": [answer], "usageMetadata": billed})
            )
            with pytest.raises((litellm.ContentPolicyViolationError, litellm.UnprocessableEntityError)):
                await router.aimage_generation(model="image", prompt="a single small red dot")
            with pytest.raises(litellm.RateLimitError, match="Model rate limit exceeded"):
                await router.aimage_generation(model="image", prompt="a single small red dot")
    finally:
        router.discard()

    assert route.call_count == 1


_IMAGE_REFUSAL: Final = {"finishReason": "IMAGE_PROHIBITED_CONTENT", "content": {"role": "model"}}
_BILLED_REFUSAL_USAGE: Final = {"promptTokenCount": 777, "candidatesTokenCount": 333, "totalTokenCount": 1110}


def _gemini_image_deployment(name: str, limit: dict[str, int]) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "gemini/gemini-3.1-flash-image",
            "api_key": "test-key",
            "api_base": f"https://{name}.invalid/v1",
            **limit,
        },
        "model_info": {"id": name},
    }


@pytest.mark.parametrize("limit", [pytest.param({"tpm": 1000}, id="tpm"), pytest.param({"itpm": 700}, id="itpm")])
def test_a_sync_image_refusal_counts_its_answered_tokens_toward_the_deployment_limits(
    limit: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[_gemini_image_deployment("image", limit)],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=0,
    )

    try:
        with respx.mock(assert_all_called=False) as upstream:
            route = upstream.post(url__startswith="https://image.invalid").mock(
                return_value=httpx.Response(200, json={"candidates": [_IMAGE_REFUSAL], "usageMetadata": _BILLED_REFUSAL_USAGE})
            )
            with pytest.raises(litellm.ContentPolicyViolationError):
                router.image_generation(model="image", prompt="a single small red dot")
            with pytest.raises(litellm.RateLimitError, match="Model rate limit exceeded"):
                router.image_generation(model="image", prompt="a single small red dot")
    finally:
        router.discard()

    assert route.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [pytest.param({"tpm": 1000}, id="tpm"), pytest.param({"itpm": 700}, id="itpm")])
async def test_an_image_refusal_from_a_fallback_counts_its_answered_tokens_toward_the_fallback_limits(
    limit: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[_gemini_image_deployment("outage", {}), _gemini_image_deployment("image", limit)],
        fallbacks=[{"outage": ["image"]}],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=0,
    )

    logging_obj, request = litellm.utils.function_setup(
        original_function="aimage_generation",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        model="outage",
        prompt="a single small red dot",
        litellm_call_id="image-refusal-from-a-fallback",
    )

    try:
        with respx.mock(assert_all_called=False) as upstream:
            upstream.post(url__startswith="https://outage.invalid").mock(
                return_value=httpx.Response(500, json={"error": {"code": 500, "message": "scripted outage"}})
            )
            refusing = upstream.post(url__startswith="https://image.invalid").mock(
                return_value=httpx.Response(200, json={"candidates": [_IMAGE_REFUSAL], "usageMetadata": _BILLED_REFUSAL_USAGE})
            )
            with pytest.raises(litellm.ContentPolicyViolationError):
                await router.aimage_generation(**request, litellm_logging_obj=logging_obj)
            with pytest.raises(litellm.RateLimitError, match="Model rate limit exceeded"):
                await router.aimage_generation(model="image", prompt="a single small red dot")
    finally:
        router.discard()

    assert refusing.call_count == 1


@pytest.mark.asyncio
async def test_an_async_image_edit_refusal_counts_its_answered_tokens_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[_gemini_image_deployment("image", {"tpm": 2000})],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=0,
    )
    image: Final = tmp_path / "dot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    try:
        with respx.mock(assert_all_called=False) as upstream:
            route = upstream.post(url__startswith="https://image.invalid").mock(
                return_value=httpx.Response(200, json={"candidates": [_IMAGE_REFUSAL], "usageMetadata": _BILLED_REFUSAL_USAGE})
            )
            for _ in range(2):
                with image.open("rb") as file, pytest.raises(litellm.ContentPolicyViolationError):
                    await router.aimage_edit(model="image", image=file, prompt="make it blue")
            with image.open("rb") as file, pytest.raises(litellm.RateLimitError, match="current usage=2220"):
                await router.aimage_edit(model="image", image=file, prompt="make it blue")
    finally:
        router.discard()

    assert route.call_count == 2


_SERVED_IMAGE: Final = {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}]}}
_SERVED_USAGE: Final = {"promptTokenCount": 200, "candidatesTokenCount": 100, "totalTokenCount": 300}


async def _served_image_from(router: Router, **request: object) -> None:
    await router.aimage_generation(**request)
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


@pytest.mark.asyncio
async def test_a_fallback_serving_a_refused_image_counts_only_its_own_tokens_toward_its_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = Router(
        model_list=[_gemini_image_deployment("refuser", {}), _gemini_image_deployment("backup", {"tpm": 599})],
        fallbacks=[{"refuser": ["backup"]}],
        optional_pre_call_checks=["enforce_model_rate_limits"],
        num_retries=0,
    )
    logging_obj, request = litellm.utils.function_setup(
        original_function="aimage_generation",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        model="refuser",
        prompt="a single small red dot",
        litellm_call_id="image-refused-then-served-by-a-fallback",
    )

    try:
        with respx.mock(assert_all_called=False) as upstream:
            upstream.post(url__startswith="https://refuser.invalid").mock(
                return_value=httpx.Response(200, json={"candidates": [_IMAGE_REFUSAL], "usageMetadata": _BILLED_REFUSAL_USAGE})
            )
            serving = upstream.post(url__startswith="https://backup.invalid").mock(
                return_value=httpx.Response(200, json={"candidates": [_SERVED_IMAGE], "usageMetadata": _SERVED_USAGE})
            )
            await _served_image_from(router, **request, litellm_logging_obj=logging_obj)
            await _served_image_from(router, model="backup", prompt="a single small red dot")
            with pytest.raises(litellm.RateLimitError, match="current usage=600"):
                await router.aimage_generation(model="backup", prompt="a single small red dot")
    finally:
        router.discard()

    assert serving.call_count == 2
